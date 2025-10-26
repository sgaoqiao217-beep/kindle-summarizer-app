import io
import os
import re
import json
import tempfile
import unicodedata
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional, Set
from pdf_utils import register_jp_font, build_pdf_bytes

import streamlit as st
from PIL import Image
from dotenv import load_dotenv

try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None

try:
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
except Exception:
    build = None
    MediaIoBaseDownload = None
    MediaIoBaseUpload = None

try:
    from google_api_utils import get_credentials as get_google_credentials
except Exception:
    get_google_credentials = None
# --- Fallback for google_api_utils が無い環境 ---
if get_google_credentials is None:
    import json
    from google.oauth2 import service_account
    # googleapiclient.build は app.py 先頭で try-import 済み（build が None の可能性あり）

    def get_google_credentials():
        """Secretsに入れたサービスアカウントJSONからDocs/Drive権限でCredentialsを作る"""
        raw = st.secrets["GOOGLE_CREDENTIALS"]  # secrets.toml に JSON 全文を入れる運用
        info = json.loads(raw) if isinstance(raw, str) else raw
        scopes = [
            "https://www.googleapis.com/auth/documents",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = service_account.Credentials.from_service_account_info(info, scopes=scopes)
        return creds


# --- Fallback ここまで ---

# Optional: 既存モジュールがあれば使う（無ければImportErrorを握りつぶしてフォールバック）
try:
    from sort_image_file import normalize_filenames as normalize_filenames_external
except Exception:
    normalize_filenames_external = None

try:
    from ocr_utils import extract_info_type1 as extract_info_external
    from ocr_utils import _vision_client_from_secrets 
    # ↑ ユーザ環境の関数名に合わせてお好みで
except Exception:
    extract_info_external = None


from google.oauth2 import service_account

load_dotenv()

FONT_NAME = "NotoSansJP"
FONT_PATH = os.path.join("assets", "NotoSansCJKjp-Regular.otf")

# 句点や括弧で段落/文末を判断するための安全な終端記号セット
_JP_SENT_END = "。．！？!?" + "」』）】］》〉"
# 文頭に来やすい始まり記号（次行がこれで始まるなら改行を残す）
_JP_SENT_BEGIN = "「『（【［《〈"

# =========================================================
# ★ 追加：要約前クリーニング（見出し修正・レンジ行削除・文中改行の解消）
# =========================================================
def clean_presummary_text(text: str) -> str:
    """
    目的:
      1) 見出し 'Part 01.doc' → 'Part01' に置換（.doc削除 & 半角スペース除去、ゼロパディング維持）
      2) 'Part 01 (1–9,278)' のようなページレンジ行の完全削除
      3) 文章途中の不要改行を除去（段落の空行は保持）
    """
    if not isinstance(text, str):
        return text

    # 改行統一
    t = text.replace("\r\n", "\n").replace("\r", "\n")

    # 行ごとに前処理（1,2を適用）
    lines = t.splitlines()
    cleaned_lines = []
    for ln in lines:
        # 2) 'Part 01 (1–9,278)' を削除（ハイフン/エンダッシュ・カンマ桁区切りに対応）
        if re.fullmatch(
            r"\s*Part\s*\d+\s*\(\s*\d{1,3}(?:,\d{3})*\s*[–-]\s*\d{1,3}(?:,\d{3})*\s*\)\s*\.*\s*",
            ln
        ):
            continue

        # 1) 'Part 01.doc' → 'Part01'
        m = re.fullmatch(r"\s*Part\s*([0-9]{1,3})\s*\.?\s*[dD][oO][cC]\s*", ln)
        if m:
            num = m.group(1).zfill(2)
            cleaned_lines.append(f"Part{num}")
            continue

        cleaned_lines.append(ln)

    t = "\n".join(cleaned_lines).strip("\n")

    # 3) 段落内の“文中改行”を削除し、空行での段落区切りは保持
    paragraphs = re.split(r"\n{2,}", t)

    def join_soft_wraps(p: str) -> str:
        lines = p.split("\n")
        if len(lines) == 1:
            return lines[0].strip()

        buf = []
        cur = lines[0].strip()

        for nxt in lines[1:]:
            nxt_stripped = nxt.strip()
            if not nxt_stripped:
                # 段落内に空行が紛れていたら改行を維持
                buf.append(cur)
                cur = ""
                continue

            # 直前が文末記号 / 次行頭が開き記号 → 改行維持
            keep_newline = (
                len(cur) > 0 and cur[-1] in _JP_SENT_END
            ) or (
                len(nxt_stripped) > 0 and nxt_stripped[0] in _JP_SENT_BEGIN
            )

            if keep_newline:
                buf.append(cur)
                cur = nxt_stripped
            else:
                # 文途中の改行は結合（英数どうしのみ半角スペースを挿入）
                if (cur and nxt_stripped and cur[-1].isascii() and nxt_stripped[0].isascii()):
                    cur = cur + " " + nxt_stripped
                else:
                    cur = cur + nxt_stripped

        if cur:
            buf.append(cur)

        return "\n".join(buf)

    joined = [join_soft_wraps(p) for p in paragraphs if p.strip()]
    return "\n\n".join(joined)

format_prompt = """
You are given raw text from a Japanese book (a novel or story).
Your task is ONLY to format the text into a clean book-like structure.

Rules:
1. Insert one line break between paragraphs.
2. At the beginning of each paragraph, insert one full-width space character (全角スペース1文字分).
3. Remove unnecessary line breaks inside sentences.
4. If there are unnatural word breaks or typos caused by OCR, fix them using context.
5. Do NOT summarize or shorten the content. Keep all original content.

最終出力は本文の整形後テキストのみを出力してください。
余計な説明や注釈は一切不要です。
"""

summary_prompt_template = """
You are given a chapter from a Japanese novel or story (already formatted).
Your task is to summarize it into a polished, story-like text.

Use the following chapter title exactly as provided:
{chapter_title}

**Output structure (must follow exactly):**

1. First line: the chapter title as an H2 heading.
   * If Google Docs: apply **HEADING_2** style to the title line.
2. One blank line.
3. The summary body text (Japanese), 3000–4000 characters, formatted per the rules below.

**Critical constraints (do not violate):**

* Do **not** stop after writing the heading. Writing only the heading is invalid.
* Always include the full summary body after the blank line.
* If the input lacks a clear chapter title, infer a concise title from the content (e.g., main topic or scene) and still output it as H2.

**Rules for the summary body:**

1. Length: **3000–4000 Japanese characters** (strict).
2. Include all key events, characters, and emotional flow.
3. Ensure the writing is natural, grammatically correct, and coherent.
4. Preserve the original style, tone, and atmosphere.
5. Japanese book-style formatting:

   * Insert **one line break between paragraphs**.
   * At the beginning of each paragraph, insert **one full-width space character**（全角スペース1文字分）.
   * Remove unnecessary line breaks inside sentences.

**Output requirements:**

* 最終出力は必ず日本語で書き、整理された完成版のみを出力してください。
* 字数は必ず3000字以上4000字以内にしてください。
* 出力は本文の内容のみとし、余計な説明や注釈、注意書き、メタコメントは一切付けないでください。
"""

# =========================
# LLM
# =========================
def _is_llm_available() -> bool:
    return (
        genai is not None
        and types is not None
        and bool(os.getenv("GEMINI_API_KEY"))
        and bool(os.getenv("GEMINI_MODEL"))
    )

def call_model():
    if genai is None or types is None:
        raise ImportError("google-genai がインストールされていません。`pip install google-genai` を実行してください。")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("'GEMINI_API_KEY' setting is missing in environment variables.")

    client = genai.Client(api_key=api_key)
    model = os.getenv("GEMINI_MODEL")
    if not model:
        raise ValueError("'GEMINI_MODEL' setting is missing in environment variables.")
    return client, model

def format_text(raw_text: str) -> str:
    client, model = call_model()
    full_prompt = f"{format_prompt}\n\nHere is the raw text:\n{raw_text}"
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=full_prompt)],
        )
    ]
    response = client.models.generate_content(model=model, contents=contents)
    return response.text

def summarize_text(formatted_text: str, chapter_title: str) -> str:
    client, model = call_model()
    summary_prompt = summary_prompt_template.format(chapter_title=chapter_title)
    full_prompt = f"{summary_prompt}\n\nHere is the formatted text:\n{formatted_text}"
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=full_prompt)],
        )
    ]
    response = client.models.generate_content(model=model, contents=contents)
    response_text = response.text.strip()

    heading_line = f"## {chapter_title}".strip()
    lines = response_text.splitlines()
    if lines:
        first_line = lines[0].strip()
        if first_line.startswith("##"):
            lines[0] = heading_line
        else:
            lines.insert(0, heading_line)
    else:
        lines = [heading_line]

    body_lines = [line.rstrip() for line in lines[1:]]

    while body_lines and not body_lines[0].strip():
        body_lines.pop(0)
    while body_lines and not body_lines[-1].strip():
        body_lines.pop()

    normalized_body = "\n".join(body_lines)
    if normalized_body:
        final_text = f"{heading_line}\n\n{normalized_body}"
    else:
        final_text = heading_line

    return final_text


def _ensure_folder_path(drive_service, parts: List[str]) -> Optional[str]:
    """['OCR結果','書籍タイトル'] のようなパスをDrive上に作成して親IDを返す。"""
    parent_id = None
    for name in parts:
        safe = (name or "").replace("'", r"\'")
        q = "mimeType='application/vnd.google-apps.folder' and name='%s'" % safe
        if parent_id:
            q += f" and '{parent_id}' in parents"
        res = drive_service.files().list(q=q, fields="files(id,name)", pageSize=1).execute()
        items = res.get("files", [])
        if items:
            parent_id = items[0]["id"]
            continue
        meta = {"name": name, "mimeType": "application/vnd.google-apps.folder"}
        if parent_id:
            meta["parents"] = [parent_id]
        created = drive_service.files().create(body=meta, fields="id").execute()
        parent_id = created["id"]
    return parent_id

def _prepare_pdf_sections(sections: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    PDF生成向けに (heading, body) タプル配列をサニタイズ。
    先頭行が '## heading' の形式になっている要約テキストは除去して本文だけにする。
    """
    prepared: List[Tuple[str, str]] = []
    for raw_heading, raw_body in sections:
        heading = (raw_heading or "無題").strip() or "無題"
        body = (raw_body or "").strip()
        if body.startswith("##"):
            body_lines = body.splitlines()
            expected = f"## {heading}"
            if body_lines and body_lines[0].strip() == expected:
                body_lines = body_lines[1:]
                while body_lines and not body_lines[0].strip():
                    body_lines.pop(0)
                body = "\n".join(body_lines).strip()
        prepared.append((heading, body))
    return prepared


def _summaries_as_sections(
    chapters: List[Tuple[str, str]],
    summaries: Dict[str, str],
) -> List[Tuple[str, str]]:
    """
    章リストに沿って要約dictを並べ替え、欠損はスキップしつつ残りを末尾に追加。
    """
    sections: List[Tuple[str, str]] = []
    seen: Set[str] = set()
    for heading, _ in chapters:
        summary = summaries.get(heading)
        if summary:
            sections.append((heading, summary))
            seen.add(heading)
    for heading, summary in summaries.items():
        if heading in seen:
            continue
        sections.append((heading, summary))
    return sections

def _get_drive_service(creds):
    if build is None:
        raise ImportError("googleapiclient がインストールされていません。`pip install google-api-python-client` を実行してください。")
    return build("drive", "v3", credentials=creds)

def ensure_pdf_font_registered(font_name: str, font_path: str) -> Tuple[bool, Optional[str]]:
    """
    ReportLab に日本語フォントを登録。成功可否とエラーメッセージ（あれば）を返す。
    """
    if st.session_state.get("pdf_font_registered"):
        return True, None
    if not os.path.exists(font_path):
        return False, f"フォントファイルが見つかりません: {font_path}"
    try:
        register_jp_font(font_name, font_path)
    except RuntimeError as e:
        return False, str(e)
    st.session_state.pdf_font_registered = True
    return True, None


def _upload_pdf_to_drive(
    creds,
    *,
    file_name: str,
    file_bytes: bytes,
    parent_folder_id: Optional[str],
):
    """
    PDFファイルを Google Drive にアップロードし、file metadata を返す。
    """
    if build is None or MediaIoBaseUpload is None:
        raise ImportError("googleapiclient がインストールされていません。`pip install google-api-python-client` を実行してください。")
    service = build("drive", "v3", credentials=creds)
    media = MediaIoBaseUpload(io.BytesIO(file_bytes), mimetype="application/pdf", resumable=False)
    metadata: Dict[str, Any] = {"name": file_name, "mimeType": "application/pdf"}
    if parent_folder_id:
        metadata["parents"] = [parent_folder_id]
    return service.files().create(body=metadata, media_body=media, fields="id, webViewLink").execute()


def _get_cached_google_credentials():
    # ここで get_google_credentials が None の可能性はない（上のフォールバックで定義済み）
    creds = st.session_state.get("google_creds")
    if creds is None:
        creds = get_google_credentials()
        st.session_state.google_creds = creds
    return creds


def normalize_drive_folder_input(raw_value: str) -> str:
    """DriveのフォルダURLから folderId を抽出。IDならそのまま返す。"""
    if not raw_value:
        return raw_value
    raw_value = raw_value.strip()
    match = re.search(r"/folders/([a-zA-Z0-9_-]+)", raw_value)
    if match:
        return match.group(1)
    match = re.search(r"id=([a-zA-Z0-9_-]+)", raw_value)
    if match:
        return match.group(1)
    return raw_value

def _list_drive_images(creds, folder_id: str) -> List[Dict[str, str]]:
    if build is None:
        raise ImportError("googleapiclient がインストールされていません。`pip install google-api-python-client` を実行してください。")
    service = build("drive", "v3", credentials=creds)
    query = f"'{folder_id}' in parents and mimeType contains 'image/' and trashed = false"
    fields = "nextPageToken, files(id, name, mimeType, modifiedTime)"
    page_token = None
    files: List[Dict[str, str]] = []
    while True:
        response = service.files().list(
            q=query,
            spaces="drive",
            fields=fields,
            orderBy="name_natural",
            pageToken=page_token,
        ).execute()
        files.extend(response.get("files", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            break
    return files

def _download_drive_images(creds, file_entries: List[Dict[str, str]], dest_dir: str) -> List[str]:
    if build is None or MediaIoBaseDownload is None:
        raise ImportError("googleapiclient がインストールされていません。`pip install google-api-python-client` を実行してください。")
    os.makedirs(dest_dir, exist_ok=True)
    service = build("drive", "v3", credentials=creds)
    saved_paths: List[str] = []
    for entry in file_entries:
        file_id = entry.get("id")
        filename = entry.get("name") or f"{file_id}.img"
        base_name, ext = os.path.splitext(filename)
        ext = ext if ext else ".jpg"
        safe_base = re.sub(r"[^\w\-.ぁ-んァ-ン一-龥]", "_", base_name)[:100]
        safe_name = f"{safe_base}{ext}"
        dest_path = os.path.join(dest_dir, safe_name)
        counter = 1
        while os.path.exists(dest_path):
            dest_path = os.path.join(dest_dir, f"{safe_base}_{counter}{ext}")
            counter += 1

        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        with open(dest_path, "wb") as f:
            f.write(fh.getvalue())
        saved_paths.append(dest_path)
    return saved_paths

# =========================
# Utility
# =========================
def save_uploaded_images(files, workdir: str) -> List[str]:
    os.makedirs(workdir, exist_ok=True)
    saved = []
    for f in files:
        # iOSのlive photo拡張子対策含む拡張子正規化
        suffix = os.path.splitext(f.name)[1].lower()
        suffix = ".jpg" if suffix in {".jpeg", ".jpg"} else ".png" if suffix in {".png"} else ".jpg"
        # タイムスタンプ＋元名で衝突回避
        fname = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{os.path.basename(f.name)}"
        path = os.path.join(workdir, fname)
        # PIL経由で画像だけ保存（HEIC等は別途ライブラリ必要）
        img = Image.open(f).convert("RGB")
        img.save(path if suffix == ".jpg" else path.replace(".jpg", ".png"))
        saved.append(path if suffix == ".jpg" else path.replace(".jpg", ".png"))
    return saved

def normalize_filenames_local(paths: List[str]) -> List[str]:
    """
    名前中の 'スクリーンショット YYYY-MM-DD HH.MM.SS [連番?]' を並べ替え。
    ファイル名が上記規則でない場合はmtimeでソート。
    """
    def parse_key(p: str) -> Tuple[int, str]:
        base = os.path.splitext(os.path.basename(p))[0]
        m = re.match(r"(スクリーンショット (\d{4}-\d{2}-\d{2}) (\d{2}\.\d{2}\.\d{2}))(?: (\d+))?$", base)
        if m:
            dt = f"{m.group(2)} {m.group(3).replace('.',':')}"
            num = int(m.group(4)) if m.group(4) else 0
            try:
                ts = int(datetime.strptime(dt, "%Y-%m-%d %H:%M:%S").timestamp())
            except ValueError:
                ts = int(os.path.getmtime(p))
            return (ts * 100 + num, p)
        else:
            return (int(os.path.getmtime(p)) * 100, p)

    sorted_paths = sorted(paths, key=parse_key)
    return sorted_paths

# OCR結果dictから本文を堅牢に抜き出す
def _pick_text(d: Dict[str, Any]) -> str:
    """OCR/外部抽出の結果 dict から本文文字列を堅牢に取り出す"""
    if not isinstance(d, dict):
        return ""
    for k in ("Text", "text", "Body", "body", "FullText", "full_text", "content"):
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v
    # Google Vision の raw をそのまま入れた場合の保険
    v = d.get("full_text_annotation", {}).get("text") if isinstance(d.get("full_text_annotation"), dict) else None
    return v.strip() if isinstance(v, str) else ""

# =========================
# OCR
# =========================
@st.cache_resource(show_spinner=False)
# def get_vision_client(json_key_path: Optional[str] = None):
#     # json_key_pathが指定されていれば一時的に環境変数を差し替え（セッション存続中のみ）
#     if json_key_path:
#         os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = json_key_path
#     from google.cloud import vision
#     credentials_info = json.loads(st.secrets["GOOGLE_CREDENTIALS"])
#     st.write(credentials_info)
#     credentials = service_account.Credentials.from_service_account_info(credentials_info)
#     client = vision.ImageAnnotatorClient(credentials=credentials)
#     # client = _vision_client_from_secrets(credentials=credentials)
    
#     return client

def get_vision_client(json_key_path: Optional[str] = None):
    if json_key_path:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = json_key_path

    # ← 追加（環境変数で quota project が勝手に付くのを防ぐ）
    for k in ("GOOGLE_CLOUD_QUOTA_PROJECT", "GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT"):
        os.environ.pop(k, None)

    from google.cloud import vision
    import json
    from google.oauth2 import service_account

    raw = st.secrets["GOOGLE_CREDENTIALS"]
    credentials_info = json.loads(raw) if isinstance(raw, str) else raw

    # スコープを明示して作る（推奨）
    scopes = ["https://www.googleapis.com/auth/cloud-platform"]
    credentials = service_account.Credentials.from_service_account_info(credentials_info, scopes=scopes)

    # ★ ここで with_quota_project は付けない（今は不要）
    # credentials = credentials.with_quota_project("sunny-advantage-471612-v1")

    client = vision.ImageAnnotatorClient(credentials=credentials)
    return client


def _extract_with_vision(img_path: str, client) -> Dict[str, Any]:
    from google.cloud import vision
    with open(img_path, "rb") as f:
        content = f.read()
    image = vision.Image(content=content)
    # 和書の縦書きを含む日本語ヒント
    context = vision.ImageContext(language_hints=["ja"])
    resp = client.document_text_detection(image=image, image_context=context)
    if resp.error.message:
        raise RuntimeError(resp.error.message)
    return resp

def fix_broken_chapter_tokens(text: str) -> str:
    """
    OCRで「第 1 章」「プ ロ ロ ー グ」などに割れたトークンを修復
    """
    t = text
    # 全角/半角統一
    t = unicodedata.normalize("NFKC", t)
    # 連続空白の除去（ただし改行は残す）
    t = re.sub(r"[ \t\u3000]+", " ", t)
    # 「第 X 章」パターンの隙間を潰す
    t = re.sub(r"第\s*([0-9０-９一二三四五六七八九十百千〇零]+)\s*章", r"第\1章", t)
    # プロローグ/エピローグ/序章/終章/結論
    for token in ["プロローグ", "エピローグ", "序章", "終章", "結論", "あとがき"]:
        t = re.sub(r"(" + r"\s*".join(list(token)) + r")", token, t)
    return t

# 章タイトル（行頭限定）検出（残しているが現行は使わない）
CHAPTER_RE = re.compile(
    r"^(?P<heading>"
    r"第(?:[1-9][0-9]*|[０-９]+|[一二三四五六七八九十百千〇零]+)章(?:[ \t\u3000：:\-・][^\n]*)?"
    r"|(?:プロローグ|序章|終章|エピローグ|結論|あとがき)(?:[ \t\u3000：:\-・][^\n]*)?"
    r")$",
    re.MULTILINE,
)

def split_by_chapter_linehead(text: str) -> List[Tuple[str, str]]:
    """
    章見出しを行頭限定で分割（本文中の「第X章」には反応しない）
    """
    text = fix_broken_chapter_tokens(text)
    parts: List[Tuple[str, str]] = []
    current_title = None
    buff: List[str] = []
    for line in text.splitlines():
        m = CHAPTER_RE.match(line.strip())
        if m:
            if current_title and buff:
                parts.append((current_title, "\n".join(buff).strip()))
            current_title = m.group("heading").strip()
            buff = []
        else:
            buff.append(line)
    if current_title and buff:
        parts.append((current_title, "\n".join(buff).strip()))
    return parts if parts else [("本文", text.strip())]

def split_by_fixed_chars(text: str, size: int = 10000) -> List[Tuple[str, str]]:
    """
    本文を固定文字数 size ごとに分割する。
    末尾でちょうど切れない場合は、できるだけ近い段落境界（\n\n）まで戻して切る。
    それでも見つからなければそのまま size で切る。
    返り値は [(タイトル, 本文)] のタプル配列（後工程互換のためタイトルを付与）。
    """
    t = text or ""
    n = len(t)
    parts: List[Tuple[str, str]] = []
    i = 0
    idx = 1
    while i < n:
        end = min(i + size, n)
        # できれば段落区切りで切る（後方 1200 文字以内を探索）
        if end < n:
            window_start = max(i, end - 1200)
            cut_pos = t.rfind("\n\n", window_start, end)
            if cut_pos != -1 and cut_pos > i + 1000:  # あまり手前で切れすぎないように軽いガード
                end = cut_pos
        chunk = t[i:end].strip()
        title = f"Part {idx:02d} ({i+1:,}–{end:,})"
        parts.append((title, chunk))
        i = end
        idx += 1
    return parts if parts else [("Part 01", t)]

def simple_info_extractor(full_text: str) -> Dict[str, Optional[str]]:
    """
    フォールバック用の簡易info抽出（Right/Leftをテキストから拾う）
    """
    t = unicodedata.normalize("NFKC", full_text)
    right = None
    m = re.search(r"(\d{1,3})\s*%", t)
    if m:
        right = f"{m.group(1)}%"

    left = None
    m2 = re.search(r"(本を読み終えるまで\d+分|[0-9０-９]+ページ中[0-9０-９]+ページ)", t)
    if m2:
        left = m2.group(1)

    return {"Title": None, "Subtitle": None, "Right": right, "Left": left}

def ocr_one_image(img_path: str, client) -> Dict[str, Any]:
    """
    1画像のOCR→info抽出→本文返却
    """
    if extract_info_external:
        # 外部実装の戻りが Body/Text など様々でも Text を必ず埋める
        res = extract_info_external(img_path)
        res = dict(res) if isinstance(res, dict) else {}
        text = _pick_text(res)
        res.setdefault("Filename", os.path.basename(img_path))
        res["Text"] = text
        return res

    # 無い場合は汎用フォールバック
    resp = _extract_with_vision(img_path, client)
    full_text = ""
    if getattr(resp, "full_text_annotation", None) and resp.full_text_annotation.text:
        full_text = resp.full_text_annotation.text
    elif getattr(resp, "text_annotations", None):
        full_text = resp.text_annotations[0].description

    info = simple_info_extractor(full_text)
    return {
        "Filename": os.path.basename(img_path),
        "Text": full_text,
        **info
    }

def best_effort_summarize(chapter_title: str, chapter_text: str) -> str:
    """
    LLMが利用可能なら整形→要約を実施し、不可なら素朴に短縮。
    """
    if _is_llm_available():
        try:
            formatted = format_text(chapter_text)
            return summarize_text(formatted, chapter_title)
        except Exception as e:
            st.warning(f"LLMによる要約処理でエラーが発生しました: {e}")
    # フォールバック：先頭をいい感じに要約風サマリ
    trimmed = re.sub(r"\s+", " ", chapter_text).strip()
    return trimmed[:1200] + ("…" if len(trimmed) > 1200 else "")

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Kindle書籍 自動要約ツール (MVP)", layout="wide")

st.title("📚 Kindle書籍 自動要約ツール")
st.caption("画像アップロード → 並べ替え → OCR → 章/固定長分割 → 要約 → Googleドキュメント出力 まで")

with st.sidebar:
    st.header("設定")
    st.write("Google Cloud 認証")
    cred_mode = st.radio("認証方法", ["環境変数を使う", "JSONをアップロード"], horizontal=True)
    uploaded_key = None
    if cred_mode == "JSONをアップロード":
        key_file = st.file_uploader("サービスアカウントのJSON", type=["json"])
        if key_file is not None:
            # 一時ファイルに保存
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
            tmp.write(key_file.read())
            tmp.flush()
            uploaded_key = tmp.name
            st.success("認証情報をメモリに読み込みました。")

    st.divider()
    st.write("要約の長さ（フォールバック時）")
    max_chars = st.slider("要約上限文字（フォールバック）", 600, 2000, 1200, 100)
    st.session_state.max_chars = max_chars

# セッション状態
if "workdir" not in st.session_state:
    st.session_state.workdir = tempfile.mkdtemp(prefix="kindle_ocr_")
if "images" not in st.session_state:
    st.session_state.images = []
if "ocr_results" not in st.session_state:
    st.session_state.ocr_results = []
if "full_text" not in st.session_state:
    st.session_state.full_text = ""
if "chapters" not in st.session_state:
    st.session_state.chapters = []
if "summaries" not in st.session_state:
    st.session_state.summaries = {}
if "drive_folder_input" not in st.session_state:
    st.session_state.drive_folder_input = ""
if "drive_loaded_folder_id" not in st.session_state:
    st.session_state.drive_loaded_folder_id = None
if "drive_files" not in st.session_state:
    st.session_state.drive_files = []
if "needs_chapter_split" not in st.session_state:
    st.session_state.needs_chapter_split = False

# Step 1: 画像アップロード
st.subheader("Step 1. 画像アップロード")
files = st.file_uploader("Kindleスクリーンショット（複数選択可）: JPG/PNG", type=["jpg","jpeg","png"], accept_multiple_files=True)
col1, col2 = st.columns([1,1])
with col1:
    if st.button("アップロードして保存", use_container_width=True) and files:
        saved = save_uploaded_images(files, st.session_state.workdir)
        st.session_state.images.extend(saved)
        st.success(f"{len(saved)}枚の画像を保存しました。")

with col2:
    if st.button("アップロード済み画像を表示", use_container_width=True):
        st.write(f"保存先: `{st.session_state.workdir}`")
        for p in st.session_state.images[:12]:
            st.image(p, width=220)
        if len(st.session_state.images) > 12:
            st.caption(f"…ほか {len(st.session_state.images) - 12} 枚")

with st.expander("Googleドライブから取得", expanded=False):
    st.write("Google Drive のフォルダから直接画像を読み込み、Step 1 に追加します。")
    drive_folder_value = st.text_input(
        "フォルダID または URL",
        key="drive_folder_input",
        placeholder="例: https://drive.google.com/drive/folders/xxxxxxxxxxxxxxxxx",
    )
    if st.button("フォルダ内の画像を読み込む", use_container_width=True):
        folder_id = normalize_drive_folder_input(drive_folder_value)
        if not folder_id:
            st.warning("フォルダIDまたはURLを入力してください。")
        else:
            if folder_id == st.session_state.get("drive_loaded_folder_id"):
                st.info("同じフォルダは既に読み込まれています。Step 2 に進んでください。")
            else:
                try:
                    creds = _get_cached_google_credentials()
                    with st.spinner("Google Drive から画像を取得中..."):
                        files = _list_drive_images(creds, folder_id)
                        st.session_state.drive_files = files
                        if not files:
                            st.info("指定したフォルダには画像ファイルが見つかりませんでした。")
                        else:
                            saved_paths = _download_drive_images(creds, files, st.session_state.workdir)
                            existing = set(st.session_state.images)
                            new_paths = [p for p in saved_paths if p not in existing]
                            if not new_paths:
                                st.info("新しい画像は追加されませんでした。")
                            else:
                                st.session_state.images.extend(new_paths)
                                st.session_state.drive_loaded_folder_id = folder_id
                                st.success(f"{len(new_paths)} 件の画像を追加しました。Step 2 で並び替えを実行してください。")
                except Exception as e:
                    st.error(f"フォルダの読み込みまたはダウンロードに失敗しました: {e}")

# Step 2: 並び替え（ファイル名/時刻ベース）
st.subheader("Step 2. ページ順に並び替え")
if st.session_state.images:
    colA, colB = st.columns([1,1])
    with colA:
        st.write("並び替え方式")
        how = st.radio("ルール", ["あなたの`normalize_filenames`を使用", "MVP内の簡易ルール"], horizontal=False)
    with colB:
        if st.button("並び替えを実行", use_container_width=True):
            if how == "あなたの`normalize_filenames`を使用" and normalize_filenames_external:
                sorted_list = normalize_filenames_external(st.session_state.workdir)
                # ↑ あなたの関数の返り値仕様に合わせて調整が必要な場合あり
                # ここではworkdir内をリネーム→再取得を想定
                st.session_state.images = [os.path.join(st.session_state.workdir, f) for f in os.listdir(st.session_state.workdir)]
                st.session_state.images = normalize_filenames_local(st.session_state.images)
            else:
                st.session_state.images = normalize_filenames_local(st.session_state.images)
            st.success("並び替え完了")
    st.caption("※ プロジェクトの命名規則が厳密に決まっている場合は、外部関数の呼び出しを優先してください。")

# Step 3: OCR & info抽出
st.subheader("Step 3. OCR & info抽出")
if st.session_state.images:
    client = get_vision_client(uploaded_key)
    if st.button("OCRを実行", type="primary", use_container_width=True):
        results = []
        prog = st.progress(0.0, text="OCR処理中…")
        for i, p in enumerate(st.session_state.images):
            try:
                info = ocr_one_image(p, client)
                info["Path"] = p
                results.append(info)
            except Exception as e:
                st.error(f"OCR失敗: {os.path.basename(p)} — {e}")
            prog.progress((i+1)/len(st.session_state.images), text=f"OCR処理中… {i+1}/{len(st.session_state.images)}")
        st.session_state.ocr_results = results

        # 1つの本文に連結（ページ間に改行挿入）— キー差異を吸収
        texts = []
        for r in results:
            t = _pick_text(r)
            if t:
                texts.append(t)
        combined_text = "\n\n".join(texts).strip()

        # ★ ここでクリーニングを適用（見出し修正/レンジ行削除/文中改行解消）
        cleaned_text = clean_presummary_text(combined_text)

        st.session_state.full_text = cleaned_text
        st.session_state.needs_chapter_split = bool(st.session_state.full_text)

        st.success(f"OCR完了：{len(results)}ページ")

if st.session_state.ocr_results:
    with st.expander("抽出結果（最初の数件）", expanded=False):
        # 中身を目視確認しやすいよう snippet を表示
        preview = []
        for r in st.session_state.ocr_results[:3]:
            preview.append({
                "Filename": r.get("Filename"),
                "Snippet": (_pick_text(r) or "")[:120]
            })
        st.json(preview)

# Step 4: 文字数で分割（10,000字ごと）
st.subheader("Step 4. 文字数で分割（10,000字ごと）")

# Step 3 の結果から full_text が空なら復元（キー差異を吸収）
if st.session_state.ocr_results and not (st.session_state.full_text or "").strip():
    reconstructed = "\n\n".join([_pick_text(r) for r in st.session_state.ocr_results if _pick_text(r)]).strip()
    if reconstructed:
        # ★ 復元時にもクリーニング適用
        reconstructed = clean_presummary_text(reconstructed)
        st.session_state.full_text = reconstructed
        if not st.session_state.chapters:
            st.session_state.needs_chapter_split = True  # 変数名は互換利用

full_text_value = (st.session_state.get("full_text") or "").strip()
# 分割は本文があるときのみ有効（空分割を防止）
can_split = bool(full_text_value)

# st.caption(f"full_text length = {len(full_text_value)}")  # ←デバッグ用に必要であればコメント解除

col_step4_run, col_step4_clear = st.columns([2, 1])
with col_step4_run:
    run_split_clicked = st.button(
        "10,000字ごとに分割を実行",
        use_container_width=True,
        disabled=not can_split,
        key="run_char_split",
    )
with col_step4_clear:
    clear_split_clicked = st.button(
        "分割結果をクリア",
        use_container_width=True,
        disabled=not bool(st.session_state.chapters),
        key="clear_char_split",
    )

if run_split_clicked:
    try:
        # 章トークン修復は不要だが、OCRのノイズ整形として残しても害はない
        fixed = fix_broken_chapter_tokens(st.session_state.full_text)
        parts = split_by_fixed_chars(fixed, size=10000)
        st.session_state.chapters = parts            # 下流互換のため同じキーに入れる
        st.session_state.needs_chapter_split = False
        st.success(f"生成チャンク数: {len(parts)}")
    except Exception as e:
        st.error(f"文字数分割に失敗しました: {e}")

if clear_split_clicked:
    st.session_state.chapters = []
    st.session_state.needs_chapter_split = False
    st.info("分割結果をクリアしました。")

if st.session_state.get("needs_chapter_split") and can_split and not st.session_state.get("chapters"):
    try:
        fixed = fix_broken_chapter_tokens(st.session_state.full_text)
        parts = split_by_fixed_chars(fixed, size=10000)
        st.session_state.chapters = parts
        st.session_state.needs_chapter_split = False
        if parts:
            st.success(f"生成チャンク数: {len(parts)}")
        else:
            st.warning("本文が空の可能性があります。Step 3 の OCR を確認してください。")
    except Exception as e:
        st.error(f"文字数分割に失敗しました: {e}")
        st.session_state.needs_chapter_split = False

if st.session_state.chapters:
    st.success(f"生成チャンク数: {len(st.session_state.chapters)}")
    with st.expander("チャンクプレビュー（先頭3つ）", expanded=False):
        for title, body in st.session_state.chapters[:3]:
            st.markdown(f"### {title}")
            st.text(body[:800] + ("\n…(以下略)" if len(body) > 800 else ""))
elif not can_split:
    st.info("Step 3 で OCR を実行し本文を取得してください。")

# Step 5: 要約
st.subheader("Step 5. 要約生成")
if st.session_state.chapters:
    # まとめて要約
    if st.button("全章を要約する", type="primary", use_container_width=True):
        st.session_state.summaries = {}
        for idx, (title, body) in enumerate(st.session_state.chapters, start=1):
            with st.spinner(f"{idx}/{len(st.session_state.chapters)} 要約中: {title}"):
                summary = best_effort_summarize(title, body)
                # フォールバック長制御
                if not _is_llm_available():
                    summary = summary[:st.session_state.get("max_chars", 1200)]
                st.session_state.summaries[title] = summary
        st.success("全章の要約が完了しました。")

    if st.session_state.summaries:
        with st.expander("要約結果（上位3章）", expanded=True):
            for i, (title, summ) in enumerate(list(st.session_state.summaries.items())[:3], start=1):
                #st.markdown(f"## {title}")
                st.write(summ)

# Step 6: PDFエクスポート
st.subheader("Step 6. PDFエクスポート")

has_chapters = bool(st.session_state.chapters)
has_summaries = bool(st.session_state.summaries)

if not (has_chapters or has_summaries):
    st.info("Step 4 で分割した本文または Step 5 の要約ができあがると PDF を生成できます。")
else:
    st.caption("Google Docs を使わずに PDF を生成します。必要に応じて Google Drive へもアップロードできます。")
    default_book_title = st.session_state.get("book_title_input", "Kindle書籍")
    book_title_input = st.text_input(
        "書籍タイトル（PDF内タイトル／Driveフォルダ名）",
        value=default_book_title,
        key="book_title_input",
    )
    st.session_state.book_title_input = book_title_input
    safe_book_title = book_title_input.strip() or "Kindle書籍"

    drive_upload_enabled = st.checkbox(
        "生成したPDFをGoogle Driveにもアップロードする",
        value=st.session_state.get("drive_upload_enabled", False),
        key="drive_upload_enabled",
    )

    drive_root_value = st.session_state.get("drive_root_input", "OCR結果")
    if drive_upload_enabled:
        drive_root_value = st.text_input(
            "Driveの保存先ルートフォルダ",
            value=drive_root_value,
            key="drive_root_input",
        )
    st.session_state.drive_root_input = drive_root_value

    drive_available = drive_upload_enabled and build is not None and MediaIoBaseUpload is not None
    if drive_upload_enabled and not drive_available:
        st.warning("Google Drive へのアップロードには google-api-python-client が必要です。requirements.txt を確認してください。")

    font_ok, font_err = ensure_pdf_font_registered(FONT_NAME, FONT_PATH)
    if not font_ok:
        st.error(font_err or "PDFフォントの初期化に失敗しました。assets ディレクトリに日本語フォント（.otf/.ttf）を配置してください。")

    tab_labels: List[str] = []
    tab_kinds: List[str] = []
    if has_chapters:
        tab_labels.append("本文PDF")
        tab_kinds.append("body")
    if has_summaries:
        tab_labels.append("要約PDF")
        tab_kinds.append("summary")

    tabs = st.tabs(tab_labels) if tab_labels else []

    for tab, kind in zip(tabs, tab_kinds):
        with tab:
            if not font_ok:
                st.info("フォントが登録できないため PDF を生成できません。")
                continue

            if kind == "body":
                base_sections = st.session_state.chapters or [("本文", st.session_state.full_text or "")]
                sections = _prepare_pdf_sections(base_sections)
                if not any(body for _, body in sections):
                    st.warning("本文が空です。Step 3〜4 の結果を確認してください。")
                    continue

                pdf_title = st.text_input(
                    "PDFタイトル",
                    value=st.session_state.get("body_pdf_title", f"{safe_book_title}（本文）"),
                    key="body_pdf_title",
                )
                file_name_input = st.text_input(
                    "ファイル名（.pdf）",
                    value=st.session_state.get("body_pdf_filename", f"{safe_book_title}_本文.pdf"),
                    key="body_pdf_filename",
                )
                file_name = file_name_input.strip() or "document.pdf"
                if not file_name.lower().endswith(".pdf"):
                    file_name += ".pdf"

                pdf_bytes = build_pdf_bytes(title=pdf_title, chapters=sections, font_name=FONT_NAME)
                st.download_button(
                    "本文PDFをダウンロード",
                    data=pdf_bytes,
                    file_name=file_name,
                    mime="application/pdf",
                    key="download_body_pdf",
                )

                if drive_upload_enabled:
                    upload_disabled = not drive_available
                    if st.button("Drive にアップロード", key="upload_body_pdf", disabled=upload_disabled):
                        try:
                            creds = _get_cached_google_credentials()
                            drive_service = _get_drive_service(creds)
                            folder_parts = [drive_root_value or "OCR結果"]
                            folder_parts.append(safe_book_title)
                            folder_id = _ensure_folder_path(drive_service, folder_parts)
                            with st.spinner("Drive にアップロード中…"):
                                meta = _upload_pdf_to_drive(
                                    creds,
                                    file_name=file_name,
                                    file_bytes=pdf_bytes,
                                    parent_folder_id=folder_id,
                                )
                            link = meta.get("webViewLink")
                            if link:
                                st.success(f"アップロード完了: {link}")
                            else:
                                st.success("アップロード完了。Google Drive を確認してください。")
                        except Exception as e:
                            st.error(f"Drive へのアップロードに失敗しました: {e}")
            else:
                summary_sections = _summaries_as_sections(st.session_state.chapters or [], st.session_state.summaries)
                summary_sections = _prepare_pdf_sections(summary_sections)
                if not summary_sections:
                    st.info("要約結果がありません。")
                    continue

                pdf_title = st.text_input(
                    "PDFタイトル",
                    value=st.session_state.get("summary_pdf_title", f"{safe_book_title}（要約）"),
                    key="summary_pdf_title",
                )
                file_name_input = st.text_input(
                    "ファイル名（.pdf）",
                    value=st.session_state.get("summary_pdf_filename", f"{safe_book_title}_要約.pdf"),
                    key="summary_pdf_filename",
                )
                file_name = file_name_input.strip() or "summary.pdf"
                if not file_name.lower().endswith(".pdf"):
                    file_name += ".pdf"

                pdf_bytes = build_pdf_bytes(title=pdf_title, chapters=summary_sections, font_name=FONT_NAME)
                st.download_button(
                    "要約PDFをダウンロード",
                    data=pdf_bytes,
                    file_name=file_name,
                    mime="application/pdf",
                    key="download_summary_pdf",
                )

                if drive_upload_enabled:
                    upload_disabled = not drive_available
                    if st.button("Drive にアップロード", key="upload_summary_pdf", disabled=upload_disabled):
                        try:
                            creds = _get_cached_google_credentials()
                            drive_service = _get_drive_service(creds)
                            folder_parts = [drive_root_value or "OCR結果", safe_book_title, "要約"]
                            folder_id = _ensure_folder_path(drive_service, folder_parts)
                            with st.spinner("Drive にアップロード中…"):
                                meta = _upload_pdf_to_drive(
                                    creds,
                                    file_name=file_name,
                                    file_bytes=pdf_bytes,
                                    parent_folder_id=folder_id,
                                )
                            link = meta.get("webViewLink")
                            if link:
                                st.success(f"アップロード完了: {link}")
                            else:
                                st.success("アップロード完了。Google Drive を確認してください。")
                        except Exception as e:
                            st.error(f"Drive へのアップロードに失敗しました: {e}")
