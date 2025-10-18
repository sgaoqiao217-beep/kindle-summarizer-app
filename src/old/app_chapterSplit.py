import io
import os
import re
import tempfile
import unicodedata
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

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
    from googleapiclient.http import MediaIoBaseDownload
except Exception:
    build = None
    MediaIoBaseDownload = None

try:
    from google_api_utils import (
        get_credentials as get_google_credentials,
        create_google_doc as create_google_doc_external,
    )
except Exception:
    get_google_credentials = None
    create_google_doc_external = None

# Optional: 既存モジュールがあれば使う（無ければImportErrorを握りつぶしてフォールバック）
try:
    from sort_image_file import normalize_filenames as normalize_filenames_external
except Exception:
    normalize_filenames_external = None

try:
    from ocr_utils import extract_info_type1 as extract_info_external
    # ↑ ユーザ環境の関数名に合わせてお好みで
except Exception:
    extract_info_external = None

load_dotenv()

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


def _build_google_doc_content(sections: List[Tuple[str, str]]) -> str:
    """
    Googleドキュメント用に章ごとの本文を連結。
    """
    blocks: List[str] = []
    for raw_title, raw_body in sections:
        heading = raw_title.strip() or "無題"
        body = (raw_body or "").strip()
        if body.startswith("##"):
            body_lines = body.splitlines()
            first_line = body_lines[0].strip()
            normalized_heading = f"## {heading}"
            if first_line == normalized_heading:
                body_lines = body_lines[1:]
                while body_lines and not body_lines[0].strip():
                    body_lines.pop(0)
                body = "\n".join(body_lines).strip()
        block_parts = [heading]
        if body:
            block_parts.append(body)
        blocks.append("\n".join(block_parts))
    return "\n\n\n".join(blocks).strip() or "本文がありません。"


def _get_cached_google_credentials():
    if get_google_credentials is None:
        raise RuntimeError("google_api_utils が見つからないためGoogle認証を利用できません。")
    creds = st.session_state.get("google_creds")
    if creds is None:
        creds = get_google_credentials()
        st.session_state.google_creds = creds
    return creds


def _extract_drive_folder_id(raw_value: str) -> Optional[str]:
    if not raw_value:
        return None
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

# =========================
# OCR
# =========================
@st.cache_resource(show_spinner=False)
def get_vision_client(json_key_path: Optional[str] = None):
    # json_key_pathが指定されていれば一時的に環境変数を差し替え（セッション存続中のみ）
    if json_key_path:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = json_key_path
    from google.cloud import vision
    return vision.ImageAnnotatorClient()

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

# 章タイトル（行頭限定）検出：過去に共有してくれたパターンをベースに改良
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
            # 直前を確定
            if current_title and buff:
                parts.append((current_title, "\n".join(buff).strip()))
            current_title = m.group("heading").strip()
            buff = []
        else:
            buff.append(line)
    if current_title and buff:
        parts.append((current_title, "\n".join(buff).strip()))
    return parts if parts else [("本文", text.strip())]

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
        # あなたの高精度関数がある場合はこちらを優先
        return extract_info_external(img_path)

    # 無い場合は汎用フォールバック
    resp = _extract_with_vision(img_path, client)
    full_text = ""
    if resp.full_text_annotation and resp.full_text_annotation.text:
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
st.caption("画像アップロード → 並べ替え → OCR → 章検出 → 要約 → Googleドキュメント出力 まで")

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
        folder_id = _extract_drive_folder_id(drive_folder_value)
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
    st.caption("※ もしあなたのプロジェクトの命名規則が厳密に決まっている場合は、外部関数の呼び出しを優先してください。")

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
        # 1つの本文に連結（ページ間に改行挿入）
        combined_text = "\n\n".join([r.get("Text", "") for r in results if (r.get("Text") or "").strip()]).strip()
        st.session_state.full_text = combined_text
        st.session_state.needs_chapter_split = bool(st.session_state.full_text or st.session_state.ocr_results)
        st.success(f"OCR完了：{len(results)}ページ")
        # UI を確実に最新状態にする
        st.rerun()

if st.session_state.ocr_results:
    with st.expander("抽出結果（最初の数件）", expanded=False):
        st.json(st.session_state.ocr_results[:3])

# Step 4: 章分割
st.subheader("Step 4. 章分割")
if st.session_state.ocr_results and not (st.session_state.full_text or "").strip():
    reconstructed = "\n\n".join(
        [r.get("Text", "") for r in st.session_state.ocr_results if (r.get("Text") or "").strip()]
    ).strip()
    if reconstructed:
        st.session_state.full_text = reconstructed
        if not st.session_state.chapters:
            st.session_state.needs_chapter_split = True

full_text_value = (st.session_state.get("full_text") or "").strip()
has_full_text = bool(full_text_value.strip())
has_ocr_text = any(bool((r.get("Text") or "").strip()) for r in st.session_state.get("ocr_results", []))
#can_split_chapters = bool(full_text_value or has_ocr_text)
can_split_chapters = bool(full_text_value or st.session_state.get("ocr_results"))
col_step4_run, col_step4_clear = st.columns([2, 1])
with col_step4_run:
    run_split_clicked = st.button(
        "章分割を実行",
        use_container_width=True,
        disabled=not can_split_chapters,
        key="run_chapter_split",
    )
with col_step4_clear:
    clear_split_clicked = st.button(
        "章分割結果をクリア",
        use_container_width=True,
        disabled=not bool(st.session_state.chapters),
        key="clear_chapter_split",
    )

if run_split_clicked:
    try:
        fixed = fix_broken_chapter_tokens(st.session_state.full_text)
        parts = split_by_chapter_linehead(fixed)
        st.session_state.chapters = parts
        st.session_state.needs_chapter_split = False
        st.success(f"検出章数: {len(parts)}")
    except Exception as e:
        st.error(f"章分割に失敗しました: {e}")

if clear_split_clicked:
    st.session_state.chapters = []
    st.session_state.needs_chapter_split = False
    st.info("章分割結果をクリアしました。")

if st.session_state.get("needs_chapter_split") and can_split_chapters and not st.session_state.get("chapters"):
    try:
        fixed = fix_broken_chapter_tokens(st.session_state.full_text)
        parts = split_by_chapter_linehead(fixed)
        st.session_state.chapters = parts
        st.session_state.needs_chapter_split = False
        if parts:
            st.success(f"検出章数: {len(parts)}")
        else:
            st.warning("章見出しが検出できませんでした。必要に応じて本文を確認してください。")
    except Exception as e:
        st.error(f"章分割に失敗しました: {e}")
        st.session_state.needs_chapter_split = False

if st.session_state.chapters:
    st.success(f"検出章数: {len(st.session_state.chapters)}")
    with st.expander("章プレビュー（上位3章）", expanded=False):
        for title, body in st.session_state.chapters[:3]:
            st.markdown(f"### {title}")
            st.text(body[:800] + ("\n…(以下略)" if len(body) > 800 else ""))
elif not can_split_chapters:
    st.info("Step 3 で OCR を実行し本文を取得してください。")
elif not st.session_state.get("needs_chapter_split"):
    st.info("章見出しが見つからない可能性があります。本文を確認するか、カスタムルールで再試行してください。")

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
                st.markdown(f"## {title}")
                st.write(summ)

# Step 6: エクスポート
st.subheader("Step 6. エクスポート")
if st.session_state.summaries:
    st.write("Googleドキュメントに出力します。初回はブラウザでGoogle認証が求められます。")
    default_book_title = st.session_state.get("book_title_input", "Kindle書籍")
    default_root = st.session_state.get("drive_root_input", "OCR結果")
    book_title_input = st.text_input("書籍タイトル（Googleドキュメント名に使用）", value=default_book_title)
    drive_root_input = st.text_input("Google Driveの保存先ルートフォルダ", value=default_root)
    st.session_state.book_title_input = book_title_input
    st.session_state.drive_root_input = drive_root_input

    if st.button("Googleドキュメントを作成", type="primary", use_container_width=True):
        if create_google_doc_external is None or get_google_credentials is None:
            st.error("google_api_utils.py が利用できないため、Googleドキュメント出力に対応していません。")
        else:
            try:
                creds = st.session_state.get("google_creds")
                if creds is None:
                    with st.spinner("Googleアカウント認証中…"):
                        creds = get_google_credentials()
                    st.session_state.google_creds = creds

                chapters_for_doc = (
                    st.session_state.chapters
                    if st.session_state.chapters
                    else [("本文", st.session_state.full_text or "")]
                )
                full_content = _build_google_doc_content(chapters_for_doc)
                summary_sections = list(st.session_state.summaries.items())
                summary_content = _build_google_doc_content(summary_sections)

                with st.spinner("文章全体のドキュメントを作成中…"):
                    create_google_doc_external(
                        book_title_input,
                        "文章全体",
                        full_content,
                        creds,
                        root_name=drive_root_input or "OCR結果",
                    )
                with st.spinner("要約ドキュメントを作成中…"):
                    create_google_doc_external(
                        book_title_input,
                        "要約",
                        summary_content,
                        creds,
                        root_name=drive_root_input or "OCR結果",
                    )
                st.success("Googleドキュメントの作成が完了しました。Google Drive をご確認ください。")
            except Exception as e:
                st.error(f"Googleドキュメントの作成に失敗しました: {e}")

st.divider()
st.caption(
    "💡 GEMINI_API_KEY / GEMINI_MODEL が設定されていればLLM要約を自動で使用します。環境が整っていない場合でもフォールバックで一通り動作し、エクスポートでは文章全体と要約のGoogleドキュメントを作成できます。"
)
