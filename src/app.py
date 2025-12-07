import io
import os
import re
import json
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
    from google_auth_oauthlib.flow import InstalledAppFlow
except Exception:
    InstalledAppFlow = None

try:
    from google_api_utils import (
        get_credentials as get_google_credentials,
        create_google_doc as create_google_doc_external,
    )
except Exception:
    get_google_credentials = None
    create_google_doc_external = None
# --- Fallback for google_api_utils が無い環境 ---
if get_google_credentials is None or create_google_doc_external is None:
    from google.oauth2 import service_account
    # googleapiclient.build は app.py 先頭で try-import 済み（build が None の可能性あり）

    _SERVICE_ACCOUNT_SCOPES = [
        "https://www.googleapis.com/auth/documents",
        "https://www.googleapis.com/auth/drive",
    ]

    _USER_OAUTH_SCOPES = [
        "https://www.googleapis.com/auth/drive.file",
        "https://www.googleapis.com/auth/documents",
    ]

    def _load_json_secret(raw):
        return json.loads(raw) if isinstance(raw, str) else raw

    def _get_service_account_credentials():
        raw = st.secrets["GOOGLE_CREDENTIALS"]  # secrets.toml に JSON 全文を入れる運用
        info = _load_json_secret(raw)
        return service_account.Credentials.from_service_account_info(info, scopes=_SERVICE_ACCOUNT_SCOPES)

    def _get_user_oauth_creds():
        if InstalledAppFlow is None:
            raise ImportError("google-auth-oauthlib が必要です。`pip install google-auth-oauthlib` を実行してください。")
        if "google_oauth" not in st.secrets:
            raise KeyError("st.secrets['google_oauth'] が設定されていません")
        section = st.secrets["google_oauth"]
        try:
            client_json = section["client_json"]
        except Exception as exc:
            raise KeyError("st.secrets['google_oauth']['client_json'] が設定されていません") from exc
        config = _load_json_secret(client_json)
        flow = InstalledAppFlow.from_client_config(config, _USER_OAUTH_SCOPES)
        creds = flow.run_local_server(port=0)
        st.write("Using user OAuth credentials (files count toward your Drive quota).")
        return creds

    def get_google_credentials(use_user_oauth: bool | None = None):
        """Secretsに応じてサービスアカウント or OAuth を選択"""
        if use_user_oauth is None:
            use_user_oauth = "google_oauth" in st.secrets
        if use_user_oauth:
            return _get_user_oauth_creds()
        return _get_service_account_credentials()

    def _ensure_folder_path(drive_service, parts):
        """['OCR結果','書籍タイトル'] のようなパスをDrive上に作成して親IDを返す"""
        parent_id = None
        for name in parts:
            q = "mimeType='application/vnd.google-apps.folder' and name='%s'" % name.replace("'", r"\'")
            if parent_id:
                q += f" and '{parent_id}' in parents"
            res = drive_service.files().list(q=q, fields="files(id,name)", pageSize=1).execute()
            items = res.get("files", [])
            if items:
                parent_id = items[0]["id"]
            else:
                meta = {"name": name, "mimeType": "application/vnd.google-apps.folder"}
                if parent_id:
                    meta["parents"] = [parent_id]
                created = drive_service.files().create(body=meta, fields="id").execute()
                parent_id = created["id"]
        return parent_id

    def create_google_doc_external(book_title: str, chapter_title: str, text: str, creds, root_name: str = "OCR結果"):
        if build is None:
            raise ImportError("googleapiclient が必要です。`pip install google-api-python-client` を実行してください。")
        docs = build("docs", "v1", credentials=creds)
        drive = build("drive", "v3", credentials=creds)

        # Driveの保存先を用意: OCR結果/書籍タイトル
        folder_id = _ensure_folder_path(drive, [root_name, book_title])

        # ドキュメント作成
        doc_title = f"{book_title}（{chapter_title}）"
        doc = docs.documents().create(body={"title": doc_title}).execute()
        doc_id = doc["documentId"]

        # 1行目を見出し2にして本文を挿入（重複見出しを避ける）
        heading = (chapter_title or "無題").strip()
        body_text = (text or "").lstrip()
        first = (body_text.splitlines() or [""])[0].strip()
        if first == f"## {heading}":
            body_text = "\n".join(body_text.splitlines()[1:]).lstrip()

        content = f"{heading}\n\n{body_text}".rstrip() + "\n"
        requests = [
            {"insertText": {"location": {"index": 1}, "text": content}},
            {
                "updateParagraphStyle": {
                    "range": {"startIndex": 1, "endIndex": 1 + len(heading) + 1},
                    "paragraphStyle": {"namedStyleType": "HEADING_2"},
                    "fields": "namedStyleType",
                }
            },
        ]
        docs.documents().batchUpdate(documentId=doc_id, body={"requests": requests}).execute()

        # 作成直後はマイドライブ直下なので、保存先フォルダへ移動
        meta = drive.files().get(fileId=doc_id, fields="parents").execute()
        prev = ",".join(meta.get("parents", []))
        drive.files().update(
            fileId=doc_id, addParents=folder_id, removeParents=prev, fields="id, parents"
        ).execute()

        return doc_id
# --- Fallback ここまで ---

# Optional: 既存モジュールがあれば使う（無ければImportErrorを握りつぶしてフォールバック）
try:
    from sort_image_file import normalize_filenames as normalize_filenames_external
except Exception:
    normalize_filenames_external = None

try:
    from ocr_utils import extract_info_type1 as extract_info_external
    from ocr_utils import _vision_client_from_secrets  # noqa: F401
    # ↑ ユーザ環境の関数名に合わせてお好みで
except Exception:
    extract_info_external = None


from google.oauth2 import service_account

load_dotenv()

# 句点や括弧で段落/文末を判断するための安全な終端記号セット
_JP_SENT_END = "。．！？!?" + "」』）】］》〉"
# 文頭に来やすい始まり記号（次行がこれで始まるなら改行を残す）
_JP_SENT_BEGIN = "「『（【［《〈"

# =========================================================
# ★ 要約前クリーニング（見出し修正・レンジ行削除・文中改行の解消）
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

def clean_presummary_text_by_direction(text: str, writing_direction: str) -> str:
    """
    縦書き/横書きの違いによって前処理の方針を切り替えるフック。
    現時点では縦書き・横書きともに共通のクリーニングを行うが、
    将来的にロジックを分けたい場合はここを編集する。
    """
    # writing_direction: "vertical" / "horizontal" を想定
    direction = (writing_direction or "vertical").lower()
    # 将来的に if direction.startswith("h"): で横書き専用ロジックを分けてもよい
    return clean_presummary_text(text)

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

def _get_drive_service(creds):
    if build is None:
        raise ImportError("googleapiclient がインストールされていません。`pip install google-api-python-client` を実行してください。")
    return build("drive", "v3", credentials=creds)

def _find_folder_id(service, name: str, parent_id: Optional[str]) -> Optional[str]:
    """親ID直下にある name のフォルダIDをひとつ返す（最初の一致）。親なしの場合はマイドライブ直下を検索。"""
    # Drive の query は単引用符で囲むので、単引用符はバックスラッシュでエスケープ
    safe_name = (name or "").replace("'", "\\'")
    name_q = "name = '{}'".format(safe_name)
    mime_q = "mimeType = 'application/vnd.google-apps.folder'"
    trashed_q = "trashed = false"

    if parent_id:
        parent_q = f"'{parent_id}' in parents"
        q = f"{name_q} and {mime_q} and {trashed_q} and {parent_q}"
    else:
        q = f"{name_q} and {mime_q} and {trashed_q}"

    res = service.files().list(
        q=q,
        fields="files(id, name, parents)",
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()
    files = res.get("files", [])
    if not files:
        return None
    if parent_id:
        return files[0]["id"]
    return files[0]["id"]

def _resolve_book_folder_id(creds, root_name: str, book_title: str) -> Optional[str]:
    """root_name/book_title のフォルダIDを推定して返す（存在しなければ None）。"""
    try:
        service = _get_drive_service(creds)
        root_id = _find_folder_id(service, root_name, parent_id=None)
        if not root_id:
            return None
        book_id = _find_folder_id(service, book_title, parent_id=root_id)
        return book_id
    except Exception:
        return None

def _folder_url(folder_id: str) -> str:
    return f"https://drive.google.com/drive/folders/{folder_id}"

def _search_url(book_title: str) -> str:
    # 見つからなかったときの保険として検索URLを提示
    from urllib.parse import quote
    return f"https://drive.google.com/drive/search?q={quote(book_title)}"


def _build_single_doc_content(heading: str, body: str) -> str:
    """
    1つのパートを1つのGoogle Docに書き出すための本文を生成。
    body 先頭に '## 見出し' が付いている場合は重複しないよう除去。
    """
    h = (heading or "無題").strip()
    b = (body or "").strip()

    if b.startswith("##"):
        lines = b.splitlines()
        if lines and lines[0].strip() == f"## {h}":
            lines = lines[1:]
            while lines and not lines[0].strip():
                lines.pop(0)
            b = "\n".join(lines).strip()

    return f"{h}\n\n{b}" if b else h


def _make_part_doc_name(idx: int) -> str:
    """
    ドキュメント名（Part 01.doc など）
    Googleドキュメント自体はネイティブ形式ですが、名前に .doc を含めても問題ありません。
    """
    return f"Part {idx:02d}.doc"

def _make_part_summary_doc_name(idx: int) -> str:
    """
    要約用のドキュメント名（Part1 _要約.doc など）
    ※ 数字はゼロ埋めしない／Part と数字の間は詰める／数字の後に半角スペース＋"_要約.doc"
    """
    return f"Part{idx} _要約.doc"

_DELEGATED_SA_SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/documents",
]


def _whoami_email(creds) -> str:
    """現在の認証ユーザーのメールアドレスを取得"""
    if build is None:
        raise ImportError("googleapiclient が必要です。`pip install google-api-python-client` を実行してください。")
    return (
        build("drive", "v3", credentials=creds)
        .about()
        .get(fields="user(emailAddress)")
        .execute()["user"]["emailAddress"]
    )


def _load_secret_dict(raw):
    return json.loads(raw) if isinstance(raw, str) else raw


def _get_delegated_service_account_credentials():
    """Workspace サービスアカウント＋ドメイン全体委任の資格情報を取得"""
    if "GOOGLE_CREDENTIALS" not in st.secrets:
        raise KeyError("st.secrets['GOOGLE_CREDENTIALS'] が設定されていません")
    if "DELEGATED_USER" not in st.secrets:
        raise KeyError("st.secrets['DELEGATED_USER'] が設定されていません")
    info = _load_secret_dict(st.secrets["GOOGLE_CREDENTIALS"])
    subject = st.secrets["DELEGATED_USER"]
    base = service_account.Credentials.from_service_account_info(info, scopes=_DELEGATED_SA_SCOPES)
    return base.with_subject(subject)

def _get_shared_drive_member_sa_credentials():
    """共有ドライブのメンバーに追加済みのサービスアカウントで実行する（DWDなし）"""
    if "GOOGLE_CREDENTIALS" not in st.secrets:
        raise KeyError("st.secrets['GOOGLE_CREDENTIALS'] が設定されていません")
    info = _load_secret_dict(st.secrets["GOOGLE_CREDENTIALS"])
    return service_account.Credentials.from_service_account_info(info, scopes=_DELEGATED_SA_SCOPES)


def _ensure_drive_docs_creds():
    """Drive/Docs 用資格情報（優先: SA+DWD, フォールバック: ユーザーOAuth）"""
    creds = st.session_state.get("google_creds")
    if creds is not None:
        return creds

    try:
        creds = _get_delegated_service_account_credentials()
        st.session_state.google_creds = creds
        st.session_state.google_creds_source = "delegated_sa"
        return creds
    except KeyError as missing_conf:
        st.warning(f"SA+DWD 用の設定が不足しています: {missing_conf}\nユーザーOAuthで再取得します。")
    except Exception as delegated_error:
        raise RuntimeError(f"SA+DWD 認証に失敗しました: {delegated_error}") from delegated_error

    with st.spinner("Googleアカウント認証中…"):
        creds = get_google_credentials(use_user_oauth=True)
    st.session_state.google_creds = creds
    st.session_state.google_creds_source = "user_oauth"
    return creds


def _get_cached_google_credentials():
    # ここで get_google_credentials が None の可能性はない（上のフォールバックで定義済み）
    return _ensure_drive_docs_creds()


def _escape_drive_query_value(value: str) -> str:
    return (value or "").replace("'", "\\'")


def _ensure_child_folder(drive_service, parent_id: str, folder_name: str) -> str:
    """
    親フォルダ直下に folder_name のフォルダを保証してIDを返す。
    共有ドライブでも動作するよう supportsAllDrives を有効にする。
    """
    normalized = (folder_name or "").strip() or "Untitled"
    safe_name = _escape_drive_query_value(normalized)
    query = (
        f"name = '{safe_name}' and "
        "mimeType = 'application/vnd.google-apps.folder' and "
        "trashed = false and "
        f"'{parent_id}' in parents"
    )
    res = drive_service.files().list(
        q=query,
        fields="files(id, name)",
        pageSize=1,
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()
    items = res.get("files", [])
    if items:
        return items[0]["id"]

    metadata = {
        "name": normalized,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_id],
    }
    created = drive_service.files().create(
        body=metadata,
        fields="id",
        supportsAllDrives=True,
    ).execute()
    return created["id"]


def _ensure_export_folder_tree(drive_service, parent_folder_id: str, book_title: str) -> Dict[str, str]:
    """
    共有ドライブ/parent_folder_id 配下に
    book_title/{book_title (文章全体), book_title (要約)}
    を作成して各IDを返す。
    """
    effective_title = (book_title or "").strip() or "Kindle書籍"
    book_folder_id = _ensure_child_folder(drive_service, parent_folder_id, effective_title)
    full_folder_id = _ensure_child_folder(drive_service, book_folder_id, f"{effective_title} (文章全体)")
    summary_folder_id = _ensure_child_folder(drive_service, book_folder_id, f"{effective_title} (要約)")
    return {
        "book_folder_id": book_folder_id,
        "full_folder_id": full_folder_id,
        "summary_folder_id": summary_folder_id,
    }


def _create_doc_in_shared_drive(
    parent_folder_id: str,
    doc_title: str,
    content: str,
    creds,
    drive_service=None,
    docs_service=None,
):
    if build is None:
        raise ImportError("googleapiclient が必要です。")

    if not parent_folder_id:
        raise ValueError("parent_folder_id is required")

    # サービス生成（外から渡されたものがあれば再利用）
    drive = drive_service or build("drive", "v3", credentials=creds)
    docs  = docs_service  or build("docs",  "v1", credentials=creds)

    # ★ ここで“親が共有ドライブ配下か”を確認（driveId が付く）
    parent_meta = drive.files().get(
        fileId=parent_folder_id,
        fields="id,name,driveId",
        supportsAllDrives=True,
    ).execute()

    st.info(f"出力先確認: name={parent_meta.get('name')} driveId={parent_meta.get('driveId')}")

    if not parent_meta.get("driveId"):
        raise RuntimeError(
            "指定の親IDは共有ドライブ配下ではありません。共有ドライブ内のフォルダIDを指定してください。"
        )

    # 共有ドライブ配下が確認できたので作成
    file_meta = {
        "name": doc_title,
        "mimeType": "application/vnd.google-apps.document",
        "parents": [parent_folder_id],
    }
    created = drive.files().create(
        body=file_meta,
        fields="id,parents",
        supportsAllDrives=True,
    ).execute()
    doc_id = created["id"]

    payload = (content or "").rstrip() + "\n"
    requests = [{"insertText": {"location": {"index": 1}, "text": payload}}]

    # 1行目を見出し2に
    first_line = payload.splitlines()[0] if payload.strip() else ""
    if first_line:
        heading_end = len(first_line) + 1
        requests.append({
            "updateParagraphStyle": {
                "range": {"startIndex": 1, "endIndex": heading_end},
                "paragraphStyle": {"namedStyleType": "HEADING_2"},
                "fields": "namedStyleType",
            }
        })

    docs.documents().batchUpdate(documentId=doc_id, body={"requests": requests}).execute()
    return doc_id


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
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
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

def _log_drive_identity_once(creds, force: bool = False):
    """Drive APIで現在の認証ユーザーを表示（force=Trueで毎回表示）"""
    if st.session_state.get("whoami_logged") and not force:
        return
    try:
        email = _whoami_email(creds)
        st.info(f"Google Drive として実行中: **{email}**")
        st.session_state.whoami_logged = True
    except Exception as e:
        st.warning(f"認証ユーザー表示に失敗しました: {e}")
        st.session_state.whoami_logged = True  # 失敗してもループ防止


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
def get_vision_client(json_key_path: Optional[str] = None):
    # 余計なプロジェクト環境変数が効いていると衝突するので一応クリア
    for k in ("GOOGLE_CLOUD_QUOTA_PROJECT", "GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT"):
        os.environ.pop(k, None)

    from google.cloud import vision
    from google.oauth2 import service_account as _sa
    import json as _json

    scopes = ["https://www.googleapis.com/auth/cloud-platform"]

    if json_key_path:
        # 「JSONをアップロード」選択時はこちらを使用（ファイルパスを直接読む）
        creds = _sa.Credentials.from_service_account_file(json_key_path, scopes=scopes)
        return vision.ImageAnnotatorClient(credentials=creds)

    # ここから secrets.toml 経由（JSON本文をそのまま入れる）
    raw = st.secrets.get("GOOGLE_CREDENTIALS")
    if not raw:
        raise KeyError("st.secrets['GOOGLE_CREDENTIALS'] が未設定です。 .streamlit/secrets.toml にサービスアカウントJSON全文を入れてください。")

    info = _json.loads(raw) if isinstance(raw, str) else raw
    creds = _sa.Credentials.from_service_account_info(info, scopes=scopes)
    return vision.ImageAnnotatorClient(credentials=creds)

def _extract_with_vision(img_path: str, client, writing_direction: str = "vertical"):
    """
    Google Cloud Vision で OCR を実行する。
    縦書き/横書きどちらも document_text_detection を使い、
    language_hints だけ切り替える。
    """
    from google.cloud import vision

    with open(img_path, "rb") as f:
        content = f.read()
    image = vision.Image(content=content)

    direction = (writing_direction or "vertical").lower()

    if direction.startswith("h"):
        # 横書き: 日本語＋英語想定
        context = vision.ImageContext(language_hints=["ja", "en"])
    else:
        # 縦書き: 日本語メイン
        context = vision.ImageContext(language_hints=["ja"])

    # ★ どちらも document_text_detection を使う
    resp = client.document_text_detection(
        image=image,
        image_context=context,
    )

    if resp.error.message:
        raise RuntimeError(resp.error.message)
    return resp

def _assemble_text_from_annotation(
    resp,
    writing_direction: str = "vertical",
    img_path: Optional[str] = None,
) -> str:
    """
    Vision の full_text_annotation からテキストを取り出す。
    横書きは full_text_annotation.text をそのまま信頼し、
    うまく取れなかった場合だけ text_annotations[0].description にフォールバックする。
    """

    fta = getattr(resp, "full_text_annotation", None)
    if not fta:
        # full_text_annotation 自体が無い場合は昔ながらの description にフォールバック
        if getattr(resp, "text_annotations", None):
            return resp.text_annotations[0].description or ""
        return ""

    direction = (writing_direction or "vertical").lower()

    # ★ 修正：横書きは Vision が組んだテキストをそのまま使う
    if direction.startswith("h"):
        if getattr(fta, "text", None):
            return fta.text
        if getattr(resp, "text_annotations", None):
            return resp.text_annotations[0].description or ""
        return ""

    # 縦書きも基本はそのまま
    if getattr(fta, "text", None):
        return fta.text
    if getattr(resp, "text_annotations", None):
        return resp.text_annotations[0].description or ""
    return ""

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

def ocr_one_image(img_path: str, client, writing_direction: str = "vertical") -> Dict[str, Any]:
    """
    1画像のOCR→info抽出→本文返却
    writing_direction: "vertical" / "horizontal"
    """
    if extract_info_external:
        # 外部実装の戻りが Body/Text など様々でも Text を必ず埋める
        try:
            # writing_direction を受け取る実装があれば渡す
            res = extract_info_external(img_path, writing_direction=writing_direction)
        except TypeError:
            # 旧実装（引数が1つのみ）の場合はこちら
            res = extract_info_external(img_path)
        res = dict(res) if isinstance(res, dict) else {}
        text = _pick_text(res)
        res.setdefault("Filename", os.path.basename(img_path))
        res["Text"] = text
        return res

    # 無い場合は汎用フォールバック
    resp = _extract_with_vision(img_path, client, writing_direction=writing_direction)
    full_text = _assemble_text_from_annotation(
        resp,
        writing_direction=writing_direction,
        img_path=img_path,
    )


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
    st.markdown("### 要約するAIモデルを選択")
    model_candidates = ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-3-pro-preview"]
    current_model = st.session_state.get("gemini_model", os.getenv("GEMINI_MODEL"))
    model_index = model_candidates.index(current_model) if current_model in model_candidates else 0
    selected_model = st.selectbox("Gemini モデル", model_candidates, index=model_index)
    st.session_state.gemini_model = selected_model
    os.environ["GEMINI_MODEL"] = selected_model

    st.divider()
    st.markdown("### 認証ツール")
    if st.button("現在のGoogle認証を確認", key="btn_check_auth", use_container_width=True):
        try:
            creds = _get_shared_drive_member_sa_credentials()
            _log_drive_identity_once(creds, force=True)  # 「誰として実行中か」を青い枠で表示
        except Exception as e:
            st.error(f"認証確認に失敗: {e}")


# セッション状態
if "workdir" not in st.session_state:
    st.session_state.workdir = tempfile.mkdtemp(prefix="kindle_ocr_")
if "gemini_model" not in st.session_state:
    st.session_state.gemini_model = os.getenv("GEMINI_MODEL") or "gemini-2.5-pro"
if st.session_state.gemini_model:
    os.environ["GEMINI_MODEL"] = st.session_state.gemini_model
if not os.getenv("GEMINI_API_KEY"):
    # Streamlit の secrets から補完（環境変数優先）
    api_key_from_secrets = st.secrets.get("GEMINI_API_KEY")
    if api_key_from_secrets:
        os.environ["GEMINI_API_KEY"] = api_key_from_secrets
if "uploaded_key" not in st.session_state:
    st.session_state.uploaded_key = None
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
if "max_chars" not in st.session_state:
    st.session_state.max_chars = 1200
if "writing_direction" not in st.session_state:
    # デフォルトは縦書き
    st.session_state.writing_direction = "vertical"

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

# Step 1.5: 書籍レイアウト（縦書き/横書き）の選択
st.subheader("この書籍のレイアウトを選んでください")  # ← 見出しをここで大きく表示

writing_direction = st.radio(
    "",  # ← ラベルは空にすることで二重表示を避ける
    ["縦書き", "横書き"],
    index=0 if st.session_state.get("writing_direction", "vertical") == "vertical" else 1,
    horizontal=True,
)

st.session_state["writing_direction"] = (
    "vertical" if "縦書き" in writing_direction else "horizontal"
)

# Step 2: 並び替え（ファイル名/時刻ベース）
st.subheader("Step 2. ページ順に並び替え")
if st.session_state.images:
    if st.button("並び替えを実行", use_container_width=True):
        if normalize_filenames_external:
            normalize_filenames_external(st.session_state.workdir)
            # ↑ あなたの関数の返り値仕様に合わせて調整が必要な場合あり
            # ここではworkdir内をリネーム→再取得を想定
            st.session_state.images = [os.path.join(st.session_state.workdir, f) for f in os.listdir(st.session_state.workdir)]
        st.session_state.images = normalize_filenames_local(st.session_state.images)
        st.success("並び替え完了")

# Step 3: OCR & info抽出
st.subheader("Step 3. OCR & info抽出")
if st.session_state.images:
    uploaded_key = st.session_state.get("uploaded_key")
    client = get_vision_client(uploaded_key)
    direction = st.session_state.get("writing_direction", "vertical")
    st.caption(f"OCRモード: {'横書き' if direction == 'horizontal' else '縦書き'}")

    if st.button("OCRを実行", type="primary", use_container_width=True):
        results = []
        prog = st.progress(0.0, text="OCR処理中…")
        # direction = st.session_state.get("writing_direction", "vertical")
        for i, p in enumerate(st.session_state.images):
            try:
                info = ocr_one_image(p, client, writing_direction=direction)
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

        # ★ ここでクリーニングを適用（縦書き/横書きに応じた前処理フック）
        cleaned_text = clean_presummary_text_by_direction(combined_text, direction)

        st.session_state.full_text = cleaned_text
        st.session_state.needs_chapter_split = bool(st.session_state.full_text)

        st.success(f"OCR完了：{len(results)}ページ")

if st.session_state.ocr_results:
    with st.expander("抽出結果（最初の数件）", expanded=False):
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
        # ★ 復元時にもクリーニング適用（縦書き/横書きに応じて）
        direction = st.session_state.get("writing_direction", "vertical")
        reconstructed = clean_presummary_text_by_direction(reconstructed, direction)
        st.session_state.full_text = reconstructed
        if not st.session_state.chapters:
            st.session_state.needs_chapter_split = True

full_text_value = (st.session_state.get("full_text") or "").strip()
can_split = bool(full_text_value)

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
        fixed = fix_broken_chapter_tokens(st.session_state.full_text)
        parts = split_by_fixed_chars(fixed, size=10000)
        st.session_state.chapters = parts
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
    if st.button("全章を要約する", type="primary", use_container_width=True):
        st.session_state.summaries = {}
        for idx, (title, body) in enumerate(st.session_state.chapters, start=1):
            with st.spinner(f"{idx}/{len(st.session_state.chapters)} 要約中: {title}"):
                summary = best_effort_summarize(title, body)
                if not _is_llm_available():
                    summary = summary[:st.session_state.get("max_chars", 1200)]
                st.session_state.summaries[title] = summary
        st.success("全章の要約が完了しました。")

    if st.session_state.summaries:
        with st.expander("要約結果（上位3章）", expanded=True):
            for i, (title, summ) in enumerate(list(st.session_state.summaries.items())[:3], start=1):
                st.write(summ)

# Step 6: エクスポート
st.subheader("Step 6. エクスポート")
if st.session_state.summaries:
    st.write("Workspace サービスアカウント（ドメイン全体委任）経由で共有ドライブに直接 Googleドキュメントを作成します。")
    default_book_title = st.session_state.get("book_title_input", "Kindle書籍")
    default_root = st.session_state.get("drive_root_input", "")
    book_title_input = st.text_input("書籍タイトル（Googleドキュメント名に使用）", value=default_book_title)
    drive_root_input = st.text_input(
        "共有ドライブの親フォルダURLまたはID",
        value=default_root,
        placeholder="https://drive.google.com/drive/u/0/folders/1soiMC-Rl1q6RSr8z4DdsDDk07aq6UNeT",
        help="共有ドライブ内の任意フォルダURLを貼り付けるか、フォルダID文字列を入力してください。",
    )
    st.session_state.book_title_input = book_title_input
    st.session_state.drive_root_input = drive_root_input

    st.markdown("**出力方法**")
    export_mode = st.radio(
        "Googleドキュメントの作り方",
        ["1本にまとめる（従来）", "パートごとに分割する（Part 01.doc / Part 02.doc ...）"],
        horizontal=False,
        index=1,
    )
    split_summaries = st.checkbox("要約もパートごとに個別ドキュメントで出力する", value=True)

    if st.button("Googleドキュメントを作成", type="primary", use_container_width=True):
        if build is None or get_google_credentials is None:
            st.error("google_api_utils.py もしくは googleapiclient が利用できないため、Googleドキュメント出力に対応していません。")
        else:
            parent_folder_id = normalize_drive_folder_input(drive_root_input)
            if not parent_folder_id:
                st.error("共有ドライブのフォルダURLまたはIDを入力してください。")
                st.stop()
            try:
                creds = _get_shared_drive_member_sa_credentials()
                _log_drive_identity_once(creds)

                drive_service = build("drive", "v3", credentials=creds)
                docs_service  = build("docs",  "v1", credentials=creds)

                effective_book_title = (book_title_input or "").strip() or "Kindle書籍"
                folder_info = _ensure_export_folder_tree(
                    drive_service,
                    parent_folder_id,
                    effective_book_title,
                )
                full_folder_id = folder_info["full_folder_id"]
                summary_folder_id = folder_info["summary_folder_id"]
                book_folder_id = folder_info["book_folder_id"]

                chapters_for_doc = (
                    st.session_state.chapters
                    if st.session_state.chapters
                    else [("本文", st.session_state.full_text or "")]
                )

                if export_mode.startswith("1本にまとめる"):
                    full_content = _build_google_doc_content(chapters_for_doc)
                    summary_sections = list(st.session_state.summaries.items())
                    summary_content = _build_google_doc_content(summary_sections)

                    with st.spinner("文章全体のドキュメントを作成中…"):
                        _create_doc_in_shared_drive(
                            full_folder_id,
                            f"{effective_book_title}（文章全体）",
                            full_content,
                            creds,
                            drive_service=drive_service,
                            docs_service=docs_service,
                        )
                    with st.spinner("要約ドキュメントを作成中…"):
                        _create_doc_in_shared_drive(
                            summary_folder_id,
                            f"{effective_book_title}（要約）",
                            summary_content,
                            creds,
                            drive_service=drive_service,
                            docs_service=docs_service,
                        )
                    st.success("Googleドキュメントの作成が完了しました。Google Drive をご確認ください。")
                    st.markdown(f"📂 保存先フォルダ: [{book_folder_id}]({_folder_url(book_folder_id)})")

                else:
                    total = len(chapters_for_doc)

                    with st.spinner("パートごとの本文ドキュメントを作成中…"):
                        for idx, (title, body) in enumerate(chapters_for_doc, start=1):
                            doc_name = _make_part_doc_name(idx)
                            content = _build_single_doc_content(title, body)
                            _create_doc_in_shared_drive(
                                full_folder_id,
                                f"{effective_book_title}（{doc_name}）",
                                content,
                                creds,
                                drive_service=drive_service,
                                docs_service=docs_service,
                            )

                    if split_summaries and st.session_state.summaries:
                        with st.spinner("パートごとの要約ドキュメントを作成中…"):
                            for idx, (title, _) in enumerate(chapters_for_doc, start=1):
                                summary_text = st.session_state.summaries.get(title)
                                if not summary_text:
                                    continue
                                content = _build_single_doc_content(title, summary_text)
                                doc_name = _make_part_summary_doc_name(idx)
                                _create_doc_in_shared_drive(
                                    summary_folder_id,
                                    f"{effective_book_title}（{doc_name}）",
                                    content,
                                    creds,
                                    drive_service=drive_service,
                                    docs_service=docs_service,
                                )

                    st.success(f"パート分割の作成が完了しました（{total}件）。Google Drive をご確認ください。")
                    st.markdown(f"📂 保存先フォルダ: [{book_folder_id}]({_folder_url(book_folder_id)})")

            except Exception as e:
                st.error(f"Googleドキュメントの作成に失敗しました: {e}")
