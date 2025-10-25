#### Google Docs/Drive 認証 & 出力
from googleapiclient.discovery import build
import json
import streamlit as st
from google.oauth2 import service_account


SCOPES = [
    "https://www.googleapis.com/auth/documents",
    "https://www.googleapis.com/auth/drive",
]

def get_credentials():
    """サービスアカウントの資格情報（Docs/Driveスコープ）を返す"""
    # 1) サービスアカウント資格情報を Secrets から読み込み
    raw = st.secrets["GOOGLE_CREDENTIALS"]
    info = json.loads(raw) if isinstance(raw, str) else raw  # TOMLは文字列のことが多い
    creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
    # project_id = info.get("project_id")
    # if project_id:
    #     creds = creds.with_quota_project(project_id)

    # flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET, SCOPES)
    # creds = flow.run_local_server(port=0)
    try:
        info_project = info.get("project_id")
        st.write(f"SA email: {info.get('client_email')}")
        st.write(f"SA project_id (from JSON): {info_project}")
    except Exception:
        pass
    return creds


def get_or_create_folder(service, name, parent_id=None):
    """
    Google Drive 上でフォルダを取得、存在しなければ作成
    """
    safe = name.replace("'", r"\'")
    query = f"mimeType='application/vnd.google-apps.folder' and name='{safe}' and trashed=false"
    if parent_id:
        query += f" and '{parent_id}' in parents"

    results = service.files().list(q=query, 
                                   fields="files(id, name)",
                                   supportsAllDrives=True,
                                   includeItemsFromAllDrives=True
                                   ).execute()
    items = results.get("files", [])

    if items:
        return items[0]["id"]  # 既存フォルダのIDを返す

    # フォルダ作成
    metadata = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
    }
    if parent_id:
        metadata["parents"] = [parent_id]

    folder = service.files().create(body=metadata, fields="id",
                                    supportsAllDrives=True 
                                    ).execute()
    return folder["id"]


def ensure_folder_path(drive_service, path_parts):
    """['OCR結果', 書籍タイトル] のような配列から階層を順に作成/取得して最下層IDを返す"""
    parent_id = None
    for name in path_parts:
        parent_id = get_or_create_folder(drive_service, name, parent_id)
    return parent_id

def create_google_doc(book_title: str, chapter_title: str, text: str, creds, root_name: str = "OCR結果"):
    docs_service = build("docs", "v1", credentials=creds)
    drive_service = build("drive", "v3", credentials=creds)

    # 1) 保存先フォルダを用意
    book_folder_id = ensure_folder_path(drive_service, [root_name, book_title])

    # 2) Google ドキュメントを「Drive API で」作成（親フォルダを最初から指定）
    doc_name = f"{book_title} ({chapter_title})"
    file_meta = {
        "name": doc_name,
        "mimeType": "application/vnd.google-apps.document",
        "parents": [book_folder_id],  # ← これで最初から目的フォルダに作成できる
    }
    created = drive_service.files().create(
        body=file_meta,
        fields="id, parents",
        supportsAllDrives=True
    ).execute()
    doc_id = created["id"]

    # 3) 本文を Docs API で挿入（先頭を見出し2に）
    heading = f"{chapter_title}\n"
    body_text = ""
    if text:
        lines = text.splitlines()
        if lines and lines[0].startswith("## "):
            candidate_heading = lines[0][3:].strip()
            if candidate_heading:
                heading = f"{candidate_heading}\n"
                lines = lines[1:]
        while lines and not lines[0].strip():
            lines.pop(0)
        body_text = "\n".join(lines).strip("\n")
        if body_text:
            body_text += "\n"

    content = heading + body_text if body_text else heading

    requests = [
        {"insertText": {"location": {"index": 1}, "text": content}},
        {
            "updateParagraphStyle": {
                "range": {"startIndex": 1, "endIndex": 1 + len(heading)},
                "paragraphStyle": {"namedStyleType": "HEADING_2"},
                "fields": "namedStyleType",
            }
        },
    ]

    special_heading_titles = {"プロローグ", "結論"}
    if body_text:
        cursor = 1 + len(heading)
        for line in body_text.splitlines(keepends=True):
            if line.rstrip("\n").strip() in special_heading_titles:
                requests.append({
                    "updateParagraphStyle": {
                        "range": {"startIndex": cursor, "endIndex": cursor + len(line)},
                        "paragraphStyle": {"namedStyleType": "HEADING_2"},
                        "fields": "namedStyleType",
                    }
                })
            cursor += len(line)

    docs_service.documents().batchUpdate(documentId=doc_id, body={"requests": requests}).execute()

    print(f"{doc_name}: https://docs.google.com/document/d/{doc_id}/edit")
    return doc_id
