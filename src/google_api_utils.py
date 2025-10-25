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
    """
    Google Docs に新規ドキュメントを作成し、章タイトル(=見出し2)と本文を挿入。
    出力先は Google Drive の `root_name/book_title/` 配下に配置する。
    - 見出しの正規表現判定はしない（split段階で章は確定しているため）
    - 挿入は1回の insertText と1回の updateParagraphStyle で最小化
    - Drive 上の完全移動のため removeParents を使用
    """
    # --- サービス生成（大量作成する場合は外で使い回すとさらに高速） ---
    docs_service = build("docs", "v1", credentials=creds)
    drive_service = build("drive", "v3", credentials=creds)

    # --- フォルダ構成の用意 ---
    book_folder_id = ensure_folder_path(drive_service, [root_name, book_title])

    # --- ドキュメント作成 ---
    doc_name = f"{book_title} ({chapter_title})"
    doc = docs_service.documents().create(body={"title": doc_name}).execute()
    doc_id = doc["documentId"]

    # --- ドキュメントを目的フォルダへ“完全に移動” ---
    # 既存の親（通常は root）を取得して removeParents で外し、addParents に目的フォルダを指定
    file_meta = drive_service.files().get(
        fileId=doc_id,
        fields="parents",
        supportsAllDrives=True
    ).execute()
    prev_ids = file_meta.get("parents", [])
    update_kwargs = {
        "fileId": doc_id,
        "addParents": book_folder_id,
        "fields": "id, parents",
        "supportsAllDrives": True,
    }
    if prev_ids:
        update_kwargs["removeParents"] = ",".join(prev_ids)
    drive_service.files().update(**update_kwargs).execute()

    # --- 本文挿入（章タイトル=見出し2 → 本文の順で一括） ---
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
        # 本文先頭（index=1）にまとめて挿入
        {"insertText": {"location": {"index": 1}, "text": content}},
        # 先頭段落（章タイトル）だけ HEADING_2 に変更
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
            line_without_newline = line.rstrip("\n")
            if line_without_newline.strip() in special_heading_titles:
                requests.append(
                    {
                        "updateParagraphStyle": {
                            "range": {
                                "startIndex": cursor,
                                "endIndex": cursor + len(line),
                            },
                            "paragraphStyle": {"namedStyleType": "HEADING_2"},
                            "fields": "namedStyleType",
                        }
                    }
                )
            cursor += len(line)
    docs_service.documents().batchUpdate(documentId=doc_id, body={"requests": requests}).execute()

    print(f"{doc_name}: https://docs.google.com/document/d/{doc_id}/edit")
    return doc_id
