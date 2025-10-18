#### Google Docs/Drive 認証 & 出力
import re
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
import os

from dotenv import load_dotenv

load_dotenv()


SCOPES = [
    "https://www.googleapis.com/auth/documents",
    "https://www.googleapis.com/auth/drive",
]

# google_api_utils.py のあるディレクトリを基準にして1つ上（OCRTool/）へ移動
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIENT_SECRET = os.path.join(
    BASE_DIR,
    os.getenv("CLIENT_KEY")
)

def get_credentials():
    """OAuth 認証を行い、Google API の認証情報を返す"""
    flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET, SCOPES)
    creds = flow.run_local_server(port=0)
    return creds

def get_or_create_folder(service, name, parent_id=None):
    """
    Google Drive 上でフォルダを取得、存在しなければ作成
    """
    query = f"mimeType='application/vnd.google-apps.folder' and name='{name}'"
    if parent_id:
        query += f" and '{parent_id}' in parents"
    results = service.files().list(q=query, fields="files(id, name)").execute()
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

    folder = service.files().create(body=metadata, fields="id").execute()
    return folder["id"]

def create_google_doc(book_title, chapter_title, text, creds):
    """
    Google Docs に新規ドキュメントを作成し、OCR結果を挿入
    - 章タイトルを見出し2に変換
    - 出力先を Google Drive の OCR結果/書籍タイトル/ 配下にする
    """
    docs_service = build("docs", "v1", credentials=creds)
    drive_service = build("drive", "v3", credentials=creds)

    # --- フォルダ構造を作成 ---
    root_folder_id = get_or_create_folder(drive_service, "OCR結果")
    book_folder_id = get_or_create_folder(drive_service, book_title, root_folder_id)

    # --- ドキュメントを作成 ---
    doc_name = f"{book_title} ({chapter_title})"
    doc = docs_service.documents().create(body={"title": doc_name}).execute()
    doc_id = doc.get("documentId")

    # --- ドキュメントをフォルダに移動 ---
    drive_service.files().update(
        fileId=doc_id,
        addParents=book_folder_id,
        fields="id, parents"
    ).execute()

    print(f"{doc_name}: https://docs.google.com/document/d/{doc_id}/edit")

    # --- 本文挿入処理 ---
    requests = []
    cursor = 1

    for line in text.splitlines():
        if not line.strip():
            continue

        m = re.match(r"^\s*(序章|第[0-9一二三四五六七八九十]+章|プロローグ|結論)", line)
        if m:
            chapter_heading = m.group(0).strip()
            rest_text = line[m.end():].strip()

            requests.append({
                "insertText": {"location": {"index": cursor}, "text": chapter_heading + "\n"}
            })
            requests.append({
                "updateParagraphStyle": {
                    "range": {"startIndex": cursor, "endIndex": cursor + len(chapter_heading) + 1},
                    "paragraphStyle": {"namedStyleType": "HEADING_2"},
                    "fields": "namedStyleType"
                }
            })
            cursor += len(chapter_heading) + 1

            if rest_text:
                requests.append({
                    "insertText": {"location": {"index": cursor}, "text": rest_text + "\n"}
                })
                cursor += len(rest_text) + 1

        else:
            requests.append({
                "insertText": {"location": {"index": cursor}, "text": line + "\n"}
            })
            cursor += len(line) + 1

    docs_service.documents().batchUpdate(documentId=doc_id, body={"requests": requests}).execute()
