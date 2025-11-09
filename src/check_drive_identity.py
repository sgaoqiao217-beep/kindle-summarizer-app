# scripts/check_drive_identity.py
from googleapiclient.discovery import build
from google_api_utils import get_credentials  # 既存の関数を流用

def main():
    creds = get_credentials(use_user_oauth=True)
    drive = build("drive", "v3", credentials=creds)
    me = drive.about().get(fields="user(emailAddress)").execute()
    print("Authenticated as:", me["user"]["emailAddress"])

if __name__ == "__main__":
    main()
