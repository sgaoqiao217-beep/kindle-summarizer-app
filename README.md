# kindle-summarizer-app
Kindle書籍を自動OCR・要約するStreamlitアプリ

## Google Docs 出力用の認証

デフォルトでは `.streamlit/secrets.toml` の `GOOGLE_CREDENTIALS` に配置したサービスアカウントを使います。
サービスアカウントに Drive ストレージが割り当てられていない環境では、下記のようにユーザーOAuthを設定すると自分の My Drive に出力されます。

```toml
GOOGLE_CREDENTIALS = "{...サービスアカウントJSON...}"  # Vision APIなど他機能で継続使用

[google_oauth]
client_json = """
{
  "installed": {
    "client_id": "xxxxxxxx.apps.googleusercontent.com",
    "project_id": "kindle-summarizer",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_secret": "xxxxxxxx",
    "redirect_uris": ["http://localhost"]
  }
}
"""
```

`client_json` には Google Cloud Console で発行した「デスクトップアプリ」OAuth クライアントの JSON を貼り付けてください。設定すると Drive/Docs はユーザーOAuthで認可され、SAの `storageQuotaExceeded` を回避できます。
