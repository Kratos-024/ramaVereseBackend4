import io
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import os
secret_json_str = os.getenv("secretKey.json")  

if not secret_json_str:
    possible_paths = [
        "/etc/secrets/secretKey.json",
        "/etc/secrets/secretKey"  
    ]
    secret_json_str = None
    for path in possible_paths:
        if os.path.exists(path):
            with open(path, "r") as f:
                secret_json_str = f.read()
            break

if not secret_json_str:
    raise ValueError("Secret not found in env var or secret files.")

secret_data = json.loads(secret_json_str)

class Drive:
    def __init__(self):
        self.SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
        self.SERVICE_ACCOUNT_FILE = secret_data
        self.PARENT_FOLDER_ID = '12vJKghq9UFsBcstqztUGXk3kji2rs3zi'

    def authorize(self):
        credentials = service_account.Credentials.from_service_account_file(
            self.SERVICE_ACCOUNT_FILE,
            scopes=self.SCOPES
        )
        return credentials

    def download_file(self, auth, file_id, file_name):
        try:
            drive_service = build('drive', 'v3', credentials=auth)
            
            file = drive_service.files().get(fileId=file_id).execute()
            print(f"File found: {file['name']} (MIME type: {file['mimeType']})")

            request = drive_service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)

            done = False
            while not done:
                status, done = downloader.next_chunk()
                print(f"Download {int(status.progress() * 100)}%.")

            fh.seek(0)
            with open(file_name, 'wb') as f:
                f.write(fh.read())

            print(f"File saved as {file_name}")

        except Exception as e:
            print("Error downloading file:", e)


