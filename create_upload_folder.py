# # create_upload_folder.py — RUN ONCE then delete
# import json
# from google.oauth2 import service_account
# from google.auth.transport.requests import Request as GoogleRequest
# import requests
#
# with open("serviceAccountKey.json") as f:
#     info = json.load(f)
#
# creds = service_account.Credentials.from_service_account_info(
#     info, scopes=["https://www.googleapis.com/auth/drive"]
# )
# creds.refresh(GoogleRequest())
# token = creds.token
#
# res = requests.post(
#     "https://www.googleapis.com/drive/v3/files?fields=id",
#     headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
#     json={"name": "Eduket Exam Uploads", "mimeType": "application/vnd.google-apps.folder"}
# )
# folder_id = res.json()["id"]
# print(f"\n✅ SA_UPLOAD_FOLDER_ID={folder_id}\n")
# print("Add this to your .env and Render environment variables")