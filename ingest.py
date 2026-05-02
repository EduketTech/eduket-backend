"""
chunk_pdf.py  —  EduCAT PDF Chunking Pipeline with Google Drive Integration  (Universal)

WHAT CHANGED FROM PREVIOUS VERSION
───────────────────────────────────
1. GOOGLE DRIVE SOURCE:
   - Reads PDFs directly from a Google Drive folder
   - Uses service account authentication
   - Downloads PDFs to temp, extracts text, then cleans up

2. PROCESS-ONCE MEMORY:
   - Tracks processed PDFs via Drive file IDs in Firestore
   - Detects modified PDFs via Drive modifiedTime timestamp
   - Stores extraction state per file: pending, processing, completed, failed

3. DRIVE OUTPUT:
   - Saves extracted chunks JSON back to Google Drive
   - Creates folder structure: /EduCAT/processed/{subject}/

4. UNIVERSAL SUBJECT SUPPORT:
   - Detects subject from PDF filename
   - Skips exam papers and memos (only theory/study content)
"""

import os
import io
import re
import json
import hashlib
import tempfile
import traceback
from datetime import datetime, timezone
from typing import Optional
from dotenv import load_dotenv

# -- PDF extraction --
import pdfplumber

# -- Google imports --
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload

load_dotenv()

# -- Configuration --
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

SOURCE_FOLDER_ID = os.getenv("EDUCAT_DRIVE_FOLDER_ID", "")
OUTPUT_FOLDER_ID = os.getenv("EDUCAT_OUTPUT_FOLDER_ID", "")

SUBJECT_PATTERNS = {
    r'\bmathematics\b': 'Mathematics',
    r'\bmaths\b': 'Mathematics',
    r'\bmath\s+lit\b': 'Mathematical Literacy',
    r'\btechnical\s+math\b': 'Technical Mathematics',
    r'\bphysical\s+sciences?\b': 'Physical Sciences',
    r'\blife\s+sciences?\b': 'Life Sciences',
    r'\bgeography\b': 'Geography',
    r'\bhistory\b': 'History',
    r'\baccounting\b': 'Accounting',
    r'\beconomics\b': 'Economics',
    r'\bbusiness\s+studies\b': 'Business Studies',
    r'\bcat\b': 'Computer Applications Technology',
    r'\bit\b': 'Information Technology',
    r'\bengineering\s+graphics\b': 'Engineering Graphics & Design',
    r'\benglish\b': 'English',
    r'\bafrikaans\b': 'Afrikaans',
    r'\bisizulu\b': 'isiZulu',
    r'\bsesotho\b': 'Sesotho',
}

_EXAM_KEYWORDS = ["exam", "paper", "question", "p1", "p2", "p3", "nov", "term", "trial"]
_MEMO_KEYWORDS = ["memo", "memorandum", "answers", "marking"]

# ===============================================================================
# GOOGLE DRIVE & FIRESTORE SERVICES
# ===============================================================================

def _get_drive_service():
    inline_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
    if inline_json and inline_json.strip():
        sa_info = json.loads(inline_json)
        creds = service_account.Credentials.from_service_account_info(
            sa_info, scopes=["https://www.googleapis.com/auth/drive"])
    else:
        sa_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "serviceAccountKey.json")
        creds = service_account.Credentials.from_service_account_file(
            sa_path, scopes=["https://www.googleapis.com/auth/drive"])
    return build("drive", "v3", credentials=creds)

def _get_firestore_client():
    import firebase_admin
    from firebase_admin import firestore as fs_admin
    if not firebase_admin._apps:
        inline_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
        if inline_json and inline_json.strip():
            sa_dict = json.loads(inline_json)
            cred = firebase_admin.credentials.Certificate(sa_dict)
        else:
            cred = firebase_admin.credentials.ApplicationDefault()
        firebase_admin.initialize_app(cred)
    return fs_admin.client()

# ===============================================================================
# TRACKING LOGIC
# ===============================================================================

def _get_processed_files() -> dict:
    try:
        db = _get_firestore_client()
        docs = db.collection("chunk_processed_pdfs").stream()
        return {doc.id: doc.to_dict() for doc in docs}
    except Exception as e:
        print(f"[Chunk] Firestore load error: {e}")
        return {}

def _mark_file_processed(file_id, filename, subject, modified_time, chunks=0, drive_file_id=""):
    try:
        db = _get_firestore_client()
        db.collection("chunk_processed_pdfs").document(file_id).set({
            "filename": filename,
            "subject": subject,
            "status": "completed",
            "modified_time": modified_time,
            "chunks": chunks,
            "output_drive_id": drive_file_id,
            "processed_at": datetime.now(datetime.timezone.utc).isoformat(),
        })
    except Exception as e:
        print(f"[Chunk] Mark processed error: {e}")

def _mark_file_failed(file_id, filename, error):
    try:
        db = _get_firestore_client()
        db.collection("chunk_processed_pdfs").document(file_id).set({
            "filename": filename,
            "status": "failed",
            "error": error[:500],
            "processed_at": datetime.utcnow().isoformat(),
        })
    except Exception as e:
        print(f"[Chunk] Mark failed error: {e}")

# ===============================================================================
# DRIVE OPERATIONS
# ===============================================================================

def _list_pdfs_in_folder(folder_id: str) -> list[dict]:
    if not folder_id: return []
    service = _get_drive_service()
    query = f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false"
    results = []
    page_token = None
    while True:
        response = service.files().list(q=query, spaces="drive", fields="nextPageToken, files(id, name, modifiedTime)", pageToken=page_token).execute()
        results.extend(response.get("files", []))
        page_token = response.get("nextPageToken")
        if not page_token: break
    return results

def _download_pdf(file_id: str, filename: str) -> str:
    service = _get_drive_service()
    request = service.files().get_media(fileId=file_id)
    suffix = os.path.splitext(filename)[1] or ".pdf"
    fd, local_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with io.FileIO(local_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    return local_path

def _upload_to_drive(local_path, filename, parent_folder_id, mime_type="application/json"):
    service = _get_drive_service()
    media = MediaFileUpload(local_path, mimetype=mime_type)
    file_metadata = {"name": filename, "parents": [parent_folder_id]}
    file = service.files().create(body=file_metadata, media_body=media, fields="id").execute()
    return file.get("id")

def _ensure_drive_folder(parent_id, folder_name):
    service = _get_drive_service()
    query = f"'{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and name='{folder_name}' and trashed=false"
    results = service.files().list(q=query, fields="files(id)").execute()
    items = results.get("files", [])
    if items: return items[0]["id"]
    metadata = {"name": folder_name, "mimeType": "application/vnd.google-apps.folder", "parents": [parent_id]}
    folder = service.files().create(body=metadata, fields="id").execute()
    return folder.get("id")

# ===============================================================================
# PROCESSING LOGIC
# ===============================================================================

def _extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text: text += page_text + "\n"
    return text

def _chunk_text(text: str, source_file: str, subject: str) -> list[dict]:
    words = text.split()
    raw_chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    for i in range(0, len(words), step):
        chunk_words = words[i:i + CHUNK_SIZE]
        if not chunk_words: break
        raw_chunks.append(" ".join(chunk_words))
    return [{"source": source_file, "chunk_index": i, "total_chunks": len(raw_chunks), "content": c, "subject": subject} for i, c in enumerate(raw_chunks)]


def process_files():
    print("📚 EduCAT PDF Chunking Pipeline")
    if not SOURCE_FOLDER_ID:
        return {"error": "SOURCE_FOLDER_ID missing"}

    processed_history = _get_processed_files()
    drive_pdfs = _list_pdfs_in_folder(SOURCE_FOLDER_ID)
    print(f"DEBUG: Found {len(drive_pdfs)} total PDFs in Drive")

    for pdf in drive_pdfs:
        fid, fname, mtime = pdf["id"], pdf["name"], pdf.get("modifiedTime", "")

        # 1. Skip if memo or exam
        if any(kw in fname.lower() for kw in _MEMO_KEYWORDS + _EXAM_KEYWORDS):
            print(f"⏩ Skipping (Exam/Memo): {fname}")
            continue

        # 2. Skip if already processed and unchanged
        if fid in processed_history and processed_history[fid].get("modified_time") == mtime:
            if processed_history[fid].get("status") == "completed":
                continue

        print(f"Processing: {fname}")
        try:
            # 3. Download to temporary storage
            path = _download_pdf(fid, fname)

            # 4. Extract and Detect Subject
            text = _extract_text_from_pdf(path)
            subject = "General"
            for patt, sub in SUBJECT_PATTERNS.items():
                if re.search(patt, fname.lower()):
                    subject = sub
                    break

            # 5. Create Chunks
            chunks = _chunk_text(text, fname, subject)

            # 6. SAVE LOCALLY (User-owned server storage)
            # This creates a folder structure like: processed/Mathematics/file_chunks.json
            local_dir = f"processed/{subject}"
            os.makedirs(local_dir, exist_ok=True)
            local_file_path = f"{local_dir}/{fname}_chunks.json"

            with open(local_file_path, "w") as f:
                json.dump(chunks, f, indent=2)

            # 7. UPDATE TRACKING (Firestore)
            # We explicitly pass an empty string for out_fid to avoid Google Storage Quota errors
            try:
                db = _get_firestore_client()
                db.collection("chunk_processed_pdfs").document(fid).set({
                    "filename": fname,
                    "subject": subject,
                    "status": "completed",
                    "modified_time": mtime,
                    "chunks": len(chunks),
                    "local_path": local_file_path,
                    "output_drive_id": "",  # No longer using Drive for output
                    "processed_at": datetime.now(timezone.utc).isoformat(),
                })
            except Exception as fs_err:
                print(f"⚠️ Firestore update failed: {fs_err}")

            # 8. Cleanup temp PDF
            if os.path.exists(path):
                os.unlink(path)

            print(f"✅ Finished {fname} -> Saved to {local_file_path}")

        except Exception as e:
            print(f"❌ Failed {fname}: {e}")
            _mark_file_failed(fid, fname, str(e))
if __name__ == "__main__":
    process_files()