"""
rag.py  —  EduCAT RAG Index with Google Drive Integration

ROLE IN THE SYSTEM
──────────────────
This module handles THEORY BOOKS only — textbooks, study guides, past-paper
explanations — uploaded into a Google Drive folder you designate.

It does NOT handle exam papers. Exam papers are:
  • Uploaded by teachers via the React frontend
  • Stored as Drive file references in Firestore /exams
  • Extracted by scheduled_task.py into /exam_questions
  • Served to students via /start-exam in app.py

RAG's job is to give the AI agent searchable background knowledge so it can
answer "explain capacitors", "what is a LAN?", "define GDP" etc.

WHAT THIS VERSION FIXES
───────────────────────
1. No longer a generator script — this IS rag.py, deploy it directly.
2. FAISS cache stored at ~/educat_faiss/ (persistent on PythonAnywhere,
   not /tmp which is wiped on restart).
3. Subject detection uses word-boundary regex — "cat" no longer matches
   inside "concatenate.pdf".
4. Clear startup message when env vars are missing instead of silent failure.
5. firebase_admin initialisation deferred so it reuses app.py's instance.

ENVIRONMENT VARIABLES
─────────────────────
  GOOGLE_APPLICATION_CREDENTIALS  path to service account JSON
      OR
  FIREBASE_SERVICE_ACCOUNT_JSON   inline JSON string (PythonAnywhere env var)

  HF_TOKEN                        HuggingFace token (sentence-transformers)
  EDUCAT_DRIVE_FOLDER_ID          Drive folder ID containing theory PDFs
  EDUCAT_OUTPUT_FOLDER_ID         Drive folder ID for extracted chunks output
                                  (optional — skipped if blank)
  EDUCAT_FAISS_DIR                Local directory for FAISS cache
                                  (default: ~/educat_faiss)
"""

import os
import io
import re
import json
import hashlib
import tempfile
import traceback
from typing import Optional
from datetime import datetime
from dotenv import load_dotenv

# ── LangChain ────────────────────────────────────────────────────────────────
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# ── Google Drive ─────────────────────────────────────────────────────────────
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload

load_dotenv()

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN is not set. Get one at https://huggingface.co/settings/tokens")

MODEL_NAME    = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200

SOURCE_FOLDER_ID = os.getenv("EDUCAT_DRIVE_FOLDER_ID", "")
OUTPUT_FOLDER_ID = os.getenv("EDUCAT_OUTPUT_FOLDER_ID", "")

# FIX: persistent path — not /tmp which is wiped on PythonAnywhere restart
FAISS_CACHE_DIR = os.getenv(
    "EDUCAT_FAISS_DIR",
    os.path.join(os.path.expanduser("~"), "educat_faiss")
)
os.makedirs(FAISS_CACHE_DIR, exist_ok=True)

# ── Subject patterns ─────────────────────────────────────────────────────────
# FIX: All patterns use \b word-boundaries so "cat" won't match "concatenate"
SUBJECT_PATTERNS = {
    r"\bmathematics\b":             "Mathematics",
    r"\bmaths\b":                   "Mathematics",
    r"\bmath\s+lit\b":              "Mathematical Literacy",
    r"\bmath_lit\b":                "Mathematical Literacy",
    r"\btechnical[\s_]math\b":      "Technical Mathematics",
    r"\bphysical[\s_]sciences?\b":  "Physical Sciences",
    r"\blife[\s_]sciences?\b":      "Life Sciences",
    r"\bgeography\b":               "Geography",
    r"\bhistory\b":                 "History",
    r"\baccounting\b":              "Accounting",
    r"\beconomics\b":               "Economics",
    r"\bbusiness[\s_]studies\b":    "Business Studies",
    r"\bcat\b":                     "Computer Applications Technology",   # word boundary
    r"\b(?:it|info[\s_]tech)\b":    "Information Technology",
    r"\beng[\s_]graphics\b":        "Engineering Graphics & Design",
    r"\benglish\b":                 "English",
    r"\bafrikaans\b":               "Afrikaans",
}

_EXAM_KEYWORDS = [
    "exam", "paper", "question", "p1", "p2", "p3",
    "nov", "november", "may", "june", "feb", "february",
    "march", "mar", "aug", "august", "sep", "september",
    "oct", "october", "term", "trial", "nsc", "dbe",
]
_MEMO_KEYWORDS = ["memo", "memorandum", "answers", "answer_key", "marking"]

# ── Embeddings (module-level singleton) ──────────────────────────────────────
_embeddings = HuggingFaceEndpointEmbeddings(
    model=MODEL_NAME,
    huggingfacehub_api_token=HF_TOKEN,
)


# ═══════════════════════════════════════════════════════════════════════════════
# GOOGLE DRIVE SERVICE
# ═══════════════════════════════════════════════════════════════════════════════

def _get_drive_service():
    """Build authenticated Drive service from service account credentials."""
    inline = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON", "").strip()
    if inline:
        sa_info = json.loads(inline)
        creds = service_account.Credentials.from_service_account_info(
            sa_info, scopes=["https://www.googleapis.com/auth/drive"]
        )
    else:
        sa_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "serviceAccountKey.json")
        creds = service_account.Credentials.from_service_account_file(
            sa_path, scopes=["https://www.googleapis.com/auth/drive"]
        )
    return build("drive", "v3", credentials=creds)


# ═══════════════════════════════════════════════════════════════════════════════
# FIRESTORE TRACKING
# Reuses the firebase_admin instance already initialised by app.py.
# ═══════════════════════════════════════════════════════════════════════════════

def _get_firestore_client():
    """Get Firestore client — reuses existing firebase_admin instance if present."""
    import firebase_admin
    from firebase_admin import firestore as fs_admin, credentials

    if not firebase_admin._apps:
        inline = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON", "").strip()
        if inline:
            cred = credentials.Certificate(json.loads(inline))
        else:
            cred = credentials.ApplicationDefault()
        firebase_admin.initialize_app(cred)

    return fs_admin.client()


def _get_processed_files() -> set:
    """Return set of Drive file IDs already indexed (from Firestore)."""
    try:
        db   = _get_firestore_client()
        docs = db.collection("rag_processed_pdfs").stream()
        return {doc.id for doc in docs}
    except Exception as e:
        print(f"[RAG] Could not load processed files: {e}")
        return set()


def _mark_file_processed(file_id: str, filename: str, subject: str,
                          status: str = "completed", chunks: int = 0):
    try:
        db = _get_firestore_client()
        db.collection("rag_processed_pdfs").document(file_id).set({
            "filename":     filename,
            "subject":      subject,
            "status":       status,
            "chunks":       chunks,
            "processed_at": datetime.utcnow().isoformat(),
        })
    except Exception as e:
        print(f"[RAG] Could not mark file processed: {e}")


def _mark_file_failed(file_id: str, filename: str, error: str):
    try:
        db = _get_firestore_client()
        db.collection("rag_processed_pdfs").document(file_id).set({
            "filename":     filename,
            "status":       "failed",
            "error":        error[:500],
            "processed_at": datetime.utcnow().isoformat(),
        })
    except Exception as e:
        print(f"[RAG] Could not mark file failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# DRIVE FILE OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _list_pdfs_in_folder(folder_id: str) -> list[dict]:
    if not folder_id:
        return []
    service   = _get_drive_service()
    query     = f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false"
    results   = []
    page_token = None
    while True:
        resp = service.files().list(
            q=query, spaces="drive", pageToken=page_token, pageSize=100,
            fields="nextPageToken, files(id, name, modifiedTime, size)"
        ).execute()
        results.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return results


def _download_pdf(file_id: str, filename: str) -> str:
    """Download Drive PDF to a temp file. Caller must delete the file."""
    service = _get_drive_service()
    req     = service.files().get_media(fileId=file_id)
    suffix  = os.path.splitext(filename)[1] or ".pdf"
    fd, local_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with io.FileIO(local_path, "wb") as fh:
        dl = MediaIoBaseDownload(fh, req)
        done = False
        while not done:
            _, done = dl.next_chunk()
    return local_path


def _upload_to_drive(local_path: str, filename: str, parent_id: str,
                     mime_type: str = "application/json") -> Optional[str]:
    if not parent_id:
        return None
    service = _get_drive_service()
    meta    = {"name": filename, "parents": [parent_id]}
    media   = MediaFileUpload(local_path, mimetype=mime_type, resumable=True)
    f       = service.files().create(body=meta, media_body=media, fields="id").execute()
    return f.get("id")


def _ensure_drive_folder(parent_id: str, folder_name: str) -> str:
    service = _get_drive_service()
    q = (f"'{parent_id}' in parents "
         f"and mimeType='application/vnd.google-apps.folder' "
         f"and name='{folder_name}' and trashed=false")
    items = service.files().list(q=q, spaces="drive", fields="files(id)").execute().get("files", [])
    if items:
        return items[0]["id"]
    meta   = {"name": folder_name, "mimeType": "application/vnd.google-apps.folder", "parents": [parent_id]}
    folder = service.files().create(body=meta, fields="id").execute()
    return folder.get("id")


# ═══════════════════════════════════════════════════════════════════════════════
# PDF TEXT EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_subject(filename: str) -> str:
    """Detect subject from filename using word-boundary patterns."""
    lower = filename.lower()
    for pattern, subject in SUBJECT_PATTERNS.items():
        if re.search(pattern, lower):
            return subject
    return "General"


def _is_rag_eligible(filename: str) -> bool:
    """
    Returns True for theory/study content suitable for RAG indexing.
    Returns False for exam papers and memos — those are handled separately
    by the exam extraction pipeline (scheduled_task.py).
    """
    lower = filename.lower()
    if any(kw in lower for kw in _MEMO_KEYWORDS):
        return False
    if any(kw in lower for kw in _EXAM_KEYWORDS):
        return False
    return True


def _extract_text_pdfplumber(pdf_path: str) -> list[dict]:
    try:
        import pdfplumber
        pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and text.strip():
                    pages.append({
                        "page_num": i + 1,
                        "text":     text.strip(),
                        "tables":   page.extract_tables() or [],
                    })
        return pages
    except Exception as e:
        print(f"    [pdfplumber] {e}")
        return []


def _extract_text_pypdf(pdf_path: str) -> list[dict]:
    try:
        from pypdf import PdfReader
        pages = []
        for i, page in enumerate(PdfReader(pdf_path).pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append({"page_num": i + 1, "text": text.strip(), "tables": []})
        return pages
    except Exception as e:
        print(f"    [pypdf] {e}")
        return []


def _extract_pdf(pdf_path: str) -> list[dict]:
    pages = _extract_text_pdfplumber(pdf_path)
    return pages if pages else _extract_text_pypdf(pdf_path)


def _chunk_text(text: str, source: str, page_num: int, subject: str) -> list[Document]:
    if not text or len(text.strip()) < 50:
        return []
    chunks = []
    start  = 0
    tlen   = len(text)
    while start < tlen:
        end = min(start + CHUNK_SIZE, tlen)
        if end < tlen:
            for pos in range(end, max(start + CHUNK_SIZE // 2, end - 200), -1):
                if pos < tlen and text[pos] in ".!?\n":
                    end = pos + 1
                    break
        chunk = text[start:end].strip()
        if chunk:
            cid = hashlib.md5(f"{source}:{page_num}:{start}:{chunk[:100]}".encode()).hexdigest()
            chunks.append(Document(
                page_content=chunk,
                metadata={"source": source, "page_num": page_num,
                           "chunk_id": cid, "subject": subject},
            ))
        start = end - CHUNK_OVERLAP if end < tlen else end
    return chunks


def _process_pdf(file_id: str, filename: str) -> tuple[list[Document], str]:
    subject = _detect_subject(filename)
    print(f"    Subject detected: {subject}")
    print(f"    Downloading...")
    local_path = _download_pdf(file_id, filename)
    try:
        pages = _extract_pdf(local_path)
        if not pages:
            print(f"    No text extracted")
            return [], subject
        print(f"    {len(pages)} pages extracted")
        docs = []
        for page in pages:
            docs.extend(_chunk_text(page["text"], filename, page["page_num"], subject))
        print(f"    {len(docs)} chunks created")
        return docs, subject
    finally:
        try:
            os.unlink(local_path)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# FAISS INDEX
# ═══════════════════════════════════════════════════════════════════════════════

class DriveRAGIndex:
    """
    RAG index backed by Google Drive.

    Theory PDFs → Drive (EDUCAT_DRIVE_FOLDER_ID)
                → downloaded + extracted on first encounter
                → chunks stored in FAISS
                → FAISS persisted to ~/educat_faiss/ (survives PythonAnywhere restarts)
                → processed file IDs tracked in Firestore /rag_processed_pdfs
    """

    def __init__(self):
        self._store: Optional[FAISS] = None
        self._local_index_path = os.path.join(FAISS_CACHE_DIR, "faiss_index")
        self._load_or_build()

    def _load_or_build(self):
        # Try loading existing FAISS index from persistent local cache
        if os.path.exists(self._local_index_path):
            try:
                self._store = FAISS.load_local(
                    self._local_index_path, _embeddings,
                    allow_dangerous_deserialization=True
                )
                print(f"[RAG] Loaded FAISS index from {self._local_index_path}")
            except Exception as e:
                print(f"[RAG] Could not load FAISS cache: {e}")
                self._store = None

        if not SOURCE_FOLDER_ID:
            print("[RAG] WARNING: EDUCAT_DRIVE_FOLDER_ID not set.")
            print("[RAG]   Theory book search will return empty results.")
            print("[RAG]   Set this env var to a Drive folder containing .pdf textbooks.")
            return

        processed_ids = _get_processed_files()
        all_pdfs      = _list_pdfs_in_folder(SOURCE_FOLDER_ID)
        eligible      = [p for p in all_pdfs if _is_rag_eligible(p["name"])]
        new_pdfs      = [p for p in eligible if p["id"] not in processed_ids]

        print(f"[RAG] {len(all_pdfs)} PDFs in folder | "
              f"{len(eligible)} eligible | "
              f"{len(processed_ids)} already processed | "
              f"{len(new_pdfs)} new")

        if not new_pdfs:
            if not self._store:
                print("[RAG] No index and no new PDFs — search will return empty results.")
            return

        all_new_docs: list[Document] = []
        for pdf in new_pdfs:
            fid, fname = pdf["id"], pdf["name"]
            print(f"\n  [RAG] Processing: {fname}")
            try:
                docs, subject = _process_pdf(fid, fname)
                if docs:
                    all_new_docs.extend(docs)
                    _mark_file_processed(fid, fname, subject, "completed", len(docs))
                    if OUTPUT_FOLDER_ID:
                        self._save_chunks_to_drive(docs, fname, subject)
                else:
                    _mark_file_processed(fid, fname, _detect_subject(fname), "empty", 0)
            except Exception as e:
                traceback.print_exc()
                _mark_file_failed(fid, fname, str(e))

        if all_new_docs:
            print(f"\n[RAG] Embedding {len(all_new_docs)} chunks...")
            if self._store is None:
                self._store = FAISS.from_documents(all_new_docs, _embeddings)
            else:
                self._store.add_documents(all_new_docs)
            self._store.save_local(self._local_index_path)
            print(f"[RAG] FAISS index saved to {self._local_index_path}")
            if OUTPUT_FOLDER_ID:
                self._backup_index_to_drive()

        print("[RAG] Index ready.")

    def _save_chunks_to_drive(self, docs: list[Document], filename: str, subject: str):
        try:
            folder_id     = _ensure_drive_folder(OUTPUT_FOLDER_ID, subject)
            chunks_data   = [{
                "content":   d.page_content,
                "source":    d.metadata.get("source", filename),
                "page_num":  d.metadata.get("page_num", 0),
                "chunk_id":  d.metadata.get("chunk_id", ""),
                "subject":   d.metadata.get("subject", subject),
            } for d in docs]
            base          = os.path.splitext(filename)[0]
            fd, tmp_path  = tempfile.mkstemp(suffix=".json")
            os.close(fd)
            with open(tmp_path, "w") as f:
                json.dump(chunks_data, f, indent=2)
            _upload_to_drive(tmp_path, f"{base}_chunks.json", folder_id)
            os.unlink(tmp_path)
            print(f"    Chunks saved to Drive → {subject}/{base}_chunks.json")
        except Exception as e:
            print(f"    Could not save chunks to Drive: {e}")

    def _backup_index_to_drive(self):
        try:
            folder_id = _ensure_drive_folder(OUTPUT_FOLDER_ID, "faiss_index")
            for fname in os.listdir(self._local_index_path):
                fpath = os.path.join(self._local_index_path, fname)
                if os.path.isfile(fpath):
                    _upload_to_drive(fpath, fname, folder_id)
            print("[RAG] FAISS index backed up to Drive.")
        except Exception as e:
            print(f"[RAG] Could not backup index to Drive: {e}")

    def search(self, query: str, k: int = 3, subject_filter: str = "") -> list[dict]:
        """
        Search theory index for content relevant to the query.

        Args:
            query:          Natural language query
            k:              Number of results to return
            subject_filter: Optional subject name to restrict results

        Returns:
            list of {"content", "source", "score", "subject", "page_num"}
        """
        if not self._store:
            return []

        raw = self._store.similarity_search_with_score(query, k=k * 2)
        results = []
        for doc, score in raw:
            results.append({
                "content":  doc.page_content,
                "source":   doc.metadata.get("source", "unknown"),
                "score":    round(float(score), 4),
                "subject":  doc.metadata.get("subject", "General"),
                "page_num": doc.metadata.get("page_num", 0),
            })

        if subject_filter:
            results = [r for r in results if r["subject"].lower() == subject_filter.lower()]

        return results[:k]

    def get_stats(self) -> dict:
        """Return index statistics from Firestore tracking collection."""
        try:
            db      = _get_firestore_client()
            docs    = db.collection("rag_processed_pdfs").stream()
            total   = completed = failed = 0
            by_subj: dict = {}
            for doc in docs:
                d = doc.to_dict()
                total += 1
                status = d.get("status", "")
                if status == "completed": completed += 1
                elif status == "failed":  failed    += 1
                sub = d.get("subject", "Unknown")
                by_subj[sub] = by_subj.get(sub, 0) + 1
            return {"total_pdfs": total, "completed": completed,
                    "failed": failed, "by_subject": by_subj}
        except Exception as e:
            return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API (backward-compatible)
# ═══════════════════════════════════════════════════════════════════════════════

class RAGIndex(DriveRAGIndex):
    """
    Public class — same name and interface as the original RAGIndex.
    All existing calls to RAGIndex() and rag.search() work without changes.
    """
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# MANUAL RUN  (python rag.py)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("EduCAT RAG — Manual PDF Processing Run")
    print("=" * 60)
    print(f"  Source folder : {SOURCE_FOLDER_ID or 'NOT SET'}")
    print(f"  Output folder : {OUTPUT_FOLDER_ID or 'NOT SET (chunks not saved)'}")
    print(f"  FAISS cache   : {FAISS_CACHE_DIR}")
    print()
    index = RAGIndex()
    stats = index.get_stats()
    print(f"\nStats: {json.dumps(stats, indent=2)}")

    # Quick search test
    test_q = "What is a LAN?"
    results = index.search(test_q, k=2)
    print(f"\nTest search: '{test_q}'")
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r['subject']}] p{r['page_num']} score={r['score']}")
        print(f"     {r['content'][:120]}...")
    print("\nDone.")