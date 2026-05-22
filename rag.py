"""
rag.py  —  EduCAT RAG  (Groq-native, zero disk storage)

HOW IT WORKS
────────────
Previous version used FAISS which wrote large binary index files to disk —
a problem on PythonAnywhere's limited storage quota.

This version replaces FAISS entirely with a two-stage Groq approach:

  Stage 1 — Lightweight keyword index (stored in Firestore, ~1–2 KB per chunk)
    When a new theory PDF is detected in Google Drive, the text is extracted,
    split into chunks, and each chunk is stored as a Firestore document in
    /rag_chunks/{chunkId}.  No local files written at all.

  Stage 2 — Groq LLM reranking at query time (in-memory, no disk)
    When search() is called:
      a. Pull candidate chunks from Firestore using simple keyword matching
         (Firestore array-contains on a keywords field we store per chunk).
      b. Send the top N candidates + the query to Groq's llama3 model with a
         reranking prompt — ask it to select and summarise the most relevant.
      c. Return the reranked results.

WHY THIS IS FINE FOR EDUCAT
────────────────────────────
• Theory books for CAPS/IEB subjects are not huge — a few hundred chunks total.
• Firestore reads are fast (<100ms) and the free tier handles thousands/day.
• Groq's llama3-8b-8192 is extremely fast (tokens/sec) and the free tier
  covers well more traffic than a school-sized deployment needs.
• Zero disk usage on PythonAnywhere — the only I/O is temp PDF download
  during extraction, which is deleted immediately after chunking.

ROLE IN THE SYSTEM
──────────────────
This module handles THEORY BOOKS only — textbooks, study guides.
Exam papers are handled by the separate extraction pipeline (scheduled_task.py)
and stored in /exams + /exam_questions in Firestore.

ENVIRONMENT VARIABLES
─────────────────────
  GROQ_API_KEY                    Groq API key (already set for marking)
  FIREBASE_SERVICE_ACCOUNT_JSON   Inline service account JSON (or)
  GOOGLE_APPLICATION_CREDENTIALS  Path to service account file

  EDUCAT_DRIVE_FOLDER_ID          Drive folder containing theory PDFs
  EDUCAT_OUTPUT_FOLDER_ID         Optional: Drive folder for chunk JSON backups
  EDUCAT_RAG_CANDIDATES           How many keyword candidates to pull before
                                  Groq reranking (default: 20)
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

# ── Groq ─────────────────────────────────────────────────────────────────────
from groq import Groq

# ── Google Drive ─────────────────────────────────────────────────────────────
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

load_dotenv()

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

GROQ_API_KEY      = os.getenv("GROQ_API_KEY")
GROQ_RAG_MODEL    = "llama3-8b-8192"   # fast + generous free tier
GROQ_MAX_TOKENS   = 1024

SOURCE_FOLDER_ID  = os.getenv("EDUCAT_DRIVE_FOLDER_ID", "")
OUTPUT_FOLDER_ID  = os.getenv("EDUCAT_OUTPUT_FOLDER_ID", "")
RAG_CANDIDATES    = int(os.getenv("EDUCAT_RAG_CANDIDATES", "20"))

CHUNK_SIZE        = 900    # chars — kept small so Groq context fits many chunks
CHUNK_OVERLAP     = 150

# ── Subject detection (word-boundary safe) ───────────────────────────────────
SUBJECT_PATTERNS = {
    r"\bmathematics\b":             "Mathematics",
    r"\bmaths\b":                   "Mathematics",
    r"\bmath[\s_]lit\b":            "Mathematical Literacy",
    r"\btechnical[\s_]math\b":      "Technical Mathematics",
    r"\bphysical[\s_]sciences?\b":  "Physical Sciences",
    r"\blife[\s_]sciences?\b":      "Life Sciences",
    r"\bgeography\b":               "Geography",
    r"\bhistory\b":                 "History",
    r"\baccounting\b":              "Accounting",
    r"\beconomics\b":               "Economics",
    r"\bbusiness[\s_]studies\b":    "Business Studies",
    r"\bcat\b":                     "Computer Applications Technology",
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

# ── Groq client (module-level) ────────────────────────────────────────────────
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set.")
_groq = Groq(api_key=GROQ_API_KEY)


# ═══════════════════════════════════════════════════════════════════════════════
# FIRESTORE
# ═══════════════════════════════════════════════════════════════════════════════

def _get_db():
    """Return Firestore client — reuses firebase_admin if already initialised."""
    import firebase_admin
    from firebase_admin import firestore as fsa, credentials

    if not firebase_admin._apps:
        inline = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON", "").strip()
        cred = (credentials.Certificate(json.loads(inline))
                if inline else credentials.ApplicationDefault())
        firebase_admin.initialize_app(cred)

    return fsa.client()


# ─── Processed-file tracking ──────────────────────────────────────────────────

def _get_processed_ids() -> set:
    try:
        return {d.id for d in _get_db().collection("rag_processed_pdfs").stream()}
    except Exception as e:
        print(f"[RAG] Could not load processed IDs: {e}")
        return set()


def _mark_processed(file_id: str, filename: str, subject: str,
                    status: str = "completed", chunks: int = 0):
    try:
        _get_db().collection("rag_processed_pdfs").document(file_id).set({
            "filename":     filename,
            "subject":      subject,
            "status":       status,
            "chunks":       chunks,
            "processed_at": datetime.utcnow().isoformat(),
        })
    except Exception as e:
        print(f"[RAG] Could not mark processed: {e}")


def _mark_failed(file_id: str, filename: str, error: str):
    try:
        _get_db().collection("rag_processed_pdfs").document(file_id).set({
            "filename":     filename,
            "status":       "failed",
            "error":        error[:500],
            "processed_at": datetime.utcnow().isoformat(),
        })
    except Exception as e:
        print(f"[RAG] Could not mark failed: {e}")


# ─── Chunk storage ────────────────────────────────────────────────────────────

def _save_chunks_to_firestore(chunks: list[dict]):
    """
    Store extracted text chunks in Firestore /rag_chunks.
    Each document:
      content   : str   — the raw chunk text
      source    : str   — filename it came from
      subject   : str   — detected subject
      page_num  : int
      keywords  : list  — cleaned lowercase words for keyword matching
      chunk_id  : str   — md5 hash (document ID)
    """
    db    = _get_db()
    batch = db.batch()
    count = 0

    for chunk in chunks:
        ref = db.collection("rag_chunks").document(chunk["chunk_id"])
        batch.set(ref, chunk, merge=True)
        count += 1

        # Firestore batches are limited to 500 operations
        if count % 400 == 0:
            batch.commit()
            batch = db.batch()

    if count % 400 != 0:
        batch.commit()


def _keyword_search_firestore(keywords: list[str], subject_filter: str = "",
                               limit: int = RAG_CANDIDATES) -> list[dict]:
    """
    Pull candidate chunks from Firestore using keyword matching.

    Strategy: run one query per keyword (Firestore array-contains is single-key),
    collect results, deduplicate by chunk_id, return top `limit` by keyword hits.
    """
    db       = _get_db()
    seen     = {}      # chunk_id → (chunk_dict, hit_count)

    for kw in keywords[:8]:   # cap at 8 keywords to limit read ops
        try:
            q = db.collection("rag_chunks").where("keywords", "array_contains", kw)
            if subject_filter:
                q = q.where("subject", "==", subject_filter)
            q = q.limit(limit)
            for doc in q.stream():
                d = doc.to_dict()
                cid = d.get("chunk_id", doc.id)
                if cid in seen:
                    seen[cid] = (seen[cid][0], seen[cid][1] + 1)
                else:
                    seen[cid] = (d, 1)
        except Exception as e:
            print(f"[RAG] Keyword query error for '{kw}': {e}")

    # Sort by hit count descending, return top candidates
    ranked = sorted(seen.values(), key=lambda x: x[1], reverse=True)
    return [item[0] for item in ranked[:limit]]


# ═══════════════════════════════════════════════════════════════════════════════
# GOOGLE DRIVE
# ═══════════════════════════════════════════════════════════════════════════════
def _get_drive():
    sa_json = (
        os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON") or
        os.getenv("FIREBASE_SERVICE_ACCOUNT")
    )

    if sa_json:
        info = json.loads(sa_json.strip())
    else:
        # Read from file path
        creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "serviceAccountKey.json")
        with open(creds_path) as f:
            info = json.load(f)

    creds = service_account.Credentials.from_service_account_info(
        info,
        scopes=["https://www.googleapis.com/auth/drive.readonly"]
    )
    return build("drive", "v3", credentials=creds)


def _list_pdfs(folder_id: str) -> list[dict]:
    if not folder_id:
        return []
    svc        = _get_drive()
    q          = f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false"
    results    = []
    page_token = None
    while True:
        resp = svc.files().list(
            q=q, spaces="drive", pageToken=page_token, pageSize=100,
            fields="nextPageToken, files(id, name, size)"
        ).execute()
        results.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return results


def _download_pdf(file_id: str, filename: str) -> str:
    """Download PDF to temp file. Caller must delete after use."""
    svc        = _get_drive()
    req        = svc.files().get_media(fileId=file_id)
    suffix     = os.path.splitext(filename)[1] or ".pdf"
    fd, path   = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with io.FileIO(path, "wb") as fh:
        dl = MediaIoBaseDownload(fh, req)
        done = False
        while not done:
            _, done = dl.next_chunk()
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# PDF EXTRACTION + CHUNKING
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_subject(filename: str) -> str:
    lower = filename.lower()
    for pattern, subject in SUBJECT_PATTERNS.items():
        if re.search(pattern, lower):
            return subject
    return "General"


def _is_rag_eligible(filename: str) -> bool:
    """Theory books only — exclude exam papers and memos."""
    lower = filename.lower()
    if any(kw in lower for kw in _MEMO_KEYWORDS):
        return False
    if any(kw in lower for kw in _EXAM_KEYWORDS):
        return False
    return True


def _extract_pages(pdf_path: str) -> list[dict]:
    """Extract page text — tries pdfplumber, falls back to pypdf."""
    # Try pdfplumber
    try:
        import pdfplumber
        pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append({"page_num": i + 1, "text": text.strip()})
        if pages:
            return pages
    except Exception as e:
        print(f"    [pdfplumber] {e}")

    # Fallback to pypdf
    try:
        from pypdf import PdfReader
        pages = []
        for i, page in enumerate(PdfReader(pdf_path).pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"page_num": i + 1, "text": text.strip()})
        return pages
    except Exception as e:
        print(f"    [pypdf] {e}")
        return []


def _extract_keywords(text: str) -> list[str]:
    """
    Extract meaningful keywords from chunk text for Firestore keyword search.
    Returns lowercase words, 4+ chars, deduplicated, max 40.
    """
    words = re.findall(r"[a-zA-Z]{4,}", text.lower())
    # Remove common stop words
    stops = {
        "that", "this", "with", "from", "have", "will", "been",
        "they", "were", "when", "what", "which", "their", "there",
        "about", "would", "could", "should", "also", "into", "some",
        "more", "than", "then", "each", "such", "only", "most",
    }
    filtered = [w for w in words if w not in stops]
    # Deduplicate preserving order, take top 40
    seen  = set()
    uniq  = []
    for w in filtered:
        if w not in seen:
            seen.add(w)
            uniq.append(w)
        if len(uniq) >= 40:
            break
    return uniq


def _chunk_page(text: str, source: str, page_num: int,
                subject: str) -> list[dict]:
    """Split page text into overlapping chunks stored as plain dicts."""
    if not text or len(text.strip()) < 50:
        return []

    chunks = []
    start  = 0
    tlen   = len(text)

    while start < tlen:
        end = min(start + CHUNK_SIZE, tlen)
        # Try to break at sentence boundary
        if end < tlen:
            for pos in range(end, max(start + CHUNK_SIZE // 2, end - 150), -1):
                if text[pos] in ".!?\n":
                    end = pos + 1
                    break
        content = text[start:end].strip()
        if content:
            cid = hashlib.md5(
                f"{source}:{page_num}:{start}:{content[:80]}".encode()
            ).hexdigest()
            chunks.append({
                "chunk_id":  cid,
                "content":   content,
                "source":    source,
                "subject":   subject,
                "page_num":  page_num,
                "keywords":  _extract_keywords(content),
            })
        start = end - CHUNK_OVERLAP if end < tlen else end

    return chunks


def _process_pdf(file_id: str, filename: str) -> tuple[list[dict], str]:
    """
    Download, extract and chunk a single PDF.
    Temp file is deleted immediately after extraction.
    Returns (chunks, subject).
    """
    subject    = _detect_subject(filename)
    local_path = _download_pdf(file_id, filename)
    try:
        pages = _extract_pages(local_path)
        if not pages:
            print(f"    No text extracted from {filename}")
            return [], subject

        print(f"    {len(pages)} pages | subject: {subject}")
        all_chunks = []
        for page in pages:
            all_chunks.extend(
                _chunk_page(page["text"], filename, page["page_num"], subject)
            )
        print(f"    {len(all_chunks)} chunks created")
        return all_chunks, subject
    finally:
        try:
            os.unlink(local_path)    # delete temp file immediately
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# GROQ RERANKER
# ═══════════════════════════════════════════════════════════════════════════════

def _groq_rerank(query: str, candidates: list[dict], k: int = 4) -> list[dict]:
    """
    Send candidate chunks to Groq and ask it to select + paraphrase the
    most relevant ones for the query.

    Returns a list of result dicts with keys:
      content, source, subject, page_num, score (1-10 from model)
    """
    if not candidates:
        return []

    # Build numbered context block — trim context to avoid hitting token limits
    context_lines = []
    for i, c in enumerate(candidates, 1):
        preview = c.get("content", "")[:400].replace("\n", " ")
        context_lines.append(
            f"[{i}] Source: {c.get('source','?')} | "
            f"Subject: {c.get('subject','?')} | "
            f"Page: {c.get('page_num','?')}\n{preview}"
        )
    context_block = "\n\n".join(context_lines)

    prompt = f"""You are a study assistant for South African high school students.

A student asked: "{query}"

Below are {len(candidates)} text excerpts from theory books.
Select the {k} most relevant excerpts and for each one:
1. Give it a relevance score from 1-10.
2. Write a concise 1-3 sentence explanation using that excerpt's content.

Respond in JSON only — no extra text — as a list:
[
  {{
    "index": <1-based index from the list above>,
    "score": <1-10>,
    "explanation": "<concise explanation using the excerpt>"
  }},
  ...
]

EXCERPTS:
{context_block}
"""

    try:
        resp = _groq.chat.completions.create(
            model=GROQ_RAG_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=GROQ_MAX_TOKENS,
            temperature=0.1,
        )
        raw = resp.choices[0].message.content.strip()

        # Strip markdown code fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*$", "", raw)

        ranked = json.loads(raw)
        results = []
        for item in ranked:
            idx = item.get("index", 1) - 1
            if 0 <= idx < len(candidates):
                c = candidates[idx]
                results.append({
                    "content":  item.get("explanation", c.get("content", ""))[:600],
                    "source":   c.get("source", ""),
                    "subject":  c.get("subject", ""),
                    "page_num": c.get("page_num", 0),
                    "score":    item.get("score", 5),
                })
        # Sort by Groq's score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]

    except json.JSONDecodeError:
        # Fixes structural bug: Use full untruncated content on JSON parse failures
        print("[RAG] Groq rerank JSON parse failed — falling back to candidate structures.")
        return [
            {
                "content":  c.get("content", ""),
                "source":   c.get("source", ""),
                "subject":  c.get("subject", ""),
                "page_num": c.get("page_num", 0),
                "score":    5,
            }
            for c in candidates[:k]
        ]

    except Exception as e:
        print(f"[RAG] Groq rerank error: {e}")
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class RAGIndex:
    """
    Zero-disk RAG index.

    Extraction (one-time per PDF):
      Drive PDF → extract text → chunk → store in Firestore /rag_chunks
      Processed file IDs tracked in /rag_processed_pdfs

    Search (each query):
      Query → keyword extraction → Firestore keyword fetch →
      Groq reranking → return top-k results

    No FAISS, no local index files, no HuggingFace token needed.
    The only I/O on disk is a short-lived temp file during PDF download,
    deleted immediately after chunking.
    """

    def __init__(self):
        self._check_env()
        self._ingest_new_pdfs()

    def _check_env(self):
        if not SOURCE_FOLDER_ID:
            print("[RAG] WARNING: EDUCAT_DRIVE_FOLDER_ID not set.")
            print("[RAG]   No PDFs will be indexed. Set this env var to a")
            print("[RAG]   Google Drive folder containing theory textbooks.")
        else:
            print(f"[RAG] Source folder: {SOURCE_FOLDER_ID}")

    def _ingest_new_pdfs(self):
        """Check Drive folder for new PDFs and index any not yet processed."""
        if not SOURCE_FOLDER_ID:
            return

        processed   = _get_processed_ids()
        all_pdfs    = _list_pdfs(SOURCE_FOLDER_ID)
        eligible    = [p for p in all_pdfs if _is_rag_eligible(p["name"])]
        new_pdfs    = [p for p in eligible if p["id"] not in processed]

        print(
            f"[RAG] {len(all_pdfs)} PDFs in Drive | "
            f"{len(eligible)} eligible | "
            f"{len(processed)} already indexed | "
            f"{len(new_pdfs)} new"
        )

        for pdf in new_pdfs:
            fid, fname = pdf["id"], pdf["name"]
            print(f"\n  [RAG] Indexing: {fname}")
            try:
                chunks, subject = _process_pdf(fid, fname)
                if chunks:
                    print(f"    Saving {len(chunks)} chunks to Firestore...")
                    _save_chunks_to_firestore(chunks)
                    _mark_processed(fid, fname, subject, "completed", len(chunks))
                    print(f"    Done.")
                else:
                    _mark_processed(fid, fname, _detect_subject(fname), "empty", 0)
            except Exception as e:
                traceback.print_exc()
                _mark_failed(fid, fname, str(e))

        print("[RAG] Ingestion complete.")

    def search(self, query: str, k: int = 4,
               subject_filter: str = "") -> list[dict]:
        """
        Search theory content relevant to query.

        Args:
            query:          Natural language question or topic
            k:              Number of results to return (default 4)
            subject_filter: Limit to a subject e.g. "Mathematics" (optional)

        Returns:
            list of dicts:
              content   : str   explanation from Groq using the source chunk
              source    : str   filename the chunk came from
              subject   : str   detected subject
              page_num  : int
              score     : int   Groq relevance score 1-10
        """
        if not query or not query.strip():
            return []

        # Stage 1: keyword extraction from query
        query_keywords = _extract_keywords(query)
        if not query_keywords:
            # Fallback: split query into words
            query_keywords = [w.lower() for w in query.split() if len(w) >= 4]

        # Stage 2: Firestore keyword fetch
        candidates = _keyword_search_firestore(
            query_keywords,
            subject_filter=subject_filter,
            limit=RAG_CANDIDATES,
        )

        if not candidates:
            return []

        # Stage 3: Groq reranking
        return _groq_rerank(query, candidates, k=k)

    def get_stats(self) -> dict:
        """Return indexing statistics."""
        try:
            db       = _get_db()
            proc     = list(db.collection("rag_processed_pdfs").stream())
            chunks   = db.collection("rag_chunks").count().get()
            total    = len(proc)
            done     = sum(1 for d in proc if d.to_dict().get("status") == "completed")
            failed   = sum(1 for d in proc if d.to_dict().get("status") == "failed")
            by_subj  = {}
            for d in proc:
                sub = d.to_dict().get("subject", "Unknown")
                by_subj[sub] = by_subj.get(sub, 0) + 1
            return {
                "total_pdfs":    total,
                "completed":     done,
                "failed":        failed,
                "total_chunks":  chunks[0][0].value if chunks else "?",
                "by_subject":    by_subj,
            }
        except Exception as e:
            return {"error": str(e)}

    def reindex(self):
        """
        Force re-check of Drive for new PDFs.
        Call this from a scheduled task or manually if you add new books.
        """
        print("[RAG] Reindexing...")
        self._ingest_new_pdfs()


# ═══════════════════════════════════════════════════════════════════════════════
# MANUAL RUN  (python rag.py)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("EduCAT RAG — Manual Run")
    print("=" * 60)
    print(f"  Source folder : {SOURCE_FOLDER_ID or 'NOT SET'}")
    print(f"  Output folder : {OUTPUT_FOLDER_ID or 'not set'}")
    print(f"  Groq model    : {GROQ_RAG_MODEL}")
    print(f"  Candidates    : {RAG_CANDIDATES} per query")
    print()

    rag   = RAGIndex()
    stats = rag.get_stats()
    print(f"\nStats: {json.dumps(stats, indent=2)}")

    # Interactive test
    while True:
        q = input("\nSearch query (blank to quit): ").strip()
        if not q:
            break
        results = rag.search(q, k=3)
        if not results:
            print("  No results.")
        for i, r in enumerate(results, 1):
            print(f"\n  {i}. [{r['subject']}] {r['source']} p{r['page_num']} "
                  f"score={r['score']}/10")
            print(f"     {r['content'][:200]}...")