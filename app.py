"""
app.py — EduCAT Flask API (Auto-Extraction Prototype)

WHAT'S NEW:
  - Auto-extraction: when teacher uploads, extraction triggers automatically
  - Word doc support: .docx files extracted via python-docx (no OCR needed, perfect text)
  - PDF support: native text first, Groq vision OCR fallback for scanned PDFs
  - No manual trigger needed — teacher uploads then POST /auto-extract is called

FLOW:
  1. Teacher uploads via frontend → doc saved to teacherExamUploads in Firestore
  2. Frontend calls POST /auto-extract immediately after upload
  3. Backend downloads file from Drive, extracts text, parses questions with Groq
  4. Questions written to exam_questions, exam doc set to status: ready
  5. Students see exam in picker immediately

COLLECTIONS:
  teacherExamUploads/{teacherId}  uploads[] — upload metadata (nested array)
  exams/{examId}                  — exam metadata, status: ready
  exam_questions/{examId}_{i}     — individual questions with memos
"""

from dotenv import load_dotenv
load_dotenv()

import os, io, json, uuid, re, traceback, threading, base64
from datetime import datetime

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests as http_requests
import threading
import time

# ── Firebase ──────────────────────────────────────────────────────────────────
import firebase_admin
from firebase_admin import credentials, firestore as fs_admin

def _init_firebase():
    if firebase_admin._apps:
        return
    raw = os.getenv("FIREBASE_SERVICE_ACCOUNT", "")
    if raw.strip():
        cred = credentials.Certificate(json.loads(raw))
    else:
        cred = credentials.ApplicationDefault()
    firebase_admin.initialize_app(cred)

_init_firebase()
db = fs_admin.client()

# ── App modules ───────────────────────────────────────────────────────────────
from model import generate_answer, mark_answer, generate_exam_feedback
from rag import RAGIndex
import memory as mem
import agent
from agent import run_agent

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": [
    "http://localhost:3000",
    "http://localhost:5174",
    "http://localhost:5175",
    "http://localhost:5176",
    "http://localhost:5177",
    "https://eduket.netlify.app",
]}})

rag = RAGIndex()
agent.set_rag(rag)

# ═══════════════════════════════════════════════════════════════════════════════
# DRIVE DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════════════
def _drive_token() -> str:
    from google.oauth2 import service_account
    from google.auth.transport.requests import Request as GoogleRequest

    sa_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON") or os.getenv("FIREBASE_SERVICE_ACCOUNT")

    if sa_json:
        info = json.loads(sa_json.strip())
    else:
        creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "serviceAccountKey.json")
        with open(creds_path) as f:
            info = json.load(f)

    creds = service_account.Credentials.from_service_account_info(
        info,
        scopes=["https://www.googleapis.com/auth/drive.readonly"]
    )
    creds.refresh(GoogleRequest())
    return creds.token



def download_file_bytes(file_id: str, filename: str = ""):
    """Download file from Drive — handles both regular files and native Google Docs."""
    try:
        token = _drive_token()

        # First check what type of file this is
        meta_res = http_requests.get(
            f"https://www.googleapis.com/drive/v3/files/{file_id}?fields=mimeType,name",
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
        mime = meta_res.json().get("mimeType", "") if meta_res.status_code == 200 else ""
        print(f"[Drive] File mime type: {mime}")

        # Google Docs → export as docx
        if mime == "application/vnd.google-apps.document":
            print(f"[Drive] Google Doc detected — exporting as docx")
            res = http_requests.get(
                f"https://www.googleapis.com/drive/v3/files/{file_id}/export"
                f"?mimeType=application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                headers={"Authorization": f"Bearer {token}"},
                timeout=120,
            )
        # Google Sheets → export as xlsx (future use)
        elif mime == "application/vnd.google-apps.spreadsheet":
            res = http_requests.get(
                f"https://www.googleapis.com/drive/v3/files/{file_id}/export"
                f"?mimeType=application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={"Authorization": f"Bearer {token}"},
                timeout=120,
            )
        # Regular file (pdf, docx uploaded directly) → download as-is
        else:
            res = http_requests.get(
                f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media",
                headers={"Authorization": f"Bearer {token}"},
                timeout=120,
            )

        if res.status_code == 200:
            print(f"[Drive] Downloaded {len(res.content)} bytes for {file_id}")
            return res.content

        print(f"[Drive] Failed {res.status_code}: {res.text[:300]}")
        return None

    except Exception as e:
        print(f"[Drive] Error: {e}")
        return None
# ═══════════════════════════════════════════════════════════════════════════════
# TEXT EXTRACTION — WORD DOC
# ═══════════════════════════════════════════════════════════════════════════════

def extract_text_from_docx(file_bytes: bytes) -> str:
    try:
        from docx import Document
        from docx.oxml.ns import qn

        doc = Document(io.BytesIO(file_bytes))
        lines = []

        def collect_text(element) -> str:
            """Recursively collect all w:t text from any element."""
            return "".join(
                node.text or ""
                for node in element.iter()
                if node.tag in (
                    qn("w:t"),
                    "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"
                )
            )

        for element in doc.element.body:
            tag = element.tag.split("}")[-1] if "}" in element.tag else element.tag

            if tag == "p":
                text = collect_text(element).strip()
                if text:
                    lines.append(text)

            elif tag == "tbl":
                for row in element.iter(qn("w:tr")):
                    cells = [
                        collect_text(cell).strip()
                        for cell in row.iter(qn("w:tc"))
                    ]
                    row_text = " | ".join(c for c in cells if c)
                    if row_text:
                        lines.append(row_text)

        # Also grab text boxes (drawing objects) — common in SA exam papers
        for txbx in doc.element.body.iter(qn("w:txbxContent")):
            text = collect_text(txbx).strip()
            if text:
                lines.append(text)

        # Also grab headers and footers
        for section in doc.sections:
            for hdr in [section.header, section.first_page_header, section.even_page_header]:
                try:
                    text = hdr.text.strip()
                    if text:
                        lines.append(text)
                except Exception:
                    pass

        text = "\n".join(lines)
        print(f"[DOCX] Extracted {len(text)} chars, {len(lines)} lines")
        return text

    except Exception as e:
        traceback.print_exc()
        print(f"[DOCX] Error: {e}")
        return ""

# ═══════════════════════════════════════════════════════════════════════════════
# TEXT EXTRACTION — PDF
# ═══════════════════════════════════════════════════════════════════════════════

def extract_text_from_pdf(file_bytes: bytes, max_pages: int = 20) -> str:
    text = ""

    # Stage 1: pymupdf native text
    try:
        import fitz
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
        if len(text.strip()) > 200:
            print(f"[PDF] pymupdf native: {len(text)} chars")
            return text
    except Exception as e:
        print(f"[PDF] pymupdf: {e}")

    # Stage 2: pdfplumber
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                text += (page.extract_text() or "") + "\n"
        if len(text.strip()) > 200:
            print(f"[PDF] pdfplumber: {len(text)} chars")
            return text
    except Exception as e:
        print(f"[PDF] pdfplumber: {e}")

    # Stage 3: Groq vision OCR for scanned PDFs
    print("[PDF] Scanned — using Groq vision OCR")
    return _groq_vision_ocr(file_bytes, max_pages)


def _groq_vision_ocr(pdf_bytes: bytes, max_pages: int = 20) -> str:
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    all_text = ""
    try:
        import fitz
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages_to_process = min(max_pages, len(doc))
        print(f"[OCR] {pages_to_process}/{len(doc)} pages")
        for i in range(pages_to_process):
            try:
                page = doc[i]
                mat = fitz.Matrix(150 / 72, 150 / 72)
                pix = page.get_pixmap(matrix=mat)
                img_b64 = base64.b64encode(pix.tobytes("png")).decode()
                resp = client.chat.completions.create(
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    messages=[{"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                        {"type": "text", "text": (
                            "South African NSC/CAPS exam page. Extract ALL text exactly. "
                            "Preserve question numbers (1.1, 1.2, 2.1.1), marks in brackets like (2), "
                            "MCQ options A B C D, table contents row by row. "
                            "For diagrams write [DIAGRAM: description]. Plain text only."
                        )}
                    ]}],
                    max_tokens=2000,
                )
                page_text = resp.choices[0].message.content.strip()
                all_text += f"\n--- PAGE {i+1} ---\n{page_text}\n"
                print(f"[OCR] Page {i+1}: {len(page_text)} chars")
            except Exception as e:
                print(f"[OCR] Page {i+1} failed: {e}")
        doc.close()
    except Exception as e:
        print(f"[OCR] fitz error: {e}")
    return all_text


def extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    name = filename.lower()
    if name.endswith(".docx") or name.endswith(".doc"):
        print(f"[Extract] Word doc: {filename}")
        return extract_text_from_docx(file_bytes)   # done — no PDF fallback
    elif name.endswith(".pdf"):
        print(f"[Extract] PDF: {filename}")
        return extract_text_from_pdf(file_bytes)
    else:
        print(f"[Extract] Unknown type {filename} — trying docx then pdf")
        text = extract_text_from_docx(file_bytes)
        return text if text.strip() else extract_text_from_pdf(file_bytes)


# ═══════════════════════════════════════════════════════════════════════════════
# QUESTION PARSER
# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL QUESTION PARSER — handles ANY exam structure
# ═══════════════════════════════════════════════════════════════════════════════

def parse_questions_universal(text: str, subject: str, grade: str) -> list:
    """
    Universal parser that handles ANY South African exam structure:
    - MCQ with run-together options (A. textB. textC. text)
    - MCQ with options on separate lines
    - Numbered sub-questions (1.1, 1.2, 2.1.1)
    - Unnumbered MCQ blocks (Question 1, 2, 3...)
    - True/False with correction
    - Matching/Column A & B
    - Scenario-based multi-part questions
    - Essay and long answer
    - Calculation with working
    - Table completion
    - Mixed sections in one paper

    Strategy: 3-pass approach
      Pass 1 — Pre-process: normalise text structure
      Pass 2 — Groq parse with universal prompt
      Pass 3 — Post-process: validate, fill gaps, ensure every question appears
    """
    if not text or not text.strip():
        return []

    # ── Pass 1: Pre-process text ───────────────────────────────────────────
    text = _preprocess_exam_text(text)

    # ── Pass 2: Chunk and parse ────────────────────────────────────────────
    CHUNK = 10000
    OVERLAP = 800   # larger overlap to catch questions that span chunk boundaries
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + CHUNK])
        start += CHUNK - OVERLAP

    all_questions = []
    seen_numbers = set()

    for i, chunk in enumerate(chunks):
        print(f"[Universal] Chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")
        qs = _parse_any_structure(chunk, subject, grade)
        for q in qs:
            qn = _normalise_qnum(q.get("question_number", ""))
            if qn and qn not in seen_numbers:
                seen_numbers.add(qn)
                all_questions.append(q)
            elif not qn:
                # No question number — still include (e.g. instructions)
                all_questions.append(q)

    # ── Pass 3: Post-process ───────────────────────────────────────────────
    all_questions = _postprocess_questions(all_questions, text)

    print(f"[Universal] Final: {len(all_questions)} questions")
    return all_questions


def _preprocess_exam_text(text: str) -> str:
    """
    Normalise exam text before sending to Groq.
    Fixes common issues: run-together MCQ options, missing newlines, etc.
    """
    # Insert newline before MCQ option letters when run together
    # e.g. "...network?A. Personal" → "...network?\nA. Personal"
    text = re.sub(r'([a-z\?\.\,\)])\s*([A-D])\.\s', r'\1\n\2. ', text)

    # Insert newline before question numbers when run together
    # e.g. "...system. 2.1 Define" → "...system.\n2.1 Define"
    text = re.sub(r'(\w[\.\,])\s+(\d+\.\d+)', r'\1\n\2', text)

    # Ensure QUESTION headings are on their own line
    text = re.sub(r'([^\n])(QUESTION\s+\d+)', r'\1\n\2', text)

    # Normalise multiple spaces/tabs to single space
    text = re.sub(r'[ \t]{2,}', ' ', text)

    # Ensure marks indicators are clean e.g. "(2 marks)" → "(2)"
    text = re.sub(r'\((\d+)\s+marks?\)', r'(\1)', text, flags=re.IGNORECASE)

    return text.strip()


def _parse_any_structure(text: str, subject: str, grade: str) -> list:
    """
    Single Groq call with a universal prompt that handles any structure.
    """
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    prompt = f"""You are an expert at parsing South African CAPS/NSC/IEB exam papers.

Your job: extract EVERY question from the text into a JSON array.
Handle ANY structure — do not skip anything.

═══ STRUCTURE DETECTION RULES ═══

MCQ (Multiple Choice):
- Options may be on separate lines OR run together like "A. textB. textC. text"
- Always split into separate A/B/C/D keys in options object
- Question text = everything BEFORE the first option letter
- type = "mcq"

NUMBERED QUESTIONS without numbers (QUESTION 1 block with no sub-numbers):
- Number them sequentially as 1.1, 1.2, 1.3 etc within that question block

TRUE/FALSE:
- type = "true_false"
- Include the statement as question text

MATCHING (Column A / Column B):
- type = "matching"
- column_a = list of items from Column A
- column_b = list of items from Column B

SCENARIO/CONTEXT QUESTIONS:
- The scenario/context text goes in parent_context field
- Each sub-question is a separate item with the same parent_question

CALCULATION:
- type = "calculation"
- Include any given values in the question text

ESSAY / PARAGRAPH:
- type = "essay"
- Include word/line count if specified

SHORT ANSWER (1-3 sentences):
- type = "short_answer"

DEFAULT:
- type = "open"

═══ OUTPUT FORMAT ═══

Return ONLY a valid JSON array, nothing else. No markdown, no explanation.

Each item:
{{
  "question_number": "1.1",        // string — use section.sub format
  "parent_question": "QUESTION 1", // string — main question heading
  "parent_context": null,          // string or null — scenario/passage text
  "section": "A",                  // string — section letter or number
  "question": "Full question text here (no options for MCQ)",
  "type": "mcq",                   // see types above
  "marks": 2,                      // integer — from brackets (2) or default 1
  "options": {{"A":"...","B":"...","C":"...","D":"..."}}, // MCQ only, else null
  "column_a": null,                // matching only
  "column_b": null,                // matching only
  "memo": null                     // answer if visible in text, else null
}}

Subject: {subject} | Grade: {grade}

═══ EXAM TEXT ═══
{text}
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8000,
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()

        # Extract JSON array even if wrapped in explanation text
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\s*```\s*$", "", raw, flags=re.MULTILINE)
        raw = raw.strip()

        # Find outermost JSON array
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            raw = match.group(0)

        result = json.loads(raw)
        return result if isinstance(result, list) else []

    except json.JSONDecodeError as e:
        print(f"[Universal] JSON error: {e}")
        print(f"[Universal] Raw (first 300): {raw[:300]}")
        # Try to salvage partial JSON
        return _salvage_partial_json(raw)
    except Exception as e:
        print(f"[Universal] Error: {e}")
        return []


def _salvage_partial_json(raw: str) -> list:
    """
    If Groq returns truncated JSON, salvage whatever complete objects we can.
    Finds all complete {...} objects within the array.
    """
    questions = []
    # Find all complete JSON objects
    for match in re.finditer(r'\{[^{}]*\}', raw, re.DOTALL):
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict) and obj.get("question"):
                questions.append(obj)
        except Exception:
            continue
    print(f"[Universal] Salvaged {len(questions)} questions from partial JSON")
    return questions


def _postprocess_questions(questions: list, original_text: str) -> list:
    """
    Post-process parsed questions:
    1. Fill missing fields with defaults
    2. Ensure marks are integers
    3. Detect and fix MCQ options that ended up in question text
    4. Sort by question number
    5. Remove duplicates
    """
    processed = []
    seen = set()

    for q in questions:
        # Fill defaults
        q.setdefault("question_number", "")
        q.setdefault("parent_question", "")
        q.setdefault("parent_context", None)
        q.setdefault("section", "A")
        q.setdefault("question", "")
        q.setdefault("type", "open")
        q.setdefault("marks", 1)
        q.setdefault("options", None)
        q.setdefault("column_a", None)
        q.setdefault("column_b", None)
        q.setdefault("memo", None)

        # Ensure marks is integer
        try:
            q["marks"] = int(q["marks"]) if q["marks"] else 1
        except (ValueError, TypeError):
            q["marks"] = 1

        # Detect MCQ options in question text (A. textB. textC. text pattern)
        if q["type"] == "mcq" and not q.get("options"):
            extracted = _extract_options_from_text(q["question"])
            if extracted:
                q["options"] = extracted["options"]
                q["question"] = extracted["question_text"]

        # Validate options for MCQ
        if q["type"] == "mcq":
            opts = q.get("options")
            if not isinstance(opts, dict) or len(opts) < 2:
                q["type"] = "open"
                q["options"] = None

        # Normalise options — ensure dict format
        if isinstance(q.get("options"), list):
            opts = q["options"]
            if opts and isinstance(opts[0], str):
                q["options"] = {chr(65+i): v for i, v in enumerate(opts)}

        # Skip empty questions
        if not q["question"].strip():
            continue

        # Deduplicate
        key = _normalise_qnum(q["question_number"]) or q["question"][:50]
        if key in seen:
            continue
        seen.add(key)

        processed.append(q)

    # Sort by question number
    def sort_key(q):
        qn = q.get("question_number", "")
        # Convert "3.1.2" → (3, 1, 2) for correct numeric sorting
        parts = re.findall(r'\d+', qn)
        return tuple(int(p) for p in parts) if parts else (999,)

    processed.sort(key=sort_key)
    return processed


def _extract_options_from_text(text: str) -> dict | None:
    """
    Extract MCQ options from question text when they're run together.
    e.g. "Which system?A. ServerB. PCC. MobileD. Embedded"
    Returns {"question_text": "Which system?", "options": {"A":"Server",...}}
    """
    # Pattern: letter followed by dot and text, repeated
    pattern = r'([A-D])\.\s*(.+?)(?=[A-D]\.|$)'
    matches = re.findall(pattern, text, re.DOTALL)

    if len(matches) >= 2:
        options = {}
        for letter, option_text in matches:
            options[letter] = option_text.strip()

        # Question text is everything before the first option
        first_option_pos = re.search(r'[A-D]\.', text)
        question_text = text[:first_option_pos.start()].strip() if first_option_pos else text

        return {"question_text": question_text, "options": options}

    return None


def _normalise_qnum(qn: str) -> str:
    """Normalise question number for deduplication and memo matching."""
    s = str(qn).lower().strip()
    s = re.sub(r"^(question|q|ques|no|nr)[\s\.\-]*", "", s)
    s = re.sub(r"^[\(\[\{]|[\)\]\}]$", "", s)
    s = re.sub(r"[^a-z0-9]", "", s)
    return s

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #MEMO PARSER
    #==============================================================
def parse_memo_answers(text: str, subject: str, grade: str) -> dict:
    if not text or not text.strip():
        return {}

    # Preprocess memo text same way as exam
    text = _preprocess_exam_text(text)

    CHUNK = 12000
    OVERLAP = 500
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + CHUNK])
        start += CHUNK - OVERLAP

    memo_map = {}
    for i, chunk in enumerate(chunks):
        print(f"[Memo] Chunk {i+1}/{len(chunks)}")
        chunk_map = _parse_memo_chunk(chunk, subject, grade)
        for qn, answer in chunk_map.items():
            norm = _normalise_qnum(qn)
            if not norm:
                continue
            existing = memo_map.get(norm, "")
            if len(str(answer)) > len(str(existing)):
                memo_map[norm] = answer

    print(f"[Memo] Total answers: {len(memo_map)}")
    return memo_map

def _parse_memo_chunk(text: str, subject: str, grade: str) -> dict:
    """Parse one chunk of memo text into {question_number: answer} dict."""
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    prompt = f"""You are reading a South African NSC CAPS exam MARKING MEMORANDUM.

Extract EVERY answer from the text below.

Return ONLY a valid JSON object mapping question_number to answer string.
Include ALL sub-questions (1.1, 1.2, 2.1.1 etc.)
For MCQ answers just give the letter (e.g. "C").
For True/False give "True" or "False".
For written answers give the full expected answer text.
For matching give the correct pairs.

No markdown, no explanation. Just the JSON object.

Example format:
{{
  "1.1": "C",
  "1.2": "True",
  "1.3": "The CPU processes instructions by fetching, decoding and executing them.",
  "2.1": "A",
  "2.2": "RAM is volatile memory that loses data when power is off."
}}

Subject: {subject} | Grade: {grade}

MEMO TEXT:
{text}
"""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8000,
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            print(f"[Memo] No JSON object in Groq response. Preview: {raw[:300]}")
            return {}
        result = json.loads(match.group())
        return result if isinstance(result, dict) else {}
    except Exception as e:
        print(f"[Memo] Chunk error: {e}")
        return {}

# ═══════════════════════════════════════════════════════════════════════════════
# EXTRACTION PIPELINE (background thread)
# ═══════════════════════════════════════════════════════════════════════════════

def run_extraction_pipeline(exam_id: str, meta: dict, teacher_doc_id: str):
    doc_ref = db.collection("teacherExamUploads").document(teacher_doc_id)

    def set_upload_status(status: str, extra: dict = {}):
        try:
            data = doc_ref.get().to_dict() or {}
            updated = []
            for upload in data.get("uploads", []):
                if upload.get("examId") == exam_id or upload.get("id") == exam_id:
                    upload["status"] = status
                    upload.update(extra)
                updated.append(upload)
            doc_ref.update({"uploads": updated})
        except Exception as e:
            print(f"[Pipeline] Status update failed: {e}")

    try:
        subject  = meta.get("subject", "General")
        grade    = meta.get("grade", "12")
        title    = meta.get("title", meta.get("examFileName", "Exam"))
        exam_fid = meta.get("examDriveFileId")
        memo_fid = meta.get("memoDriveFileId")
        exam_fn  = meta.get("examFileName", "exam.pdf")
        memo_fn  = meta.get("memoFileName", "memo.pdf")

        print(f"\n[Pipeline] ═══ Starting: {exam_id} | {subject} Gr{grade}")
        set_upload_status("processing")

        # 1. Download exam file
        exam_bytes = download_file_bytes(exam_fid, exam_fn)
        if not exam_bytes:
            raise ValueError(f"Could not download exam file (Drive id: {exam_fid})")

        # 2. Extract text
        exam_text = extract_text_from_file(exam_bytes, exam_fn)
        if not exam_text.strip():
            raise ValueError("No text extracted from exam file")
        print(f"[Pipeline] Exam text: {len(exam_text)} chars")

        # 3. Parse questions
        questions = parse_questions_universal(exam_text, subject, grade)

        print(f"[Pipeline] Questions: {len(questions)}")

        # ── 4. Download and parse memo ─────────────────────────────────────
        memo_map = {}
        if memo_fid:
            memo_bytes = download_file_bytes(memo_fid, memo_fn)
            if memo_bytes:
                memo_text = extract_text_from_file(memo_bytes, memo_fn)
                if memo_text.strip():
                    # Use dedicated memo parser — not question parser
                    memo_map = parse_memo_answers(memo_text, subject, grade)
                    print(f"[Pipeline] Memo answers: {len(memo_map)}")

        # ── 5. Merge memo into questions using fuzzy number matching ────────
        # Build normalised lookup for fast matching
        norm_memo = {_normalise_qnum(k): v for k, v in memo_map.items()}

        for q in questions:
            qn_norm = _normalise_qnum(q.get("question_number", ""))
            if qn_norm and qn_norm in norm_memo and not q.get("memo"):
                q["memo"] = norm_memo[qn_norm]

        # 6. Safety net placeholder
        if not questions:
            print("[Pipeline] 0 questions — placeholder")
            questions = [{
                "question_number": "1", "parent_question": "", "section": "A",
                "question": (
                    f"[Questions could not be auto-parsed from this {subject} paper "
                    f"({len(exam_text)} chars extracted). Please ask your teacher to "
                    f"re-upload in Word (.docx) format for best results.]"
                ),
                "type": "open", "marks": 0,
                "options": None, "column_a": None, "column_b": None, "memo": None,
            }]

        # 7. Write exam doc
        db.collection("exams").document(exam_id).set({
            "title": title, "subject": subject, "grade": grade,
            "year": meta.get("year", ""), "curriculum": meta.get("curriculum", "CAPS"),
            "teacherName": meta.get("teacherName", ""),
            "uploadedBy": meta.get("uploadedBy", ""),
            "examDriveFileId": exam_fid, "memoDriveFileId": memo_fid,
            "memoMerged": bool(memo_map), "status": "ready",
            "totalQuestions": len(questions),
            "extractedAt": fs_admin.SERVER_TIMESTAMP,
            "sourceUploadId": exam_id,
        })

        # 8. Write questions in batches of 400
        batch = db.batch()
        for i, q in enumerate(questions):
            ref = db.collection("exam_questions").document(f"{exam_id}_{i:04d}")
            batch.set(ref, {
                "examId": exam_id,
                "questionNumber": q.get("question_number", str(i + 1)),
                "parentQuestion": q.get("parent_question", ""),
                "section": q.get("section", "A"),
                "questionText": q.get("question", ""),
                "type": q.get("type", "open"),
                "marks": int(q.get("marks") or 1),
                "options": q.get("options"),
                "columnA": q.get("column_a"),
                "columnB": q.get("column_b"),
                "memo": q.get("memo") or "",
                "order": i,
            })
            if (i + 1) % 400 == 0:
                batch.commit()
                batch = db.batch()
        batch.commit()

        # 9. Mark upload extracted
        set_upload_status("extracted", {
            "extractedAt": datetime.utcnow().isoformat(),
            "totalQuestions": len(questions),
            "memoMerged": bool(memo_map),
        })
        print(f"[Pipeline] Done: {len(questions)} questions, {len(memo_map)} memo answers\n")

    except Exception as e:
        traceback.print_exc()
        print(f"[Pipeline] Failed: {e}\n")
        set_upload_status("error", {"errorMessage": str(e)[:500]})
        try:
            db.collection("exams").document(exam_id).set(
                {"status": "error", "errorMessage": str(e)[:500]}, merge=True
            )
        except Exception:
            pass

@app.route("/admin/debug-memo/<exam_id>", methods=["GET"])
def debug_memo(exam_id):
    """Show exactly what memo text was extracted and what answers were parsed."""
    try:
        meta = None
        for doc in db.collection("teacherExamUploads").stream():
            for upload in doc.to_dict().get("uploads", []):
                if upload.get("examId") == exam_id or upload.get("id") == exam_id:
                    meta = upload
                    break
            if meta:
                break

        if not meta:
            return jsonify({"error": "not found"}), 404

        memo_fid = meta.get("memoDriveFileId")
        memo_fn  = meta.get("memoFileName", "memo.pdf")

        if not memo_fid:
            return jsonify({"error": "No memo file on this upload"})

        # Download memo
        memo_bytes = download_file_bytes(memo_fid)
        if not memo_bytes:
            return jsonify({"error": "Could not download memo from Drive"})

        # Extract text
        memo_text = extract_text_from_file(memo_bytes, memo_fn)

        # Parse answers
        memo_map = parse_memo_answers(memo_text, meta.get("subject",""), meta.get("grade","12"))

        # Also get exam question numbers for comparison
        exam_q_nums = []
        for q in db.collection("exam_questions").where("examId","==",exam_id).stream():
            exam_q_nums.append(q.to_dict().get("questionNumber",""))

        # Check which exam questions have a matching memo answer
        matched = []
        unmatched = []
        norm_memo = {_normalise_qnum(k): v for k, v in memo_map.items()}
        for qn in exam_q_nums:
            if _normalise_qnum(qn) in norm_memo:
                matched.append(qn)
            else:
                unmatched.append(qn)

        return jsonify({
            "memo_file":        memo_fn,
            "memo_text_length": len(memo_text),
            "memo_text_preview": memo_text[:2000],
            "memo_answers_count": len(memo_map),
            "memo_answers_sample": dict(list(memo_map.items())[:10]),
            "exam_questions_count": len(exam_q_nums),
            "matched_count":    len(matched),
            "unmatched_count":  len(unmatched),
            "unmatched_sample": unmatched[:20],
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# debugging
@app.route("/admin/reset-exam/<exam_id>", methods=["GET"])
def reset_exam(exam_id):
    """Wipe exam + questions from Firestore so extraction runs fresh."""
    try:
        # Delete all questions
        batch = db.batch()
        count = 0
        for doc in db.collection("exam_questions").where("examId", "==", exam_id).stream():
            batch.delete(doc.reference)
            count += 1
            if count % 400 == 0:
                batch.commit()
                batch = db.batch()
        batch.commit()

        # Delete exam doc
        db.collection("exams").document(exam_id).delete()

        # Reset upload status to pending so trigger-extract will run
        for doc in db.collection("teacherExamUploads").stream():
            uploads = doc.to_dict().get("uploads", [])
            updated = False
            for u in uploads:
                if u.get("examId") == exam_id or u.get("id") == exam_id:
                    u["status"] = "pending"
                    updated = True
            if updated:
                doc.reference.update({"uploads": uploads})
                break

        return jsonify({"ok": True, "deleted_questions": count, "exam_id": exam_id})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ═══════════════════════════════════════════════════════════════════════════════
# EXAM LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_exam_from_firestore(exam_id: str):
    try:
        doc = db.collection("exams").document(exam_id).get()
        if not doc.exists:
            return None, []
        meta = doc.to_dict()
        meta["id"] = doc.id

        raw_qs = list(db.collection("exam_questions").where("examId", "==", exam_id).stream())
        raw_qs.sort(key=lambda d: d.to_dict().get("order", 0))

        questions = []
        for q in raw_qs:
            d = q.to_dict()
            options = d.get("options")
            if isinstance(options, dict) and options:
                options = [{"key": k, "value": v} for k, v in sorted(options.items())]
            elif isinstance(options, list) and options and isinstance(options[0], str):
                options = [{"key": chr(65 + i), "value": v} for i, v in enumerate(options)]
            questions.append({
                "question_number":      str(d.get("questionNumber", "")),
                "parent_question":      d.get("parentQuestion", ""),
                "parent_context":       d.get("parentContext"),
                "section":              d.get("section", "A"),
                "section_title":        d.get("sectionTitle", ""),
                "section_instructions": d.get("sectionInstructions", ""),
                "section_total_marks":  d.get("sectionTotalMarks"),
                "question":             d.get("questionText", "Question text missing"),
                "type":                 d.get("type", "open").lower(),
                "options":              options,
                "column_a":             d.get("columnA"),
                "column_b":             d.get("columnB"),
                "marks":                d.get("marks", 1),
                "memo":                 d.get("memo", ""),
                "saved_answer":         "",
            })
        return meta, questions
    except Exception as e:
        traceback.print_exc()
        return None, []


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS — find upload meta by exam_id
# ═══════════════════════════════════════════════════════════════════════════════

def _find_upload_meta(exam_id: str):
    """
    Search all teacherExamUploads docs for an upload matching exam_id.
    Returns (meta_dict, teacher_doc_id) or (None, None).
    """
    for doc in db.collection("teacherExamUploads").stream():
        for upload in doc.to_dict().get("uploads", []):
            if upload.get("examId") == exam_id or upload.get("id") == exam_id:
                return upload, doc.id
    return None, None


def _is_already_processing(exam_id: str) -> bool:
    """
    Returns True if the exam is already being processed or is done.
    Prevents duplicate pipeline threads.
    """
    try:
        exam_doc = db.collection("exams").document(exam_id).get()
        if exam_doc.exists:
            return exam_doc.to_dict().get("status") in ("ready", "extracted", "processing")
    except Exception:
        pass
    return False


def _launch_pipeline(exam_id: str, meta: dict, teacher_doc_id: str):
    """
    Mark exam as processing and launch extraction in a background thread.
    Safe to call multiple times — checks before launching.
    """
    if _is_already_processing(exam_id):
        print(f"[Pipeline] Skipping {exam_id} — already processing/ready")
        return False

    # Mark as processing immediately to prevent duplicate triggers
    try:
        db.collection("exams").document(exam_id).set(
            {"status": "processing", "startedAt": fs_admin.SERVER_TIMESTAMP},
            merge=True,
        )
    except Exception as e:
        print(f"[Pipeline] Could not mark processing: {e}")

    threading.Thread(
        target=run_extraction_pipeline,
        args=(exam_id, meta, teacher_doc_id),
        daemon=True,
    ).start()
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# FIRESTORE LISTENER — auto-triggers extraction when new exam saved
# ═══════════════════════════════════════════════════════════════════════════════

def _start_auto_extraction_listener():
    """
    Watches teacherExamUploads collection.
    Any upload with status=pending_extraction automatically triggers the pipeline.
    Frontend never needs to call /auto-extract.
    """
    def on_snapshot(col_snapshot, changes, read_time):
        for change in changes:
            if change.type.name not in ("ADDED", "MODIFIED"):
                continue

            doc_data       = change.document.to_dict() or {}
            teacher_doc_id = change.document.id

            for upload in doc_data.get("uploads", []):
                exam_id = upload.get("examId") or upload.get("id")
                status  = upload.get("status", "")

                if not exam_id:
                    continue
                if status != "pending_extraction":
                    continue
                if not upload.get("examDriveFileId"):
                    continue
                if _is_already_processing(exam_id):
                    continue

                print(f"[AutoListener] Detected pending exam: {exam_id} — launching pipeline")
                _launch_pipeline(exam_id, upload, teacher_doc_id)

    db.collection("teacherExamUploads").on_snapshot(on_snapshot)
    print("[AutoListener] Firestore extraction listener active")


# ═══════════════════════════════════════════════════════════════════════════════
# STARTUP SWEEP — catches exams missed while server was asleep (Render cold start)
# ═══════════════════════════════════════════════════════════════════════════════

def _sweep_pending_on_startup():
    """
    On boot, find any uploads stuck in pending_extraction that have no
    corresponding ready/processing exam doc, and process them.
    Handles the case where Render was asleep when the teacher uploaded.
    """
    print("[Startup] Sweeping for missed pending extractions...")
    count = 0
    try:
        for doc in db.collection("teacherExamUploads").stream():
            teacher_doc_id = doc.id
            for upload in doc.to_dict().get("uploads", []):
                exam_id = upload.get("examId") or upload.get("id")
                status  = upload.get("status", "")

                if not exam_id:
                    continue
                if status != "pending_extraction":
                    continue
                if not upload.get("examDriveFileId"):
                    continue
                if _is_already_processing(exam_id):
                    continue

                print(f"[Startup] Found missed extraction: {exam_id}")
                launched = _launch_pipeline(exam_id, upload, teacher_doc_id)
                if launched:
                    count += 1

    except Exception as e:
        print(f"[Startup] Sweep failed: {e}")

    print(f"[Startup] Sweep complete — launched {count} missed extraction(s)")


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/auto-extract", methods=["POST"])
def auto_extract():
    """
    Called by frontend after upload. Now mostly a fallback —
    the Firestore listener handles extraction automatically.
    Still useful for manual retries or direct API calls.
    """
    try:
        data    = request.get_json()
        exam_id = data.get("exam_id", "").strip()
        if not exam_id:
            return jsonify({"error": "exam_id required"}), 400

        meta, teacher_doc_id = _find_upload_meta(exam_id)

        if not meta:
            return jsonify({"error": f"Upload {exam_id} not found"}), 404
        if not meta.get("examDriveFileId"):
            return jsonify({"error": "No examDriveFileId on this upload"}), 400

        # Already done — return early
        if _is_already_processing(exam_id):
            exam_doc = db.collection("exams").document(exam_id).get()
            if exam_doc.exists and exam_doc.to_dict().get("status") == "ready":
                return jsonify({
                    "ok":      True,
                    "message": "Already extracted — skipping",
                    "exam_id": exam_id,
                })
            return jsonify({
                "ok":      True,
                "message": "Extraction already in progress",
                "exam_id": exam_id,
            })

        launched = _launch_pipeline(exam_id, meta, teacher_doc_id)

        return jsonify({
            "ok":          True,
            "message":     "Extraction started" if launched else "Already running",
            "exam_id":     exam_id,
            "poll_status": f"/admin/extraction-status/{exam_id}",
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/admin/trigger-extract/<exam_id>", methods=["GET"])
def trigger_extract(exam_id):
    """
    Manual admin trigger — forces re-extraction even if already processed.
    Useful for fixing failed or incomplete extractions.
    """
    try:
        meta, teacher_doc_id = _find_upload_meta(exam_id)

        if not meta:
            return jsonify({"error": f"Exam {exam_id} not found in any upload doc"}), 404
        if not meta.get("examDriveFileId"):
            return jsonify({"error": "No examDriveFileId on this upload"}), 400

        # Force re-extraction: reset status so _is_already_processing returns False
        try:
            db.collection("exams").document(exam_id).set(
                {"status": "pending_extraction"},
                merge=True,
            )
        except Exception:
            pass

        # Launch directly without the already-processing guard
        threading.Thread(
            target=run_extraction_pipeline,
            args=(exam_id, meta, teacher_doc_id),
            daemon=True,
        ).start()

        return jsonify({
            "ok":          True,
            "message":     "Extraction started in background",
            "exam_id":     exam_id,
            "poll_status": f"/admin/extraction-status/{exam_id}",
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/admin/uploads", methods=["GET"])
def admin_uploads():
    try:
        uploads = []
        for doc in db.collection("teacherExamUploads").stream():
            for upload in doc.to_dict().get("uploads", []):
                upload["teacherDocId"] = doc.id
                uploads.append(upload)
        uploads.sort(key=lambda x: x.get("uploadedAt", ""), reverse=True)
        return jsonify({"uploads": uploads})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/admin/extraction-status/<exam_id>", methods=["GET"])
def extraction_status(exam_id):
    try:
        exam_doc = db.collection("exams").document(exam_id).get()
        if exam_doc.exists:
            d = exam_doc.to_dict()
            q_count = sum(1 for _ in db.collection("exam_questions").where("examId", "==", exam_id).stream())
            return jsonify({
                "status":               d.get("status"),
                "title":                d.get("title"),
                "subject":              d.get("subject"),
                "questions_in_firestore": q_count,
                "memo_merged":          d.get("memoMerged", False),
                "student_accessible":   d.get("status") == "ready" and q_count > 0,
            })
        for doc in db.collection("teacherExamUploads").stream():
            for upload in doc.to_dict().get("uploads", []):
                if str(upload.get("examId") or upload.get("id") or "").strip() == exam_id.strip():
                    return jsonify({
                        "status":             upload.get("status", "pending_extraction"),
                        "error":              upload.get("errorMessage"),
                        "student_accessible": False,
                    })
        return jsonify({"status": "not_found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def save_session_to_fs(sid: str, data: dict):
    db.collection("exam_sessions").document(sid).set({
        **data,
        "createdAt": fs_admin.SERVER_TIMESTAMP,
    })

def get_session_from_fs(sid: str) -> dict | None:
    if not sid:
        return None
    doc = db.collection("exam_sessions").document(sid).get()
    return doc.to_dict() if doc.exists else None

def update_session_answers(sid: str, answers: dict):
    db.collection("exam_sessions").document(sid).update({"answers": answers})

def delete_session_from_fs(sid: str):
    db.collection("exam_sessions").document(sid).delete()


@app.route("/exams", methods=["GET"])
def list_exams():
    exams = []
    try:
        for doc in db.collection("exams").where("status", "==", "ready").stream():
            d = doc.to_dict()
            exams.append({
                "id":         doc.id,
                "name":       d.get("title", doc.id),
                "subject":    d.get("subject", ""),
                "grade":      d.get("grade", ""),
                "year":       d.get("year", ""),
                "curriculum": d.get("curriculum", "CAPS"),
                "memoMerged": d.get("memoMerged", False),
            })
    except Exception as e:
        print(f"[list_exams] {e}")
    return jsonify({"exams": exams})

# debugging
@app.route("/admin/debug-exam-text/<exam_id>", methods=["GET"])
def debug_exam_text(exam_id):
    """Show raw extracted text from exam file."""
    try:
        meta = None
        for doc in db.collection("teacherExamUploads").stream():
            for upload in doc.to_dict().get("uploads", []):
                if upload.get("examId") == exam_id or upload.get("id") == exam_id:
                    meta = upload
                    break
            if meta:
                break

        if not meta:
            return jsonify({"error": "not found"}), 404

        exam_bytes = download_file_bytes(meta.get("examDriveFileId"), meta.get("examFileName",""))
        if not exam_bytes:
            return jsonify({"error": "download failed"})

        exam_text = extract_text_from_file(exam_bytes, meta.get("examFileName","exam.pdf"))

        return jsonify({
            "filename": meta.get("examFileName"),
            "text_length": len(exam_text),
            "full_text": exam_text,  # show everything
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/start_exam", methods=["POST"])
def start_exam():
    try:
        data = request.get_json()

        # SUPPORT BOTH exam_id and exam
        exam_id = (
            data.get("exam_id")
            or data.get("exam")
            or ""
        ).strip()

        student_id = data.get("student_id", "anonymous")

        if not exam_id:
            return jsonify({
                "error": "No exam specified"
            }), 400

        meta, questions = load_exam_from_firestore(exam_id)

        if meta is None:
            return jsonify({
                "error": f"Exam '{exam_id}' not found"
            }), 404

        if not questions:
            return jsonify({
                "error": (
                    "This exam has no questions yet. "
                    "Extraction may still be processing."
                ),
                "status": meta.get("status", "unknown")
            }), 400

        mem.ensure_student(student_id)

        session_id = str(uuid.uuid4())

        save_session_to_fs(session_id, {
            "exam_id": exam_id,
            "exam": meta.get("title", exam_id),
            "title": meta.get("title", ""),
            "subject": meta.get("subject", ""),
            "student_id": student_id,
            "questions": questions,
            "answers": {},
            "started_at": datetime.utcnow().isoformat(),
        })

        return jsonify({
            "success": True,
            "session_id": session_id,
            "exam_id": exam_id,
            "title": meta.get("title", ""),
            "subject": meta.get("subject", ""),
            "total_questions": len(questions),
            "memo_merged": meta.get("memoMerged", False),
            "status": meta.get("status", "ready"),
        })

    except Exception as e:
        traceback.print_exc()

        return jsonify({
            "error": str(e)
        }), 500


@app.route("/answer", methods=["POST"])
def save_answer():
    try:
        data    = request.get_json()
        sid     = data.get("session_id")
        session = get_session_from_fs(sid)
        if not session:
            return jsonify({"error": "Invalid session"})

        answers = session.get("answers", {})
        answers[str(data.get("index"))] = data.get("answer", "")
        update_session_answers(sid, answers)
        return jsonify({"status": "saved"})
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/question", methods=["POST"])
def get_question():
    try:
        data    = request.get_json()
        session = get_session_from_fs(data.get("session_id"))
        if not session:
            return jsonify({"error": "Invalid session"})
        idx = int(data.get("index", 0))
        qs  = session["questions"]
        if idx < 0 or idx >= len(qs):
            return jsonify({"error": "Index out of range"})
        q = qs[idx].copy()
        q["saved_answer"] = session.get("answers", {}).get(str(idx), "")
        return jsonify(q)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)})


@app.route("/submit", methods=["POST"])
def submit_exam():
    try:
        data       = request.get_json()
        sid        = data.get("session_id")
        student_id = data.get("student_id", "anonymous")
        session    = get_session_from_fs(sid)

        if not session:
            return jsonify({"error": "Invalid session"})

        subject   = session.get("subject", "")
        questions = session["questions"]
        answers   = session.get("answers", {})
        results   = []
        total_score = 0
        total_marks = 0

        for i, q in enumerate(questions):
            q_num       = q.get("question_number", f"Q{i+1}")
            q_type      = q.get("type", "open").lower()
            marks       = int(q.get("marks") or 1)
            memo        = q.get("memo", "")
            student_ans = answers.get(str(i), "").strip()

            options = q.get("options")
            if isinstance(options, list) and options and isinstance(options[0], dict):
                options = {o["key"]: o["value"] for o in options}

            result = mark_answer(
                question=q.get("question", ""),
                question_number=q_num,
                q_type=q_type,
                student_answer=student_ans,
                memo=memo,
                marks=marks,
                options=options,
                instructions=q.get("instructions", ""),
                subject=subject,
            )

            if result.get("status") in ("incorrect", "missing"):
                mem.record_wrong(student_id, q_num, q.get("question", ""), q_type, subject)
            elif result.get("status") == "correct":
                mem.record_correct(student_id, q_num)

            correct_display = "Not available"
            if memo:
                if q_type == "mcq" and isinstance(options, dict):
                    cl = str(memo).strip().upper()
                    correct_display = f"{cl}. {options.get(cl, '')}" if cl in options else cl
                elif q_type == "matching" and isinstance(memo, dict):
                    correct_display = " | ".join(f"{k} → {v}" for k, v in memo.items())
                else:
                    correct_display = str(memo)

            results.append({
                **result,
                "question_number": q_num,
                "question":        q.get("question", ""),
                "type":            q_type,
                "marks":           marks,
                "earned":          result.get("score", 0),
                "student_answer":  student_ans or "No answer",
                "correct_answer":  correct_display,
            })
            total_score += result.get("score", 0)
            total_marks += marks

        percentage = round(total_score / total_marks * 100, 1) if total_marks else 0

        mem.save_session(
            student_id, session["exam"],
            total_score, total_marks, percentage,
            subject=subject,
        )

        feedback = generate_exam_feedback(
            results, total_score, total_marks, percentage,
            subject=subject,
        )

        # Update study plan in background if student has weak topics
        weak = mem.get_weak_topics(student_id)
        if weak:
            try:
                run_agent(
                    student_id,
                    f"I scored {percentage}% on {session['exam']} ({subject}). Update my study plan.",
                    rag=rag,
                )
            except Exception as agent_err:
                print(f"[submit] Agent update failed (non-fatal): {agent_err}")

        # Clean up Firestore session — no longer needed after submit
        delete_session_from_fs(sid)

        return jsonify({
            "score":      total_score,
            "total":      total_marks,
            "percentage": percentage,
            "results":    results,
            "feedback":   feedback,
            "subject":    subject,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)})

@app.route("/agent-chat", methods=["POST"])
def agent_chat():
    try:
        data       = request.get_json()
        student_id = data.get("student_id", "anonymous")
        message    = data.get("message", "").strip()
        if not message:
            return jsonify({"response": "Please enter a message."})
        response = run_agent(student_id, message, rag=rag)
        return jsonify({"response": response})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"response": f"Agent error: {e}"})


@app.route("/clear-history", methods=["POST"])
def clear_history():
    data = request.get_json()
    mem.clear_history(data.get("student_id", ""))
    return jsonify({"status": "cleared"})

@app.route("/admin/cleanup-sessions", methods=["POST"])
def cleanup_sessions():
    """Delete exam sessions older than 24 hours."""
    from datetime import timedelta, timezone
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    deleted = 0
    for doc in db.collection("exam_sessions").stream():
        created = doc.to_dict().get("createdAt")
        if created and created < cutoff:
            doc.reference.delete()
            deleted += 1
    return jsonify({"deleted": deleted})

@app.route("/dashboard", methods=["POST"])
def dashboard():
    try:
        data       = request.get_json()
        student_id = data.get("student_id", "anonymous")
        mem.ensure_student(student_id)
        return jsonify({
            "weak":       mem.get_weak_topics(student_id),
            "sessions":   mem.get_sessions(student_id, limit=8),
            "study_plan": mem.get_study_plan(student_id),
        })
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok", "service": "EduCAT API",
        "version": "2.0 — auto-extraction, docx + pdf",
        "endpoints": {
            "exams":         "GET  /exams",
            "start":         "POST /start-exam",
            "question":      "POST /question",
            "answer":        "POST /answer",
            "submit":        "POST /submit",
            "agent":         "POST /agent-chat",
            "auto_extract":  "POST /auto-extract  {exam_id}",
            "admin_uploads": "GET  /admin/uploads",
            "admin_trigger": "GET  /admin/trigger-extract/<exam_id>",
            "admin_status":  "GET  /admin/extraction-status/<exam_id>",
        }
    })

def watch_uploads():
    print("[Watcher] Starting Firestore upload watcher...")

    seen = set()

    while True:
        try:
            docs = db.collection("teacherExamUploads").stream()

            for doc in docs:
                data = doc.to_dict()
                uploads = data.get("uploads", [])

                for upload in uploads:
                    exam_id = upload.get("examId") or upload.get("id")
                    status = upload.get("status")

                    if not exam_id:
                        continue

                    key = f"{doc.id}_{exam_id}"

                    if status in ("pending", None) and key not in seen:
                        print(f"[Watcher] Auto-extracting: {exam_id}")

                        seen.add(key)

                        threading.Thread(
                            target=run_extraction_pipeline,
                            args=(exam_id, upload, doc.id),
                            daemon=True
                        ).start()

        except Exception as e:
            print("[Watcher Error]", e)

        time.sleep(10)


# Start listener for all environments (including Render/gunicorn)
_start_auto_extraction_listener()
_sweep_pending_on_startup()

if __name__ == "__main__":
    app.run(debug=True, port=8000)