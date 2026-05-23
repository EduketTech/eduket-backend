"""
app.py — EduCAT Flask API (Auto-Extraction Prototype)

WHAT'S NEW:
  - Auto-extraction: when teacher uploads, extraction triggers automatically
  - Firebase Storage Integration: Files downloaded directly from Cloud Storage bucket.
  - Word doc support: .docx files extracted via python-docx (no OCR needed, perfect text)
  - PDF support: native text first, Groq vision OCR fallback for scanned PDFs
  - No manual trigger needed — teacher uploads then POST /auto-extract is called

FLOW:
  1. Teacher uploads via frontend → doc saved to Cloud Storage and metadata to teacherExamUploads in Firestore
  2. Frontend calls POST /auto-extract immediately after upload
  3. Backend downloads file from Firebase Storage, extracts text, parses questions with Groq
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

# ── Firebase & Cloud Storage ──────────────────────────────────────────────────
import firebase_admin
from firebase_admin import credentials, storage, firestore as fs_admin


def _init_firebase():
    if firebase_admin._apps:
        return
    raw = os.getenv("FIREBASE_SERVICE_ACCOUNT", "")
    bucket_name = os.getenv("FIREBASE_STORAGE_BUCKET")

    if raw.strip():
        cred = credentials.Certificate(json.loads(raw))
    else:
        cred = credentials.ApplicationDefault()

    firebase_admin.initialize_app(cred, {
        'storageBucket': bucket_name
    })


_init_firebase()
db = fs_admin.client()
bucket = storage.bucket()

# ── App modules ───────────────────────────────────────────────────────────────
from model import generate_answer, mark_answer, generate_exam_feedback
from rag import RAGIndex
import memory as mem
import agent
from agent import run_agent

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:5175",
    "http://localhost:5176",
    "http://localhost:5177",
    "https://eduket.netlify.app",
    "https://*.netlify.app",
]}})

rag = RAGIndex()
agent.set_rag(rag)


# ═══════════════════════════════════════════════════════════════════════════════
# FIREBASE STORAGE UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def download_file_bytes_from_storage(storage_path: str) -> bytes | None:
    """Downloads a file directly from the application's Firebase Storage bucket."""
    try:
        print(f"[Storage] Downloading blob: {storage_path}")
        blob = bucket.blob(storage_path)
        file_bytes = blob.download_as_bytes()
        return file_bytes
    except Exception as e:
        print(f"[Storage] Error downloading {storage_path}: {e}")
        return None


def download_file_for_extraction(meta: dict, file_type: str) -> tuple[bytes | None, str]:
    """
    Downloads exam or memo file from wherever it was stored.
    Supports both Google Drive (old) and Firebase Storage (new).
    Returns (bytes, filename) or (None, filename).

    file_type: "exam" or "memo"
    """
    # ── Firebase Storage path (new) ────────────────────────────────────
    storage_url = meta.get(f"{file_type}StorageUrl")
    filename = meta.get(f"{file_type}FileName", f"{file_type}.pdf")

    if storage_url:
        print(f"[Download] Fetching {file_type} from Firebase Storage")
        try:
            res = http_requests.get(storage_url, timeout=60)
            if res.status_code == 200:
                print(f"[Download] Got {len(res.content)} bytes from Storage")
                return res.content, filename
            print(f"[Download] Storage fetch failed: {res.status_code}")
        except Exception as e:
            print(f"[Download] Storage error: {e}")
        return None, filename

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
                all_text += f"\n--- PAGE {i + 1} ---\n{page_text}\n"
                print(f"[OCR] Page {i + 1}: {len(page_text)} chars")
            except Exception as e:
                print(f"[OCR] Page {i + 1} failed: {e}")
        doc.close()
    except Exception as e:
        print(f"[OCR] fitz error: {e}")
    return all_text


def extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    name = filename.lower()
    if name.endswith(".docx") or name.endswith(".doc"):
        print(f"[Extract] Word doc: {filename}")
        return extract_text_from_docx(file_bytes)
    elif name.endswith(".pdf"):
        print(f"[Extract] PDF: {filename}")
        return extract_text_from_pdf(file_bytes)
    else:
        print(f"[Extract] Unknown type {filename} — trying docx then pdf")
        text = extract_text_from_docx(file_bytes)
        return text if text.strip() else extract_text_from_pdf(file_bytes)


# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL QUESTION PARSER — handles ANY exam structure
# ═══════════════════════════════════════════════════════════════════════════════

def parse_questions_universal(text: str, subject: str, grade: str) -> list:
    if not text or not text.strip():
        return []

    # ── Pass 1: Pre-process text ───────────────────────────────────────────
    text = _preprocess_exam_text(text)

    # ── Pass 2: Chunk and parse ────────────────────────────────────────────
    CHUNK = 10000
    OVERLAP = 800
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + CHUNK])
        start += CHUNK - OVERLAP

    all_questions = []
    seen_numbers = set()

    for i, chunk in enumerate(chunks):
        print(f"[Universal] Chunk {i + 1}/{len(chunks)} ({len(chunk)} chars)")
        qs = _parse_any_structure(chunk, subject, grade)
        for q in qs:
            qn = _normalise_qnum(q.get("question_number", ""))
            if qn and qn not in seen_numbers:
                seen_numbers.add(qn)
                all_questions.append(q)
            elif not qn:
                all_questions.append(q)

    # ── Pass 3: Post-process ───────────────────────────────────────────────
    all_questions = _postprocess_questions(all_questions, text)

    print(f"[Universal] Final: {len(all_questions)} questions")
    return all_questions


def _preprocess_exam_text(text: str) -> str:
    text = re.sub(r'([a-z\?\.\,\)])\s*([A-D])\.\s', r'\1\n\2. ', text)
    text = re.sub(r'(\w[\.\,])\s+(\d+\.\d+)', r'\1\n\2', text)
    text = re.sub(r'([^\n])(QUESTION\s+\d+)', r'\1\n\2', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\((\d+)\s+marks?\)', r'(\1)', text, flags=re.IGNORECASE)
    return text.strip()


def _parse_any_structure(text: str, subject: str, grade: str) -> list:
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

NUMBERED QUESTIONS without numbers:
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
  "question_number": "1.1",
  "parent_question": "QUESTION 1",
  "parent_context": null,
  "section": "A",
  "question": "Full question text here (no options for MCQ)",
  "type": "mcq",
  "marks": 2,
  "options": {{"A":"...","B":"...","C":"...","D":"..."}},
  "column_a": null,
  "column_b": null,
  "memo": null
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

        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\s*\s *$", "", raw, flags=re.MULTILINE)
        raw = raw.strip()

        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            raw = match.group(0)

        result = json.loads(raw)
        return result if isinstance(result, list) else []

    except json.JSONDecodeError as e:
        print(f"[Universal] JSON error: {e}")
        return _salvage_partial_json(raw)
    except Exception as e:
        print(f"[Universal] Error: {e}")
        return []


def _salvage_partial_json(raw: str) -> list:
    questions = []
    for match in re.finditer(r'\{[^{}]*\}', raw, re.DOTALL):
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict) and obj.get("question"):
                questions.append(obj)
        except Exception:
            continue
    return questions


def _postprocess_questions(questions: list, original_text: str) -> list:
    processed = []
    seen = set()

    for q in questions:
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

        try:
            q["marks"] = int(q["marks"]) if q["marks"] else 1
        except (ValueError, TypeError):
            q["marks"] = 1

        if q["type"] == "mcq" and not q.get("options"):
            extracted = _extract_options_from_text(q["question"])
            if extracted:
                q["options"] = extracted["options"]
                q["question"] = extracted["question_text"]

        if q["type"] == "mcq":
            opts = q.get("options")
            if not isinstance(opts, dict) or len(opts) < 2:
                q["type"] = "open"
                q["options"] = None

        if isinstance(q.get("options"), list):
            opts = q["options"]
            if opts and isinstance(opts[0], str):
                q["options"] = {chr(65 + i): v for i, v in enumerate(opts)}

        if not q["question"].strip():
            continue

        key = _normalise_qnum(q["question_number"]) or q["question"][:50]
        if key in seen:
            continue
        seen.add(key)

        processed.append(q)

    def sort_key(q):
        qn = q.get("question_number", "")
        parts = re.findall(r'\d+', qn)
        return tuple(int(p) for p in parts) if parts else (999,)

    processed.sort(key=sort_key)
    return processed


def _extract_options_from_text(text: str) -> dict | None:
    pattern = r'([A-D])\.\s*(.+?)(?=[A-D]\.|$)'
    matches = re.findall(pattern, text, re.DOTALL)

    if len(matches) >= 2:
        options = {}
        for letter, option_text in matches:
            options[letter] = option_text.strip()

        first_option_pos = re.search(r'[A-D]\.', text)
        question_text = text[:first_option_pos.start()].strip() if first_option_pos else text

        return {"question_text": question_text, "options": options}

    return None


def _normalise_qnum(qn: str) -> str:
    s = str(qn).lower().strip()
    s = re.sub(r"^(question|q|ques|no|nr)[\s\.\-]*", "", s)
    s = re.sub(r"^[\(\[\{]|[\)\]\}]$", "", s)
    s = re.sub(r"[^a-z0-9]", "", s)
    return s


# ═══════════════════════════════════════════════════════════════════════════════
# MEMO PARSER
# ═══════════════════════════════════════════════════════════════════════════════

def parse_memo_answers(text: str, subject: str, grade: str) -> dict:
    if not text or not text.strip():
        return {}

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
        print(f"[Memo] Chunk {i + 1}/{len(chunks)}")
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
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    prompt = f"""You are reading a South African NSC CAPS exam MARKING MEMORANDUM.

Extract EVERY answer from the text below.
Return ONLY a valid JSON object mapping question_number to answer string.

No markdown, no explanation. Just the JSON object.

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
        subject = meta.get("subject", "General")
        grade   = meta.get("grade", "12")
        title   = meta.get("title", meta.get("examFileName", "Exam"))

        print(f"\n[Pipeline] ═══ Starting: {exam_id} | {subject} Gr{grade}")
        set_upload_status("processing")

        # ── 1. Download exam file ──────────────────────────────────────
        exam_bytes, exam_fn = download_file_for_extraction(meta, "exam")
        if not exam_bytes:
            raise ValueError(
                "Could not download exam file. "
                "Check examStorageUrl or examDriveFileId in the upload record."
            )

        # ── 2. Extract text ────────────────────────────────────────────
        exam_text = extract_text_from_file(exam_bytes, exam_fn)
        if not exam_text.strip():
            raise ValueError("No text extracted from exam file")
        print(f"[Pipeline] Exam text: {len(exam_text)} chars")

        # ── 3. Parse questions ─────────────────────────────────────────
        questions = parse_questions_universal(exam_text, subject, grade)
        print(f"[Pipeline] Questions: {len(questions)}")

        # ── 4. Download and parse memo ─────────────────────────────────
        memo_map = {}
        memo_bytes, memo_fn = download_file_for_extraction(meta, "memo")
        if memo_bytes:
            memo_text = extract_text_from_file(memo_bytes, memo_fn)
            if memo_text.strip():
                memo_map = parse_memo_answers(memo_text, subject, grade)
                print(f"[Pipeline] Memo answers: {len(memo_map)}")

        # ── 5. Merge memo into questions ───────────────────────────────
        norm_memo = {_normalise_qnum(k): v for k, v in memo_map.items()}
        for q in questions:
            qn_norm = _normalise_qnum(q.get("question_number", ""))
            if qn_norm and qn_norm in norm_memo and not q.get("memo"):
                q["memo"] = norm_memo[qn_norm]

        # ── 6. Safety net placeholder ──────────────────────────────────
        if not questions:
            print("[Pipeline] 0 questions — placeholder")
            questions = [{
                "question_number": "1", "parent_question": "", "section": "A",
                "question": (
                    f"[Questions could not be auto-parsed from this {subject} paper "
                    f"({len(exam_text)} chars extracted). Please re-upload in "
                    f"Word (.docx) format for best results.]"
                ),
                "type": "open", "marks": 0,
                "options": None, "column_a": None, "column_b": None, "memo": None,
            }]

        # ── 7. Write exam doc ──────────────────────────────────────────
        db.collection("exams").document(exam_id).set({
            "title":          title,
            "subject":        subject,
            "grade":          grade,
            "year":           meta.get("year", ""),
            "curriculum":     meta.get("curriculum", "CAPS"),
            "teacherName":    meta.get("teacherName", ""),
            "uploadedBy":     meta.get("uploadedBy", ""),
            "schoolId":       meta.get("schoolId", ""),
            "examDuration":   meta.get("examDuration", 0),
            "memoMerged":     bool(memo_map),
            "status":         "ready",
            "totalQuestions": len(questions),
            "extractedAt":    fs_admin.SERVER_TIMESTAMP,
            "sourceUploadId": exam_id,
            # Keep storage URLs so they're accessible from the exam doc
            "examStorageUrl": meta.get("examStorageUrl", ""),
            "memoStorageUrl": meta.get("memoStorageUrl", ""),
            # Keep Drive IDs for backwards compatibility
            "examDriveFileId": meta.get("examDriveFileId", ""),
            "memoDriveFileId": meta.get("memoDriveFileId", ""),
        })

        # ── 8. Write questions in batches ──────────────────────────────
        batch = db.batch()
        for i, q in enumerate(questions):
            ref = db.collection("exam_questions").document(f"{exam_id}_{i:04d}")
            batch.set(ref, {
                "examId":         exam_id,
                "questionNumber": q.get("question_number", str(i + 1)),
                "parentQuestion": q.get("parent_question", ""),
                "section":        q.get("section", "A"),
                "questionText":   q.get("question", ""),
                "type":           q.get("type", "open"),
                "marks":          int(q.get("marks") or 1),
                "options":        q.get("options"),
                "columnA":        q.get("column_a"),
                "columnB":        q.get("column_b"),
                "memo":           q.get("memo") or "",
                "order":          i,
            })
            if (i + 1) % 400 == 0:
                batch.commit()
                batch = db.batch()
        batch.commit()

        # ── 9. Mark complete ───────────────────────────────────────────
        set_upload_status("extracted", {
            "extractedAt":    datetime.utcnow().isoformat(),
            "totalQuestions": len(questions),
            "memoMerged":     bool(memo_map),
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


# ═══════════════════════════════════════════════════════════════════════════════
# CONTROLLERS & ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/auto-extract", methods=["POST"])
def auto_extract():
    """Trigger processing manually if needed or called directly by frontend."""
    try:
        payload = request.json or {}
        exam_id = payload.get("examId")
        teacher_id = payload.get("teacherId")

        if not exam_id or not teacher_id:
            return jsonify({"error": "Missing examId or teacherId"}), 400

        doc = db.collection("teacherExamUploads").document(teacher_id).get()
        if not doc.exists:
            return jsonify({"error": "No uploads record found for this teacher"}), 404

        meta = None
        for upload in doc.to_dict().get("uploads", []):
            if upload.get("examId") == exam_id or upload.get("id") == exam_id:
                meta = upload
                break

        if not meta:
            return jsonify({"error": "Exam upload metadata structure mismatch"}), 404

        # Start standard long-running operations inside a detached context thread
        threading.Thread(
            target=run_extraction_pipeline,
            args=(exam_id, meta, teacher_id)
        ).start()

        return jsonify({"status": "processing", "examId": exam_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/admin/debug-memo/<exam_id>", methods=["GET"])
def debug_memo(exam_id):
    """Show exactly what memo text was extracted from storage and what answers were parsed."""
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

        memo_path = meta.get("memoStoragePath")
        memo_fn = meta.get("memoFileName", "memo.pdf")

        if not memo_path:
            return jsonify({"error": "No memo file on this upload entry point"})

        # Download memo from Cloud Storage
        memo_bytes = download_file_bytes_from_storage(memo_path)
        if not memo_bytes:
            return jsonify({"error": "Could not download memo from Storage structural block"})

        # Extract text
        memo_text = extract_text_from_file(memo_bytes, memo_fn)

        # Parse answers
        memo_map = parse_memo_answers(memo_text, meta.get("subject", ""), meta.get("grade", "12"))

        # Get structural comparison properties
        exam_q_nums = []
        for q in db.collection("exam_questions").where("examId", "==", exam_id).stream():
            exam_q_nums.append(q.to_dict().get("questionNumber", ""))

        matched = []
        unmatched = []
        norm_memo = {_normalise_qnum(k): v for k, v in memo_map.items()}
        for qn in exam_q_nums:
            if _normalise_qnum(qn) in norm_memo:
                matched.append(qn)
            else:
                unmatched.append(qn)

        return jsonify({
            "memo_file": memo_fn,
            "memo_text_length": len(memo_text),
            "memo_text_preview": memo_text[:2000],
            "memo_answers_count": len(memo_map),
            "memo_answers_sample": dict(list(memo_map.items())[:10]),
            "exam_questions_count": len(exam_q_nums),
            "matched_count": len(matched),
            "unmatched_count": len(unmatched),
            "unmatched_sample": unmatched[:20],
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/admin/reset-exam/<exam_id>", methods=["GET"])
def reset_exam(exam_id):
    """Wipe exam + questions from Firestore so extraction runs fresh."""
    try:
        batch = db.batch()
        count = 0
        for doc in db.collection("exam_questions").where("examId", "==", exam_id).stream():
            batch.delete(doc.reference)
            count += 1
            if count % 400 == 0:
                batch.commit()
                batch = db.batch()
        batch.commit()

        db.collection("exams").document(exam_id).delete()

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

@app.route('/')
def home():
    return {
        "status": "online",
        "message": "Eduket Backend API is running successfully!",
        "engine": "Groq-Native RAG"
    }, 200

if __name__ == "__main__":
    # Render passes a dynamic port via environment variables.
    # Locally, it defaults to 5000.
    port = int(os.environ.get("PORT", 5000))

    print(f"\n🚀 Eduket Server initializing on port {port}...")
    print(f"👉 Local access: http://127.0.0.1:{port}")

    # host="0.0.0.0" allows external access (critical for Docker/Render deployments)
    app.run(host="0.0.0.0", port=port, debug=True)