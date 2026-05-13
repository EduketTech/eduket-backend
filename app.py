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
    raw = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON") or os.getenv("FIREBASE_SERVICE_ACCOUNT")
    creds = service_account.Credentials.from_service_account_info(
        json.loads(raw),
        scopes=["https://www.googleapis.com/auth/drive.readonly"]
    )
    creds.refresh(GoogleRequest())
    return creds.token


def download_file_bytes(file_id: str):
    try:
        token = _drive_token()
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
    """
    Extract all text from a .docx Word document using python-docx.
    Handles paragraphs and tables. Best format for exam papers — no OCR needed.
    """
    try:
        from docx import Document
        from docx.oxml.ns import qn
        doc = Document(io.BytesIO(file_bytes))
        lines = []

        for element in doc.element.body:
            tag = element.tag.split("}")[-1] if "}" in element.tag else element.tag

            if tag == "p":
                para_text = "".join(
                    node.text or "" for node in element.iter()
                    if node.tag in (qn("w:t"), "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t")
                )
                if para_text.strip():
                    lines.append(para_text.strip())

            elif tag == "tbl":
                for row in element.iter(qn("w:tr")):
                    cells = []
                    for cell in row.iter(qn("w:tc")):
                        cell_text = "".join(
                            node.text or "" for node in cell.iter()
                            if node.tag in (qn("w:t"), "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t")
                        )
                        if cell_text.strip():
                            cells.append(cell_text.strip())
                    if cells:
                        lines.append(" | ".join(cells))

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
    """Route to correct extractor based on file extension."""
    name = filename.lower()
    if name.endswith(".docx") or name.endswith(".doc"):
        print(f"[Extract] Word doc: {filename}")
        text = extract_text_from_docx(file_bytes)
        if not text.strip():
            print("[Extract] DOCX empty — trying PDF fallback")
            text = extract_text_from_pdf(file_bytes)
        return text
    elif name.endswith(".pdf"):
        print(f"[Extract] PDF: {filename}")
        return extract_text_from_pdf(file_bytes)
    else:
        print(f"[Extract] Unknown type {filename} — trying docx then pdf")
        text = extract_text_from_docx(file_bytes)
        if not text.strip():
            text = extract_text_from_pdf(file_bytes)
        return text


# ═══════════════════════════════════════════════════════════════════════════════
# QUESTION PARSER
# ═══════════════════════════════════════════════════════════════════════════════

def parse_questions_from_text(text: str, subject: str, grade: str) -> list:
    """Parse questions in chunks to handle full-length papers."""
    if not text or not text.strip():
        return []

    # Split into overlapping 12000-char chunks
    CHUNK = 12000
    OVERLAP = 500
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + CHUNK])
        start += CHUNK - OVERLAP

    all_questions = []
    seen_numbers = set()

    for i, chunk in enumerate(chunks):
        print(f"[Parse] Chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")
        qs = _parse_chunk(chunk, subject, grade)
        for q in qs:
            qn = _normalise_qnum(q.get("question_number", ""))
            if qn and qn not in seen_numbers:
                seen_numbers.add(qn)
                all_questions.append(q)

    print(f"[Parse] Total unique questions: {len(all_questions)}")
    return all_questions


def _normalise_qnum(qn: str) -> str:
    """Normalise question number for matching — strip dots, spaces, lowercase."""
    return re.sub(r"[\s\.\-]+", "", str(qn)).lower().strip()


def _parse_chunk(text: str, subject: str, grade: str) -> list:
    """Parse a single text chunk into questions."""
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    prompt = f"""You are an expert parser for South African NSC CAPS exam papers.

Extract EVERY question from the text below into a JSON array.

RULES:
- Create a SEPARATE item for EACH sub-question (1.1, 1.2, 2.1.1 etc.)
- For MCQ: type="mcq", options={{"A":"...","B":"...","C":"...","D":"..."}}
- For True/False: type="true_false"
- For matching columns: type="matching", column_a=[...], column_b=[...]
- For calculations: type="calculation"
- For short answers: type="short_answer"
- For essays: type="essay"
- Default: type="open"
- marks = integer from brackets e.g. (2) or [3] — default 1 if not found
- section = section letter/number (A, B, 1, 2 etc.)
- parent_question = main heading e.g. "QUESTION 1"
- For diagrams include [DIAGRAM: description] in question text
- memo = correct answer if visible, else null
- NEVER skip questions

Return ONLY a valid JSON array. No markdown, no explanation.
Subject: {subject} | Grade: {grade}

EXAM TEXT:
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
        raw = re.sub(r"^```json\s*|^```\s*|```$", "", raw, flags=re.MULTILINE).strip()
        result = json.loads(raw)
        return result if isinstance(result, list) else []
    except Exception as e:
        print(f"[Parse] Chunk error: {e}")
        return []


    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #MEMO PARSER
    #==============================================================
def parse_memo_answers(text: str, subject: str, grade: str) -> dict:
    """
    Parse a memo document into a question_number → answer map.
    Uses answer-focused prompt instead of question-focused prompt.
    Processes in chunks to cover full document.
    """
    if not text or not text.strip():
        return {}

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
        memo_map.update(chunk_map)

    print(f"[Memo] Total answers extracted: {len(memo_map)}")
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
        raw = re.sub(r"^```json\s*|^```\s*|```$", "", raw, flags=re.MULTILINE).strip()
        result = json.loads(raw)
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
        exam_bytes = download_file_bytes(exam_fid)
        if not exam_bytes:
            raise ValueError(f"Could not download exam file (Drive id: {exam_fid})")

        # 2. Extract text
        exam_text = extract_text_from_file(exam_bytes, exam_fn)
        if not exam_text.strip():
            raise ValueError("No text extracted from exam file")
        print(f"[Pipeline] Exam text: {len(exam_text)} chars")

        # 3. Parse questions
        questions = parse_questions_from_text(exam_text, subject, grade)
        print(f"[Pipeline] Questions: {len(questions)}")

        # ── 4. Download and parse memo ─────────────────────────────────────
        memo_map = {}
        if memo_fid:
            memo_bytes = download_file_bytes(memo_fid)
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
@app.route("/auto-extract", methods=["POST"])
def auto_extract():
    try:
        data = request.get_json()
        exam_id = data.get("exam_id", "").strip()
        if not exam_id:
            return jsonify({"error": "exam_id required"}), 400

        meta = None
        teacher_doc_id = None

        # Stream all teacher upload docs to find the matching exam_id
        for doc in db.collection("teacherExamUploads").stream():
            for upload in doc.to_dict().get("uploads", []):
                if upload.get("examId") == exam_id or upload.get("id") == exam_id:
                    meta = upload
                    teacher_doc_id = doc.id
                    break
            if meta:
                break

        if not meta:
            return jsonify({"error": f"Upload {exam_id} not found"}), 404
        if not meta.get("examDriveFileId"):
            return jsonify({"error": "No examDriveFileId"}), 400

        # Skip if already extracted
        if meta.get("status") == "extracted":
            exam_doc = db.collection("exams").document(exam_id).get()
            if exam_doc.exists and exam_doc.to_dict().get("status") == "ready":
                return jsonify({"ok": True, "message": "Already extracted", "exam_id": exam_id})

        thread = threading.Thread(
            target=run_extraction_pipeline,
            args=(exam_id, meta, teacher_doc_id),
            daemon=True,
        )
        thread.start()

        return jsonify({
            "ok": True,
            "message": "Extraction started",
            "exam_id": exam_id,
            "poll_status": f"/admin/extraction-status/{exam_id}",
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/admin/trigger-extract/<exam_id>", methods=["GET"])
def trigger_extract(exam_id):
    """Manual admin trigger — same as auto-extract."""
    try:
        meta = None
        teacher_doc_id = None
        for doc in db.collection("teacherExamUploads").stream():
            for upload in doc.to_dict().get("uploads", []):
                if upload.get("examId") == exam_id or upload.get("id") == exam_id:
                    meta = upload
                    teacher_doc_id = doc.id
                    break
            if meta:
                break

        if not meta:
            return jsonify({"error": f"Exam {exam_id} not found"}), 404
        if not meta.get("examDriveFileId"):
            return jsonify({"error": "No examDriveFileId"}), 400

        thread = threading.Thread(
            target=run_extraction_pipeline,
            args=(exam_id, meta, teacher_doc_id),
            daemon=True,
        )
        thread.start()

        return jsonify({
            "ok": True,
            "message": "Extraction started in background",
            "exam_id": exam_id,
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
                if upload.get("examId") == exam_id:
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


@app.route("/start-exam", methods=["POST"])
def start_exam():
    try:
        data       = request.get_json()
        exam_id    = data.get("exam", "").strip()
        student_id = data.get("student_id", "anonymous")

        if not exam_id:
            return jsonify({"error": "No exam specified"})

        meta, questions = load_exam_from_firestore(exam_id)
        if meta is None:
            return jsonify({"error": f"Exam '{exam_id}' not found"})
        if not questions:
            return jsonify({"error": (
                f"This exam has no questions yet — extraction may still be in progress "
                f"(status: {meta.get('status', 'unknown')}). Please wait a minute and try again."
            )})

        mem.ensure_student(student_id)
        sid = str(uuid.uuid4())

        save_session_to_fs(sid, {
            "exam_id":    exam_id,
            "exam":       meta.get("title", exam_id),
            "subject":    meta.get("subject", ""),
            "student_id": student_id,
            "questions":  questions,
            "answers":    {},
        })

        return jsonify({
            "session_id":      sid,
            "total_questions": len(questions),
            "memo_merged":     meta.get("memoMerged", False),
            "subject":         meta.get("subject", ""),
            "title":           meta.get("title", ""),
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)})


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


if __name__ == "__main__":
    app.run(debug=True, port=8000)