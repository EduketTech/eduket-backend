"""
app.py — EduCAT Flask API (Clean Prototype)

ARCHITECTURE (simple, prototype-ready):
  - Teachers upload PDFs → stored in teacherExamUploads (nested array structure)
  - Admin triggers extraction → Groq vision reads PDF pages as images → questions stored in Firestore
  - Students see ready exams → answer questions → get AI-marked results

EXTRACTION PIPELINE (all free):
  - pymupdf renders PDF pages as images (no Drive write needed)
  - Groq vision model (llama-4-scout) reads each page and extracts questions
  - Questions stored in exam_questions collection
  - Memo extracted from memo PDF and merged

COLLECTIONS USED:
  teacherExamUploads/{teacherId}/uploads[]  — raw upload metadata
  exams/{examId}                            — extracted exam metadata (status: ready)
  exam_questions/{examId}_{i}              — individual questions with memos
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
    "https://edu-cat.netlify.app",
]}})

rag = RAGIndex()
agent.set_rag(rag)

sessions = {}  # in-memory session store


# ═══════════════════════════════════════════════════════════════════════════════
# DRIVE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _get_drive_token():
    """Get a Bearer token for Drive API using service account."""
    from google.oauth2 import service_account
    from google.auth.transport.requests import Request as GoogleRequest
    raw = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON") or os.getenv("FIREBASE_SERVICE_ACCOUNT")
    creds = service_account.Credentials.from_service_account_info(
        json.loads(raw),
        scopes=["https://www.googleapis.com/auth/drive.readonly"]
    )
    creds.refresh(GoogleRequest())
    return creds.token


def download_pdf_bytes(file_id: str) -> bytes | None:
    """Download a file from Google Drive as bytes."""
    try:
        token = _get_drive_token()
        res = http_requests.get(
            f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media",
            headers={"Authorization": f"Bearer {token}"},
            timeout=60,
        )
        if res.status_code == 200:
            return res.content
        print(f"[Drive] download failed {res.status_code}: {res.text[:200]}")
        return None
    except Exception as e:
        print(f"[Drive] error: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# PDF → TEXT EXTRACTION
# Uses pymupdf to render pages as images, then Groq vision to read them.
# Works on scanned PDFs. No Drive write permission needed.
# ═══════════════════════════════════════════════════════════════════════════════

def extract_text_from_pdf(pdf_bytes: bytes, max_pages: int = 10) -> str:
    """
    Extract text from PDF.
    First tries native text extraction (fast, works on digital PDFs).
    Falls back to Groq vision OCR (works on scanned PDFs).
    """
    text = ""

    # Stage 1: native text extraction via pymupdf
    try:
        import fitz
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
        if len(text.strip()) > 200:
            print(f"[PDF] Native text extracted: {len(text)} chars")
            return text
    except Exception as e:
        print(f"[PDF] pymupdf native: {e}")

    # Stage 2: pdfplumber fallback
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                text += (page.extract_text() or "") + "\n"
        if len(text.strip()) > 200:
            print(f"[PDF] pdfplumber extracted: {len(text)} chars")
            return text
    except Exception as e:
        print(f"[PDF] pdfplumber: {e}")

    # Stage 3: Groq vision OCR — renders each page as image
    print("[PDF] Scanned PDF detected — using Groq vision OCR")
    text = _groq_vision_ocr(pdf_bytes, max_pages)
    return text


def _groq_vision_ocr(pdf_bytes: bytes, max_pages: int = 10) -> str:
    """Render PDF pages as images and OCR them with Groq vision."""
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    all_text = ""

    try:
        import fitz
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages_to_process = min(max_pages, len(doc))
        print(f"[OCR] Processing {pages_to_process} of {len(doc)} pages")

        for page_num in range(pages_to_process):
            try:
                page = doc[page_num]
                # Render at 150 DPI — good quality, reasonable size
                mat = fitz.Matrix(150 / 72, 150 / 72)
                pix = page.get_pixmap(matrix=mat)
                img_b64 = base64.b64encode(pix.tobytes("png")).decode()

                response = client.chat.completions.create(
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                            },
                            {
                                "type": "text",
                                "text": (
                                    "This is a page from a South African NSC/CAPS exam paper. "
                                    "Extract ALL text exactly as it appears. "
                                    "Preserve question numbers (e.g. 1.1, 1.2, 2.1.1), marks in brackets like (2), "
                                    "multiple choice options labeled A B C D, "
                                    "table contents row by row, "
                                    "and any diagram descriptions or figure labels. "
                                    "For diagrams or images you can see, write: [DIAGRAM: brief description of what it shows] "
                                    "Output plain text only, no commentary, no markdown."
                                )
                            }
                        ]
                    }],
                    max_tokens=2000,
                )
                page_text = response.choices[0].message.content.strip()
                all_text += f"\n--- PAGE {page_num + 1} ---\n{page_text}\n"
                print(f"[OCR] Page {page_num + 1}: {len(page_text)} chars")

            except Exception as e:
                print(f"[OCR] Page {page_num + 1} failed: {e}")
                continue

        doc.close()

    except Exception as e:
        print(f"[OCR] fitz error: {e}")

    return all_text


# ═══════════════════════════════════════════════════════════════════════════════
# TEXT → QUESTIONS (Groq)
# ═══════════════════════════════════════════════════════════════════════════════

def parse_questions_from_text(text: str, subject: str, grade: str) -> list:
    """
    Send extracted text to Groq and get back structured questions.
    Returns list of question dicts.
    """
    if not text or not text.strip():
        return []

    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # Trim to Groq context limit
    text = text[:14000]

    prompt = f"""You are an expert parser for South African NSC CAPS exam papers.

    Extract EVERY question from the text below into a JSON array.

    Rules:
    - Include ALL sub-questions (1.1, 1.2, 2.1.1 etc.)
    - For MCQ include options as {{"A": "...", "B": "...", "C": "...", "D": "..."}}
    - For True/False use type "true_false"
    - For matching use type "matching" with column_a and column_b arrays
    - For calculations use type "calculation"
    - For short answers use type "short_answer"
    - For essays use type "essay"
    - Default type is "open"
    - marks must be an integer (look for numbers in brackets like (2) or [3])
    - section is the letter/number of the section (A, B, 1, 2 etc.)
    - If the question refers to a diagram, table or figure, include [DIAGRAM: description] in the question text
    - memo: include if answer visible in text, else null
    - If a question has sub-parts, create a SEPARATE item for each sub-part
    - Never skip questions even if they refer to diagrams

    Return ONLY a valid JSON array. No markdown, no explanation.

    Each item:
    {{
      "question_number": "1.1",
      "parent_question": "QUESTION 1",
      "section": "A",
      "question": "full question text here",
      "type": "mcq",
      "marks": 2,
      "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
      "column_a": null,
      "column_b": null,
      "memo": null
    }}

    Subject: {subject}
    Grade: {grade}

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
        # Strip markdown fences if present
        raw = re.sub(r"^```json\s*|^```\s*|```$", "", raw, flags=re.MULTILINE).strip()
        questions = json.loads(raw)
        print(f"[Parse] Groq returned {len(questions)} questions")
        return questions if isinstance(questions, list) else []
    except json.JSONDecodeError as e:
        print(f"[Parse] JSON decode error: {e}")
        print(f"[Parse] Raw response: {raw[:500]}")
        return []
    except Exception as e:
        print(f"[Parse] Groq error: {e}")
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# FULL EXTRACTION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_extraction_pipeline(exam_id: str, meta: dict, teacher_doc_id: str):
    """
    Full pipeline: Download PDF → extract text → parse questions → save to Firestore.
    Runs in a background thread to avoid Render's 30s timeout.
    Updates status in Firestore throughout so admin can monitor progress.
    """
    doc_ref = db.collection("teacherExamUploads").document(teacher_doc_id)

    def update_upload_status(status: str, extra: dict = {}):
        try:
            teacher_data = doc_ref.get().to_dict()
            updated = []
            for upload in teacher_data.get("uploads", []):
                if upload.get("examId") == exam_id or upload.get("id") == exam_id:
                    upload["status"] = status
                    upload.update(extra)
                updated.append(upload)
            doc_ref.update({"uploads": updated})
        except Exception as e:
            print(f"[Pipeline] Failed to update status: {e}")

    try:
        subject = meta.get("subject", "General")
        grade = meta.get("grade", "12")
        exam_file_id = meta.get("examDriveFileId")
        memo_file_id = meta.get("memoDriveFileId")

        print(f"[Pipeline] Starting: {exam_id} | {subject} Grade {grade}")
        update_upload_status("processing")

        # ── Step 1: Download exam PDF ──────────────────────────────────────
        print(f"[Pipeline] Downloading exam PDF: {exam_file_id}")
        pdf_bytes = download_pdf_bytes(exam_file_id)
        if not pdf_bytes:
            raise ValueError(f"Could not download exam PDF from Drive (file_id: {exam_file_id})")

        # ── Step 2: Extract text from exam PDF ─────────────────────────────
        print(f"[Pipeline] Extracting text from exam ({len(pdf_bytes)} bytes)")
        exam_text = extract_text_from_pdf(pdf_bytes, max_pages=12)
        if not exam_text.strip():
            raise ValueError("No text could be extracted from exam PDF")
        print(f"[Pipeline] Exam text: {len(exam_text)} chars")

        # ── Step 3: Parse questions from exam text ─────────────────────────
        print(f"[Pipeline] Parsing questions with Groq")
        questions = parse_questions_from_text(exam_text, subject, grade)
        print(f"[Pipeline] Got {len(questions)} questions")

        # ── Step 4: Extract memo if available ─────────────────────────────
        memo_map = {}
        if memo_file_id:
            print(f"[Pipeline] Downloading memo PDF: {memo_file_id}")
            memo_bytes = download_pdf_bytes(memo_file_id)
            if memo_bytes:
                memo_text = extract_text_from_pdf(memo_bytes, max_pages=20)
                if memo_text.strip():
                    memo_questions = parse_questions_from_text(
                        memo_text, subject + " MEMO", grade
                    )
                    # Build map: question_number → memo answer
                    for mq in memo_questions:
                        qn = mq.get("question_number")
                        answer = mq.get("memo") or mq.get("question", "")
                        if qn and answer:
                            memo_map[qn] = answer
                    print(f"[Pipeline] Memo map: {len(memo_map)} answers")

        # ── Step 5: Merge memo answers into questions ──────────────────────
        for q in questions:
            qn = q.get("question_number")
            if qn and qn in memo_map and not q.get("memo"):
                q["memo"] = memo_map[qn]

        # ── Step 6: If 0 questions parsed, create a placeholder ───────────
        # This ensures the exam shows up for students even if parsing failed
        # Teacher can re-extract later
        if not questions:
            print("[Pipeline] WARNING: 0 questions parsed — creating placeholder")
            questions = [{
                "question_number": "1",
                "parent_question": "",
                "section": "A",
                "question": f"[Exam text extracted but questions could not be auto-parsed. Please contact your teacher. Subject: {subject}]",
                "type": "open",
                "marks": 0,
                "options": None,
                "column_a": None,
                "column_b": None,
                "memo": None,
            }]

        # ── Step 7: Write exam doc to Firestore ────────────────────────────
        print(f"[Pipeline] Writing {len(questions)} questions to Firestore")
        db.collection("exams").document(exam_id).set({
            "title": meta.get("title", meta.get("examFileName", "Exam")),
            "subject": subject,
            "grade": grade,
            "year": meta.get("year", ""),
            "curriculum": meta.get("curriculum", "CAPS"),
            "teacherName": meta.get("teacherName", ""),
            "uploadedBy": meta.get("uploadedBy", ""),
            "examDriveFileId": exam_file_id,
            "memoDriveFileId": memo_file_id,
            "memoMerged": bool(memo_map),
            "status": "ready",
            "totalQuestions": len(questions),
            "extractedAt": fs_admin.SERVER_TIMESTAMP,
            "sourceUploadId": exam_id,
        })

        # ── Step 8: Write questions to exam_questions collection ───────────
        # Use batches of 400 to stay under Firestore 500-op limit
        batch = db.batch()
        for i, q in enumerate(questions):
            q_ref = db.collection("exam_questions").document(f"{exam_id}_{i:04d}")
            batch.set(q_ref, {
                "examId": exam_id,
                "questionNumber": q.get("question_number", str(i + 1)),
                "parentQuestion": q.get("parent_question", ""),
                "section": q.get("section", "A"),
                "questionText": q.get("question", ""),
                "type": q.get("type", "open"),
                "marks": int(q.get("marks", 1)),
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

        # ── Step 9: Mark upload as extracted ──────────────────────────────
        update_upload_status("extracted", {
            "extractedAt": datetime.utcnow().isoformat(),
            "totalQuestions": len(questions),
            "memoMerged": bool(memo_map),
        })

        print(f"[Pipeline] ✅ Done — {len(questions)} questions, {len(memo_map)} memo answers")

    except Exception as e:
        traceback.print_exc()
        print(f"[Pipeline] ❌ Failed: {e}")
        update_upload_status("error", {"errorMessage": str(e)})
        # Also update exams doc if it was partially written
        try:
            db.collection("exams").document(exam_id).set(
                {"status": "error", "errorMessage": str(e)}, merge=True
            )
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# EXAM LOADING (for student sessions)
# ═══════════════════════════════════════════════════════════════════════════════

def load_exam_from_firestore(exam_id: str):
    """Load exam metadata + questions from Firestore."""
    try:
        doc = db.collection("exams").document(exam_id).get()
        if not doc.exists:
            return None, []
        meta = doc.to_dict()
        meta["id"] = doc.id

        # Load questions — sort in Python to avoid needing Firestore composite index
        q_docs = (
            db.collection("exam_questions")
            .where("examId", "==", exam_id)
            .stream()
        )

        questions = []
        for q in q_docs:
            d = q.to_dict()
            options = d.get("options")
            if isinstance(options, dict) and options:
                options = [{"key": k, "value": v} for k, v in sorted(options.items())]
            elif isinstance(options, list) and options and isinstance(options[0], str):
                options = [{"key": chr(65 + i), "value": v} for i, v in enumerate(options)]

            questions.append({
                "question_number": str(d.get("questionNumber", "")),
                "parent_question": d.get("parentQuestion", ""),
                "parent_context": d.get("parentContext"),
                "section": d.get("section", "A"),
                "section_title": d.get("sectionTitle", ""),
                "section_instructions": d.get("sectionInstructions", ""),
                "section_total_marks": d.get("sectionTotalMarks"),
                "question": d.get("questionText", "Question text missing"),
                "type": d.get("type", "open").lower(),
                "options": options,
                "column_a": d.get("columnA"),
                "column_b": d.get("columnB"),
                "marks": d.get("marks", 1),
                "memo": d.get("memo", ""),
                "saved_answer": "",
                "order": d.get("order", 0),
            })

        # Sort by order in Python — no index needed
        questions.sort(key=lambda x: x.get("order", 0))

        # Remove the order field before sending to frontend
        for q in questions:
            q.pop("order", None)

        return meta, questions

    except Exception as e:
        traceback.print_exc()
        return None, []

# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES — ADMIN
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/admin/uploads", methods=["GET"])
def admin_uploads():
    """List all teacher uploads — flattens nested array structure."""
    try:
        uploads = []
        for doc in db.collection("teacherExamUploads").stream():
            d = doc.to_dict()
            for upload in d.get("uploads", []):
                upload["teacherDocId"] = doc.id
                uploads.append(upload)
        uploads.sort(key=lambda x: x.get("uploadedAt", ""), reverse=True)
        return jsonify({"uploads": uploads})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/admin/trigger-extract/<exam_id>", methods=["GET"])
def trigger_extract(exam_id):
    """
    Trigger extraction for an uploaded exam.
    Returns immediately — runs in background thread.
    Poll /admin/extraction-status/<exam_id> to check progress.
    """
    try:
        # Find the upload in nested structure
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
            return jsonify({"error": f"Exam {exam_id} not found in teacherExamUploads"}), 404

        if not meta.get("examDriveFileId"):
            return jsonify({"error": "No examDriveFileId on this upload"}), 400

        # Start background extraction
        thread = threading.Thread(
            target=run_extraction_pipeline,
            args=(exam_id, meta, teacher_doc_id),
            daemon=True,
        )
        thread.start()

        return jsonify({
            "ok": True,
            "message": "Extraction started in background (takes 1-3 minutes for scanned PDFs)",
            "exam_id": exam_id,
            "poll_status": f"/admin/extraction-status/{exam_id}",
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/admin/extraction-status/<exam_id>", methods=["GET"])
def extraction_status(exam_id):
    """Poll this to check if extraction is done."""
    try:
        exam_doc = db.collection("exams").document(exam_id).get()
        if exam_doc.exists:
            d = exam_doc.to_dict()
            q_count = sum(1 for _ in db.collection("exam_questions").where("examId", "==", exam_id).stream())
            return jsonify({
                "status": d.get("status"),
                "title": d.get("title"),
                "subject": d.get("subject"),
                "questions_in_firestore": q_count,
                "memo_merged": d.get("memoMerged", False),
                "student_accessible": d.get("status") == "ready" and q_count > 0,
            })

        # Not in exams yet — check upload doc
        for doc in db.collection("teacherExamUploads").stream():
            for upload in doc.to_dict().get("uploads", []):
                if upload.get("examId") == exam_id:
                    return jsonify({
                        "status": upload.get("status", "pending_extraction"),
                        "error": upload.get("errorMessage"),
                        "student_accessible": False,
                    })

        return jsonify({"status": "not_found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES — STUDENT EXAM
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/exams", methods=["GET"])
def list_exams():
    """Return all ready exams for students to pick from."""
    exams = []
    try:
        for doc in db.collection("exams").where("status", "==", "ready").stream():
            d = doc.to_dict()
            exams.append({
                "id": doc.id,
                "name": d.get("title", doc.id),
                "subject": d.get("subject", ""),
                "grade": d.get("grade", ""),
                "year": d.get("year", ""),
                "curriculum": d.get("curriculum", "CAPS"),
                "memoMerged": d.get("memoMerged", False),
            })
    except Exception as e:
        print(f"[list_exams] {e}")
    return jsonify({"exams": exams})


@app.route("/start-exam", methods=["POST"])
def start_exam():
    """Start an exam session for a student."""
    try:
        data = request.get_json()
        exam_id = data.get("exam", "").strip()
        student_id = data.get("student_id", "anonymous")

        if not exam_id:
            return jsonify({"error": "No exam specified"})

        meta, questions = load_exam_from_firestore(exam_id)

        if meta is None:
            return jsonify({"error": f"Exam '{exam_id}' not found"})

        if not questions:
            return jsonify({"error": f"Exam has no questions yet (status: {meta.get('status')}). Please wait for extraction to complete."})

        mem.ensure_student(student_id)
        sid = str(uuid.uuid4())
        sessions[sid] = {
            "exam_id": exam_id,
            "exam": meta.get("title", exam_id),
            "subject": meta.get("subject", ""),
            "student_id": student_id,
            "questions": questions,
            "answers": {},
        }

        return jsonify({
            "session_id": sid,
            "total_questions": len(questions),
            "memo_merged": meta.get("memoMerged", False),
            "subject": meta.get("subject", ""),
            "title": meta.get("title", ""),
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)})


@app.route("/question", methods=["POST"])
def get_question():
    """Get a single question by index for the current session."""
    try:
        data = request.get_json()
        session = sessions.get(data.get("session_id"))
        if not session:
            return jsonify({"error": "Invalid session"})

        idx = int(data.get("index", 0))
        questions = session["questions"]

        if idx < 0 or idx >= len(questions):
            return jsonify({"error": "Index out of range"})

        q = questions[idx].copy()
        q["saved_answer"] = session["answers"].get(str(idx), "")
        return jsonify(q)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)})


@app.route("/answer", methods=["POST"])
def save_answer():
    """Save a student's answer for a question."""
    try:
        data = request.get_json()
        session = sessions.get(data.get("session_id"))
        if not session:
            return jsonify({"error": "Invalid session"})
        session["answers"][str(data.get("index"))] = data.get("answer", "")
        return jsonify({"status": "saved"})
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/submit", methods=["POST"])
def submit_exam():
    """Mark all answers and return results with AI feedback."""
    try:
        data = request.get_json()
        session = sessions.get(data.get("session_id"))
        student_id = data.get("student_id", "anonymous")

        if not session:
            return jsonify({"error": "Invalid session"})

        subject = session.get("subject", "")
        questions = session["questions"]
        answers = session["answers"]
        results = []
        total_score = 0
        total_marks = 0

        for i, q in enumerate(questions):
            q_num = q.get("question_number", f"Q{i+1}")
            q_type = q.get("type", "open").lower()
            marks = int(q.get("marks", 1))
            memo = q.get("memo", "")
            student_ans = answers.get(str(i), "").strip()

            # Normalise options for marking
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

            # Track weak topics
            if result.get("status") in ("incorrect", "missing"):
                mem.record_wrong(student_id, q_num, q.get("question", ""), q_type, subject)
            elif result.get("status") == "correct":
                mem.record_correct(student_id, q_num)

            # Format correct answer display
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
                "question": q.get("question", ""),
                "type": q_type,
                "marks": marks,
                "earned": result.get("score", 0),
                "student_answer": student_ans or "No answer",
                "correct_answer": correct_display,
            })
            total_score += result.get("score", 0)
            total_marks += marks

        percentage = round(total_score / total_marks * 100, 1) if total_marks else 0
        mem.save_session(student_id, session["exam"], total_score, total_marks, percentage, subject=subject)
        feedback = generate_exam_feedback(results, total_score, total_marks, percentage, subject=subject)

        # Update study plan via agent
        weak = mem.get_weak_topics(student_id)
        if weak:
            try:
                run_agent(
                    student_id,
                    f"I just scored {percentage}% on {session['exam']} ({subject}). Update my study plan.",
                    rag=rag,
                )
            except Exception:
                pass

        return jsonify({
            "score": total_score,
            "total": total_marks,
            "percentage": percentage,
            "results": results,
            "feedback": feedback,
            "subject": subject,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)})


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES — AI AGENT + DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/agent-chat", methods=["POST"])
def agent_chat():
    try:
        data = request.get_json()
        student_id = data.get("student_id", "anonymous")
        message = data.get("message", "").strip()
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


@app.route("/dashboard", methods=["POST"])
def dashboard():
    try:
        data = request.get_json()
        student_id = data.get("student_id", "anonymous")
        mem.ensure_student(student_id)
        return jsonify({
            "weak": mem.get_weak_topics(student_id),
            "sessions": mem.get_sessions(student_id, limit=8),
            "study_plan": mem.get_study_plan(student_id),
        })
    except Exception as e:
        return jsonify({"error": str(e)})


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES — HEALTH CHECK
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "EduCAT API",
        "endpoints": {
            "exams": "GET /exams",
            "start": "POST /start-exam",
            "question": "POST /question",
            "answer": "POST /answer",
            "submit": "POST /submit",
            "agent": "POST /agent-chat",
            "admin_uploads": "GET /admin/uploads",
            "admin_extract": "GET /admin/trigger-extract/<exam_id>",
            "admin_status": "GET /admin/extraction-status/<exam_id>",
        }
    })


if __name__ == "__main__":
    app.run(debug=True, port=8000)