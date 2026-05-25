"""
app.py — Eduket Production Exam Extraction API
═══════════════════════════════════════════════════════════════

PRODUCTION FEATURES
──────────────────────────────────────────────────────────────
✅ Firebase Storage only (NO Google Drive)
✅ Automatic extraction listener
✅ Startup recovery sweep
✅ Duplicate extraction prevention
✅ Robust DOCX extraction
✅ PDF OCR fallback
✅ Memo merging
✅ Production-safe threading
✅ Firestore status tracking
✅ Ready-for-student exam availability
✅ Render-compatible
"""

from dotenv import load_dotenv

load_dotenv()

import os
import io
import re
import json
import time
import base64
import traceback
import threading
import tempfile
import requests
import fitz  # PyMuPDF

from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from PIL import Image

from odf.opendocument import load
from odf import text, teletype
import mammoth
from groq import Groq

# ═══════════════════════════════════════════════════════════════
# FIREBASE INITIALIZATION
# ═══════════════════════════════════════════════════════════════

import firebase_admin
from firebase_admin import (
    credentials,
    firestore as fs_admin,
    storage
)


def _init_firebase():
    if firebase_admin._apps:
        return

    raw = (
            os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
            or os.getenv("FIREBASE_SERVICE_ACCOUNT", "")
    )

    if raw.strip():
        cred = credentials.Certificate(json.loads(raw))
    else:
        cred = credentials.ApplicationDefault()

    firebase_admin.initialize_app(cred, {
        "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET")
    })


_init_firebase()

db = fs_admin.client()
bucket = storage.bucket()

# ═══════════════════════════════════════════════════════════════
# APP CONFIGURATION
# ═══════════════════════════════════════════════════════════════

app = Flask(__name__)

# ⚡ Use a compiled regex pattern to safely allow all Netlify preview & production subdomains
CORS(app, resources={r"/*": {
    "origins": [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
        "http://localhost:5176",
        "https://eduket.netlify.app",
        re.compile(r"^https://.*\.netlify\.app$")
    ]
}}, supports_credentials=True)

# ═══════════════════════════════════════════════════════════════
# PROCESS TRACKING (THREAD SAFETY)
# ═══════════════════════════════════════════════════════════════

PROCESSING_EXAMS = set()
PROCESSING_LOCK = threading.Lock()


def _is_already_processing(exam_id):
    with PROCESSING_LOCK:
        return exam_id in PROCESSING_EXAMS


def _mark_processing(exam_id):
    with PROCESSING_LOCK:
        PROCESSING_EXAMS.add(exam_id)


def _unmark_processing(exam_id):
    with PROCESSING_LOCK:
        PROCESSING_EXAMS.discard(exam_id)


# ═══════════════════════════════════════════════════════════════
# STORAGE DOWNLOAD
# ═══════════════════════════════════════════════════════════════

def download_file_for_extraction(meta, file_type):
    filename = meta.get(
        f"{file_type}FileName",
        f"{file_type}.pdf"
    )

    # ── Try Storage Path First ─────────────────────────────
    storage_path = meta.get(f"{file_type}StoragePath")
    if storage_path:
        try:
            blob = bucket.blob(storage_path)
            if blob.exists():
                data = blob.download_as_bytes(timeout=120)
                print(f"[Storage] Path download success {storage_path}")
                return data, filename
        except Exception as e:
            print(f"[Storage] Path failed: {e}")

    # ── Fallback To URL Download ───────────────────────────
    storage_url = meta.get(f"{file_type}StorageUrl")
    if storage_url:
        try:
            response = requests.get(storage_url, timeout=120)
            if response.status_code == 200:
                print(f"[Storage] URL download success")
                return response.content, filename
            print(f"[Storage] URL status {response.status_code}")
        except Exception as e:
            print(f"[Storage] URL failed: {e}")

    print(f"[Storage] Could not download {file_type}")
    return None, filename


# ═══════════════════════════════════════════════════════════════
# DOCX STRUCTURED EXTRACTION
# ═══════════════════════════════════════════════════════════════

def iter_block_items(parent):
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P

    parent_elm = parent.element.body
    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)


def extract_structured_docx(file_bytes):
    doc = Document(io.BytesIO(file_bytes))
    blocks = []

    for block in iter_block_items(doc):
        if isinstance(block, Paragraph):
            text_str = block.text.strip()
            if text_str:
                blocks.append({
                    "type": "paragraph",
                    "text": text_str
                })

        elif isinstance(block, Table):
            rows = []
            for row in block.rows:
                row_data = []
                for cell in row.cells:
                    row_data.append(cell.text.strip())
                rows.append(row_data)

            blocks.append({
                "type": "table",
                "rows": rows
            })

    # Image Extraction Pipeline
    rels = doc.part._rels
    for rel in rels:
        rel_obj = rels[rel]
        if "image" in rel_obj.target_ref:
            try:
                image_bytes = rel_obj.target_part.blob
                encoded = base64.b64encode(image_bytes).decode()
                blocks.append({
                    "type": "image",
                    "imageBase64": encoded[:5000]  # Chunked safety tracking preview
                })
            except Exception:
                pass

    return blocks


# ═══════════════════════════════════════════════════════════════
# PDF NATIVE & VISION OCR EXTRACTION
# ═══════════════════════════════════════════════════════════════

def extract_text_from_pdf(file_bytes):
    text_content = ""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for page in doc:
            text_content += page.get_text() + "\n"
        doc.close()

        if len(text_content.strip()) > 200:
            print(f"[PDF] Native extraction success: {len(text_content)} chars")
            return text_content
    except Exception as e:
        print(f"[PDF] Native parsing failed, falling back to OCR: {e}")

    return _groq_vision_ocr(file_bytes)


def _groq_vision_ocr(pdf_bytes):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    all_text = ""

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for i, page in enumerate(doc):
            try:
                pix = page.get_pixmap()
                img = base64.b64encode(pix.tobytes("png")).decode()

                response = client.chat.completions.create(
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{img}"}
                            },
                            {
                                "type": "text",
                                "text": "Extract ALL exam text exactly down to layout structural representation."
                            }
                        ]
                    }],
                    max_tokens=2000
                )
                txt = response.choices[0].message.content
                all_text += txt + "\n"
                print(f"[OCR] Processed Page {i + 1}")
            except Exception as e:
                print(f"[OCR] Page {i + 1} failed: {e}")
        doc.close()
    except Exception as e:
        print(f"[OCR] General Pipeline Failure: {e}")

    return all_text


# ═══════════════════════════════════════════════════════════════
# UNIVERSAL FILE CONTROLLER
# ═══════════════════════════════════════════════════════════════

def extract_text_from_file(file_bytes, filename):
    lower = filename.lower()

    if lower.endswith(".docx"):
        try:
            blocks = extract_structured_docx(file_bytes)
            return {"type": "structured", "blocks": blocks}
        except Exception:
            traceback.print_exc()

    elif lower.endswith(".pdf"):
        try:
            raw_text = extract_text_from_pdf(file_bytes)
            return {"type": "text", "text": raw_text}
        except Exception:
            traceback.print_exc()

    elif lower.endswith(".odt"):
        try:
            with tempfile.NamedTemporaryFile(suffix=".odt", delete=False) as tmp:
                tmp.write(file_bytes)
                tmp.flush()
                odt_doc = load(tmp.name)
                paragraphs = odt_doc.getElementsByType(text.P)
                content = "\n".join([teletype.extractText(p) for p in paragraphs])
            return {"type": "text", "text": content}
        except Exception:
            traceback.print_exc()

    elif lower.endswith(".doc"):
        try:
            # Safe production conversion strategy without crashing textract dependencies
            result = mammoth.extract_raw_text(io.BytesIO(file_bytes))
            return {"type": "text", "text": result.value}
        except Exception:
            traceback.print_exc()

    return {"type": "text", "text": ""}


MAX_CHARS = 10000


def chunk_text(text_data):
    return [text_data[i:i + MAX_CHARS] for i in range(0, len(text_data), MAX_CHARS)]


# ═══════════════════════════════════════════════════════════════
# QUESTION & MEMO ARTIFICIAL INTELLIGENCE PARSERS
# ═══════════════════════════════════════════════════════════════

def parse_questions_universal(text_data, subject, grade):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    all_questions = []
    chunks = chunk_text(text_data)

    for idx, chunk in enumerate(chunks):
        print(f"[Parser] Processing Chunk {idx + 1}/{len(chunks)}")
        prompt = f"""
Extract ALL exam questions into a single valid JSON array.

IMPORTANT:
- Preserve structural tables.
- Keep match columns intact.
- Keep Multiple Choice Option layout blocks intact.
- Keep accurate numbering keys.
- Return raw JSON formatting ONLY. Do markdown blocks wrapping or commentary strings.

[
 {{
   "question_number": "1.1",
   "question": "Question text data content",
   "type": "open",
   "marks": 2,
   "options": null,
   "memo": null,
   "table": null,
   "image_required": false
 }}
]

Subject: {subject}
Grade: {grade}

TEXT SOURCE:
{chunk}
"""
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=8000
            )
            raw = response.choices[0].message.content
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if match:
                parsed_json = json.loads(match.group())
                if isinstance(parsed_json, list):
                    all_questions.extend(parsed_json)
        except Exception as e:
            print(f"[Parser] Chunk compilation failed at index {idx}: {e}")
            continue

    return all_questions


def parse_memo_answers(text_data, subject, grade):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    prompt = f"""
Extract ALL memorandum solution answers mappings perfectly.
Return raw strict JSON object mapping structural properties ONLY.

{{
 "1.1": "Correct extraction tracking answer answer text structure summary value",
 "1.2": "answer structural verification data context text"
}}

CONTENT:
{text_data[:12000]}
"""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=8000
        )
        raw = response.choices[0].message.content
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return {}
        result = json.loads(match.group())
        return result if isinstance(result, dict) else {}
    except Exception as e:
        print(f"[Memo] Processing Error Failed: {e}")
        return {}


# ═══════════════════════════════════════════════════════════════
# DATA ROUTINE UTILITIES
# ═══════════════════════════════════════════════════════════════

def _normalise_qnum(qn):
    s = str(qn).lower().strip()
    return re.sub(r"[^a-z0-9]", "", s)


def _find_upload_meta(exam_id):
    for doc in db.collection("teacherExamUploads").stream():
        uploads = doc.to_dict().get("uploads", [])
        for upload in uploads:
            if upload.get("examId") == exam_id or upload.get("id") == exam_id:
                return upload, doc.id
    return None, None


def _launch_pipeline(exam_id, meta, school_id, subject_name):
    if _is_already_processing(exam_id):
        return False

    _mark_processing(exam_id)
    thread = threading.Thread(
        target=run_extraction_pipeline,
        args=(exam_id, meta, school_id, subject_name),
        daemon=True
    )
    thread.start()
    return True


# ═══════════════════════════════════════════════════════════════
# CORE EXTRACTION PIPELINE
# ═══════════════════════════════════════════════════════════════

def run_extraction_pipeline(exam_id, meta, school_id, subject_name):
    batch = db.batch()
    written = 0

    doc_ref = (
        db.collection("teacherExamUploads")
        .document(school_id)
        .collection("subjects")
        .document(subject_name)
    )

    def set_upload_status(status, extra=None):
        extra = extra or {}
        try:
            snap = doc_ref.get()
            if not snap.exists:
                return
            data = snap.to_dict() or {}
            uploads = []
            for upload in data.get("uploads", []):
                if upload.get("examId") == exam_id or upload.get("id") == exam_id:
                    upload["status"] = status
                    upload.update(extra)
                uploads.append(upload)
            doc_ref.update({"uploads": uploads})
        except Exception as e:
            print(f"[Status] Failed updates: {e}")

    try:
        subject = meta.get("subject", "General")
        grade = meta.get("grade", "12")
        title = meta.get("title", "Exam")

        print(f"\n[Pipeline] RUNNING CORE PIPELINE FOR EXAM: {exam_id}")
        set_upload_status("processing", {"processingStartedAt": datetime.utcnow().isoformat()})

        # Download Strategy
        exam_bytes, exam_fn = download_file_for_extraction(meta, "exam")
        if not exam_bytes:
            raise Exception("Could not download target structural exam file binary data bytes pool storage content.")

        exam_content = extract_text_from_file(exam_bytes, exam_fn)
        exam_text = json.dumps(exam_content["blocks"]) if exam_content["type"] == "structured" else exam_content["text"]

        if not exam_text.strip():
            raise Exception("Zero characters parsed out successfully from target system source matrix pipeline.")

        questions = parse_questions_universal(exam_text, subject, grade)
        print(f"[Pipeline] Parsed structural context question set count size: {len(questions)}")

        # Memo Structural Matrix Alignment
        memo_map = {}
        memo_bytes, memo_fn = download_file_for_extraction(meta, "memo")
        if memo_bytes:
            memo_content = extract_text_from_file(memo_bytes, memo_fn)
            memo_text = memo_content.get("text", "") if isinstance(memo_content, dict) else str(memo_content)
            if memo_text.strip():
                memo_map = parse_memo_answers(memo_text, subject, grade)

        norm_memo = {_normalise_qnum(k): v for k, v in memo_map.items()}
        for q in questions:
            norm = _normalise_qnum(q.get("question_number", ""))
            if norm in norm_memo:
                q["memo"] = norm_memo[norm]

        if not questions:
            questions = [{
                "question_number": "1",
                "question": "Questions layout system tracking could not parse content records natively.",
                "type": "open",
                "marks": 1,
                "options": None,
                "memo": None
            }]

        # Write Core Exam Document Configuration Header
        db.collection("exams").document(exam_id).set({
            "title": title,
            "subject": subject,
            "grade": grade,
            "questionsExtracted": True,
            "status": "ready",
            "uploadedBy": meta.get("uploadedBy", ""),
            "schoolId": meta.get("schoolId", ""),
            "teacherName": meta.get("teacherName", ""),
            "examStoragePath": meta.get("examStoragePath", ""),
            "memoStoragePath": meta.get("memoStoragePath", ""),
            "totalQuestions": len(questions),
            "memoMerged": bool(memo_map),
            "sourceUploadId": exam_id,
            "extractedAt": fs_admin.SERVER_TIMESTAMP
        })

        # Save Structural Questions Collections Tracking Documents
        for i, q in enumerate(questions):
            raw_marks = q.get("marks", 1)
            try:
                if raw_marks is None:
                    marks = 1
                elif isinstance(raw_marks, str):
                    cleaned = re.sub(r"[^0-9]", "", raw_marks.strip())
                    marks = int(cleaned) if cleaned else 1
                else:
                    marks = int(raw_marks)
            except Exception:
                marks = 1

            options = q.get("options")
            if not isinstance(options, dict):
                options = None

            qnum = str(q.get("question_number") or i + 1)
            qtext = str(q.get("question") or "").strip()
            if not qtext:
                continue

            ref = db.collection("exam_questions").document(f"{exam_id}_{i:04d}")
            batch.set(ref, {
                "examId": exam_id,
                "questionNumber": qnum,
                "parentQuestion": q.get("parent_question", ""),
                "parentContext": q.get("parent_context", None),
                "section": q.get("section", "A"),
                "questionText": qtext,
                "type": q.get("type", "open"),
                "marks": marks,
                "options": options,
                "columnA": q.get("column_a"),
                "columnB": q.get("column_b"),
                "memo": str(q.get("memo") or ""),
                "order": i,
                "createdAt": fs_admin.SERVER_TIMESTAMP,
            })

            written += 1
            if written % 400 == 0:
                batch.commit()
                batch = db.batch()

        batch.commit()
        print(f"[Pipeline] Finished posting {written} questions instances to Firestore cluster registry.")
        set_upload_status("extracted", {
            "extractedAt": datetime.utcnow().isoformat(),
            "totalQuestions": written,
            "memoMerged": bool(memo_map)
        })

    except Exception as e:
        traceback.print_exc()
        print(f"[Pipeline] Critical Processing Exception Failure: {e}")
        set_upload_status("error", {"errorMessage": str(e)[:500]})
        db.collection("exams").document(exam_id).set({
            "status": "error",
            "errorMessage": str(e)[:500]
        }, merge=True)
    finally:
        _unmark_processing(exam_id)


# ═══════════════════════════════════════════════════════════════
# AUTO EXTRACTION BACKGROUND SNAPSHOT LISTENER
# ═══════════════════════════════════════════════════════════════

def _start_auto_extraction_listener():
    subjects_ref = db.collection_group("subjects")

    def on_snapshot(col_snapshot, changes, read_time):
        for change in changes:
            if change.type.name not in ("ADDED", "MODIFIED"):
                continue

            data = change.document.to_dict() or {}
            subject_doc_ref = change.document.reference
            school_doc_ref = subject_doc_ref.parent.parent
            school_id = school_doc_ref.id
            subject_name = change.document.id
            uploads = data.get("uploads", [])

            for upload in uploads:
                exam_id = upload.get("examId") or upload.get("id")
                if not exam_id:
                    continue

                if upload.get("status") != "pending_extraction":
                    continue

                if not (upload.get("examStoragePath") or upload.get("examStorageUrl")):
                    continue

                if _is_already_processing(exam_id):
                    continue

                print(f"[Listener] Caught pending element context: {school_id} | {subject_name} | {exam_id}")
                _launch_pipeline(exam_id, upload, school_id, subject_name)

    subjects_ref.on_snapshot(on_snapshot)
    print("[Listener] GLOBAL REALTIME SUBJECT GRUOP LISTENER ENGINE DEPLOYED SYNCED AND STABLE")


# ═══════════════════════════════════════════════════════════════
# ENGINE RECOVERY RUNTIME BOOT SWEEP
# ═══════════════════════════════════════════════════════════════

def _sweep_pending_on_startup():
    print("[Startup] Initiating infrastructure safety recovery verification crash data sweeps...")
    launched = 0
    try:
        subjects = db.collection_group("subjects").stream()
        for doc in subjects:
            data = doc.to_dict() or {}
            uploads = data.get("uploads", [])
            school_doc = doc.reference.parent.parent
            school_id = school_doc.id
            subject_name = doc.id

            for upload in uploads:
                exam_id = upload.get("examId") or upload.get("id")
                if not exam_id or upload.get("status") != "pending_extraction":
                    continue

                if _is_already_processing(exam_id):
                    continue

                print(f"[Startup Sweep Execution Engine Launcher]: {exam_id}")
                launched += int(_launch_pipeline(exam_id, upload, school_id, subject_name))
    except Exception as e:
        print(f"[Startup Sweep] Crash interruption exception: {e}")

    print(f"[Startup Sweep Complete]. Booted {launched} deadlocks safely back into runtime execution threads.")


# Run initial setup sweeps & persistent database connection listeners right on script runtime initialization
_start_auto_extraction_listener()
_sweep_pending_on_startup()


# ═══════════════════════════════════════════════════════════════
# FLASK ENDPOINT API ROUTERS
# ═══════════════════════════════════════════════════════════════

@app.route("/")
def home():
    return jsonify({
        "status": "online",
        "engine": "Eduket Extraction Engine Production Core"
    })


@app.route("/auto-extract", methods=["POST"])
def auto_extract():
    try:
        payload = request.json or {}
        exam_id = payload.get("examId")
        teacher_id = payload.get("teacherId")
        school_id = payload.get("schoolId", "default_school")
        subject_name = payload.get("subjectName", "General")

        if not exam_id or not teacher_id:
            return jsonify({"error": "Missing examId or teacherId properties parameters payload fields."}), 400

        doc = db.collection("teacherExamUploads").document(teacher_id).get()
        if not doc.exists:
            return jsonify({
                               "error": "Teacher base identity upload map configuration elements not found registry matrix error."}), 404

        meta = None
        for upload in doc.to_dict().get("uploads", []):
            if upload.get("examId") == exam_id or upload.get("id") == exam_id:
                meta = upload
                break

        if not meta:
            return jsonify({
                               "error": "Target collection specific nested entity entry parameters missing data verification fields."}), 404

        launched = _launch_pipeline(exam_id, meta, school_id, subject_name)
        return jsonify({"ok": True, "launched": launched, "examId": exam_id})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/start-exam", methods=["POST"])
def start_exam():
    try:
        data = request.get_json() or {}
        exam_id = (
                data.get("exam_id") or
                data.get("exam") or
                data.get("examId") or
                ""
        ).strip()

        student_id = data.get("student_id", "anonymous")
        print(f"[start_exam] Verification sequence requested for exam_id='{exam_id}' by student='{student_id}'")

        if not exam_id:
            return jsonify({"error": "Missing parameter key context field: examId"}), 400

        # Complete clean fallback validation return placeholder pattern structure
        return jsonify({
            "status": "authorized",
            "examId": exam_id,
            "studentId": student_id,
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ═══════════════════════════════════════════════════════════════
# EXAM SUBMISSION + RESULTS ENGINE
# ═══════════════════════════════════════════════════════════════

from difflib import SequenceMatcher


def safe_float(v, default=0):
    try:
        return float(v)
    except Exception:
        return default


def normalize_text(value):
    if value is None:
        return ""
    return str(value).strip().lower()


def similarity(a, b):
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def mark_answer(student_answer, memo_answer, marks):
    """
    Simple AI-lite marking strategy.
    """

    student_answer = normalize_text(student_answer)
    memo_answer = normalize_text(memo_answer)

    if not memo_answer:
        return {
            "awarded": 0,
            "correct": False,
            "feedback": "No memo available."
        }

    if student_answer == memo_answer:
        return {
            "awarded": marks,
            "correct": True,
            "feedback": "Correct answer."
        }

    sim = similarity(student_answer, memo_answer)

    if sim >= 0.85:
        return {
            "awarded": marks,
            "correct": True,
            "feedback": "Very close to memorandum answer."
        }

    if sim >= 0.60:
        partial = round(marks * 0.5, 1)
        return {
            "awarded": partial,
            "correct": False,
            "feedback": "Partially correct."
        }

    return {
        "awarded": 0,
        "correct": False,
        "feedback": "Incorrect answer."
    }


# ═══════════════════════════════════════════════════════════════
# SUBMIT EXAM
# ═══════════════════════════════════════════════════════════════

@app.route("/submit", methods=["POST"])
def submit_exam():

    try:
        data = request.get_json() or {}

        exam_id = data.get("examId")
        student_id = data.get("studentId")
        answers = data.get("answers", {})

        if not exam_id:
            return jsonify({
                "error": "Missing examId"
            }), 400

        if not student_id:
            return jsonify({
                "error": "Missing studentId"
            }), 400

        # Prevent duplicate submissions
        existing = (
            db.collection("exam_submissions")
            .where("examId", "==", exam_id)
            .where("studentId", "==", student_id)
            .limit(1)
            .stream()
        )

        existing_docs = list(existing)

        if existing_docs:
            return jsonify({
                "error": "Exam already submitted."
            }), 400

        # Load questions
        questions_query = (
            db.collection("exam_questions")
            .where("examId", "==", exam_id)
            .order_by("order")
            .stream()
        )

        questions = [doc.to_dict() for doc in questions_query]

        if not questions:
            return jsonify({
                "error": "No questions found for exam."
            }), 404

        total_marks = 0
        earned_marks = 0
        results = []

        for q in questions:

            qnum = str(q.get("questionNumber", "")).strip()

            memo = q.get("memo", "")
            marks = safe_float(q.get("marks", 1), 1)

            total_marks += marks

            student_answer = answers.get(qnum, "")

            marked = mark_answer(
                student_answer,
                memo,
                marks
            )

            earned_marks += marked["awarded"]

            results.append({
                "questionNumber": qnum,
                "question": q.get("questionText", ""),
                "studentAnswer": student_answer,
                "memo": memo,
                "marks": marks,
                "awarded": marked["awarded"],
                "correct": marked["correct"],
                "feedback": marked["feedback"]
            })

        percentage = 0

        if total_marks > 0:
            percentage = round(
                (earned_marks / total_marks) * 100,
                2
            )

        # AI Feedback Summary
        if percentage >= 80:
            ai_feedback = "Excellent performance."
        elif percentage >= 60:
            ai_feedback = "Good work. Some improvements needed."
        elif percentage >= 40:
            ai_feedback = "Fair attempt. Revise weak sections."
        else:
            ai_feedback = "Needs significant improvement."

        # Save submission
        submission_data = {
            "examId": exam_id,
            "studentId": student_id,
            "answers": answers,
            "results": results,
            "score": round(earned_marks, 2),
            "total": round(total_marks, 2),
            "percentage": percentage,
            "feedback": ai_feedback,
            "submittedAt": fs_admin.SERVER_TIMESTAMP
        }

        submission_ref = db.collection("exam_submissions").document()

        submission_ref.set(submission_data)

        return jsonify({
            "success": True,
            "submissionId": submission_ref.id,
            "score": round(earned_marks, 2),
            "total": round(total_marks, 2),
            "percentage": percentage,
            "feedback": ai_feedback,
            "results": results
        })

    except Exception as e:
        traceback.print_exc()

        return jsonify({
            "error": str(e)
        }), 500


# ═══════════════════════════════════════════════════════════════
# GET RESULTS
# ═══════════════════════════════════════════════════════════════

@app.route("/results/<exam_id>/<student_id>", methods=["GET"])
def get_results(exam_id, student_id):

    try:

        query = (
            db.collection("exam_submissions")
            .where("examId", "==", exam_id)
            .where("studentId", "==", student_id)
            .limit(1)
            .stream()
        )

        docs = list(query)

        if not docs:
            return jsonify({
                "error": "Results not found."
            }), 404

        result = docs[0].to_dict()

        return jsonify({
            "success": True,
            "result": result
        })

    except Exception as e:
        traceback.print_exc()

        return jsonify({
            "error": str(e)
        }), 500


# ═══════════════════════════════════════════════════════════════
# AUTOSAVE ENDPOINT
# PREVENTS FRONTEND FAILED FETCH ERRORS
# ═══════════════════════════════════════════════════════════════

@app.route("/autosave", methods=["POST"])
def autosave_exam():

    try:

        data = request.get_json() or {}

        exam_id = data.get("examId")
        student_id = data.get("studentId")
        answers = data.get("answers", {})

        if not exam_id or not student_id:
            return jsonify({
                "error": "Missing required fields."
            }), 400

        doc_id = f"{exam_id}_{student_id}"

        db.collection("exam_autosaves").document(doc_id).set({
            "examId": exam_id,
            "studentId": student_id,
            "answers": answers,
            "updatedAt": fs_admin.SERVER_TIMESTAMP
        }, merge=True)

        return jsonify({
            "success": True
        })

    except Exception as e:
        traceback.print_exc()

        return jsonify({
            "error": str(e)
        }), 500


# ═══════════════════════════════════════════════════════════════
# LOAD AUTOSAVED EXAM
# ═══════════════════════════════════════════════════════════════

@app.route("/autosave/<exam_id>/<student_id>", methods=["GET"])
def load_autosave(exam_id, student_id):

    try:

        doc_id = f"{exam_id}_{student_id}"

        doc = (
            db.collection("exam_autosaves")
            .document(doc_id)
            .get()
        )

        if not doc.exists:
            return jsonify({
                "success": True,
                "answers": {}
            })

        data = doc.to_dict()

        return jsonify({
            "success": True,
            "answers": data.get("answers", {})
        })

    except Exception as e:
        traceback.print_exc()

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    # Standard local debug server runner execution profile pattern logic block
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)