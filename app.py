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

FLOW
──────────────────────────────────────────────────────────────
Teacher Upload
    ↓
Firebase Storage
    ↓
teacherExamUploads updated
    ↓
Firestore listener detects upload
    ↓
Extraction pipeline starts
    ↓
Questions parsed + memo merged
    ↓
exams/{examId} => status=ready
    ↓
Students can immediately write exam
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

from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

# ═══════════════════════════════════════════════════════════════
# FIREBASE
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
        "storageBucket": os.getenv(
            "FIREBASE_STORAGE_BUCKET"
        )
    })

_init_firebase()

db = fs_admin.client()
bucket = storage.bucket()

# ═══════════════════════════════════════════════════════════════
# APP
# ═══════════════════════════════════════════════════════════════

app = Flask(__name__)

CORS(app, resources={
    r"/*": {
        "origins": [
            "http://localhost:3000",
            "http://localhost:5173",
            "http://localhost:5174",
            "http://localhost:5175",
            "http://localhost:5176",
            "https://eduket.netlify.app",
            "https://*.netlify.app",
        ]
    }
})

# ═══════════════════════════════════════════════════════════════
# PROCESS TRACKING
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

    storage_path = meta.get(
        f"{file_type}StoragePath"
    )

    if storage_path:

        try:
            blob = bucket.blob(storage_path)

            if blob.exists():

                data = blob.download_as_bytes(
                    timeout=120
                )

                print(
                    f"[Storage] Path download success "
                    f"{storage_path}"
                )

                return data, filename

        except Exception as e:
            print(f"[Storage] Path failed: {e}")

    # ── FALLBACK TO URL DOWNLOAD ───────────────────────────

    storage_url = meta.get(
        f"{file_type}StorageUrl"
    )

    if storage_url:

        try:
            import requests

            response = requests.get(
                storage_url,
                timeout=120
            )

            if response.status_code == 200:

                print(
                    f"[Storage] URL download success"
                )

                return response.content, filename

            print(
                f"[Storage] URL status "
                f"{response.status_code}"
            )

        except Exception as e:
            print(f"[Storage] URL failed: {e}")

    print(f"[Storage] Could not download {file_type}")

    return None, filename

# ═══════════════════════════════════════════════════════════════
# DOCX EXTRACTION
# ═══════════════════════════════════════════════════════════════

def extract_text_from_docx(file_bytes):

    try:
        from docx import Document

        doc = Document(io.BytesIO(file_bytes))

        lines = []

        for para in doc.paragraphs:
            text = para.text.strip()

            if text:
                lines.append(text)

        for table in doc.tables:
            for row in table.rows:
                vals = []

                for cell in row.cells:
                    txt = cell.text.strip()

                    if txt:
                        vals.append(txt)

                if vals:
                    lines.append(" | ".join(vals))

        text = "\n".join(lines)

        print(f"[DOCX] {len(text)} chars")

        return text

    except Exception:
        traceback.print_exc()
        return ""

# ═══════════════════════════════════════════════════════════════
# PDF EXTRACTION
# ═══════════════════════════════════════════════════════════════

def extract_text_from_pdf(file_bytes):

    text = ""

    # ── Native text first ─────────────────────────────────────

    try:
        import fitz

        doc = fitz.open(
            stream=file_bytes,
            filetype="pdf"
        )

        for page in doc:
            text += page.get_text() + "\n"

        doc.close()

        if len(text.strip()) > 200:
            print(f"[PDF] Native: {len(text)} chars")
            return text

    except Exception as e:
        print(f"[PDF] Native failed: {e}")

    # ── OCR fallback ──────────────────────────────────────────

    return _groq_vision_ocr(file_bytes)

# ═══════════════════════════════════════════════════════════════
# OCR
# ═══════════════════════════════════════════════════════════════

def _groq_vision_ocr(pdf_bytes):

    from groq import Groq
    import fitz

    client = Groq(
        api_key=os.getenv("GROQ_API_KEY")
    )

    all_text = ""

    try:
        doc = fitz.open(
            stream=pdf_bytes,
            filetype="pdf"
        )

        for i, page in enumerate(doc):

            try:
                pix = page.get_pixmap()

                img = base64.b64encode(
                    pix.tobytes("png")
                ).decode()

                response = client.chat.completions.create(
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img}"
                                }
                            },
                            {
                                "type": "text",
                                "text": (
                                    "Extract ALL exam text exactly."
                                )
                            }
                        ]
                    }],
                    max_tokens=2000
                )

                txt = response.choices[0].message.content

                all_text += txt + "\n"

                print(f"[OCR] Page {i+1}")

            except Exception as e:
                print(f"[OCR] Page failed: {e}")

        doc.close()

    except Exception as e:
        print(f"[OCR] Failed: {e}")

    return all_text

# ═══════════════════════════════════════════════════════════════
# FILE EXTRACTION
# ═══════════════════════════════════════════════════════════════

def extract_text_from_file(file_bytes, filename):

    lower = filename.lower()

    if lower.endswith(".docx") or lower.endswith(".doc"):
        return extract_text_from_docx(file_bytes)

    if lower.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)

    text = extract_text_from_docx(file_bytes)

    if text.strip():
        return text

    return extract_text_from_pdf(file_bytes)

# ═══════════════════════════════════════════════════════════════
# QUESTION PARSER
# ═══════════════════════════════════════════════════════════════

def parse_questions_universal(text, subject, grade):

    from groq import Groq

    client = Groq(
        api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = f"""
Extract ALL exam questions into JSON array.

Return ONLY JSON.

[
 {{
   "question_number":"1.1",
   "question":"Question text",
   "type":"open",
   "marks":2,
   "options":null,
   "memo":null
 }}
]

Subject: {subject}
Grade: {grade}

TEXT:
{text[:12000]}
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{
                "role": "user",
                "content": prompt
            }],
            temperature=0,
            max_tokens=8000
        )

        raw = response.choices[0].message.content

        match = re.search(
            r"\[.*\]",
            raw,
            re.DOTALL
        )

        if not match:
            return []

        arr = json.loads(match.group())

        if not isinstance(arr, list):
            return []

        return arr

    except Exception as e:
        print(f"[Parser] Failed: {e}")
        return []

# ═══════════════════════════════════════════════════════════════
# MEMO PARSER
# ═══════════════════════════════════════════════════════════════

def parse_memo_answers(text, subject, grade):

    from groq import Groq

    client = Groq(
        api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = f"""
Extract ALL memo answers.

Return ONLY JSON object.

{{
 "1.1":"answer",
 "1.2":"answer"
}}

{text[:12000]}
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{
                "role": "user",
                "content": prompt
            }],
            temperature=0,
            max_tokens=8000
        )

        raw = response.choices[0].message.content

        match = re.search(
            r"\{.*\}",
            raw,
            re.DOTALL
        )

        if not match:
            return {}

        result = json.loads(match.group())

        if not isinstance(result, dict):
            return {}

        return result

    except Exception as e:
        print(f"[Memo] Failed: {e}")
        return {}

# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def _normalise_qnum(qn):

    s = str(qn).lower().strip()

    s = re.sub(r"[^a-z0-9]", "", s)

    return s

def _find_upload_meta(exam_id):

    for doc in db.collection(
        "teacherExamUploads"
    ).stream():

        uploads = doc.to_dict().get(
            "uploads",
            []
        )

        for upload in uploads:

            if (
                upload.get("examId") == exam_id
                or upload.get("id") == exam_id
            ):
                return upload, doc.id

    return None, None

# ═══════════════════════════════════════════════════════════════
# PIPELINE LAUNCHER
# ═══════════════════════════════════════════════════════════════

def _launch_pipeline(
    exam_id,
    meta,
    teacher_doc_id
):

    if _is_already_processing(exam_id):
        return False

    _mark_processing(exam_id)

    thread = threading.Thread(
        target=run_extraction_pipeline,
        args=(
            exam_id,
            meta,
            teacher_doc_id
        ),
        daemon=True
    )

    thread.start()

    return True

# ═══════════════════════════════════════════════════════════════
# EXTRACTION PIPELINE
# ═══════════════════════════════════════════════════════════════
def safe_int(value, default=1):
    try:
        if value in [None, "", "null"]:
            return default
        return int(float(value))
    except Exception:
        return default

def run_extraction_pipeline(
    exam_id,
    meta,
    teacher_doc_id
):
    batch = db.batch()
    written = 0

    doc_ref = db.collection(
        "teacherExamUploads"
    ).document(teacher_doc_id)

    def set_upload_status(status, extra=None):

        extra = extra or {}

        try:
            snap = doc_ref.get()

            if not snap.exists:
                return

            data = snap.to_dict() or {}

            uploads = []

            for upload in data.get("uploads", []):

                if (
                    upload.get("examId") == exam_id
                    or upload.get("id") == exam_id
                ):
                    upload["status"] = status
                    upload.update(extra)

                uploads.append(upload)

            doc_ref.update({
                "uploads": uploads
            })

        except Exception as e:
            print(f"[Status] Failed: {e}")

    try:

        subject = meta.get("subject", "General")
        grade = meta.get("grade", "12")
        title = meta.get("title", "Exam")

        print(f"\n[Pipeline] START {exam_id}")

        set_upload_status(
            "processing",
            {
                "processingStartedAt":
                datetime.utcnow().isoformat()
            }
        )

        # ── Download exam ───────────────────────────────────

        exam_bytes, exam_fn = download_file_for_extraction(
            meta,
            "exam"
        )

        if not exam_bytes:
            raise Exception(
                "Could not download exam file"
            )

        # ── Extract text ────────────────────────────────────

        exam_text = extract_text_from_file(
            exam_bytes,
            exam_fn
        )

        if not exam_text.strip():
            raise Exception(
                "No text extracted"
            )

        # ── Parse questions ─────────────────────────────────

        questions = parse_questions_universal(
            exam_text,
            subject,
            grade
        )

        print(f"[Pipeline] {len(questions)} questions")

        # ── Memo ────────────────────────────────────────────

        memo_map = {}

        memo_bytes, memo_fn = download_file_for_extraction(
            meta,
            "memo"
        )

        if memo_bytes:

            memo_text = extract_text_from_file(
                memo_bytes,
                memo_fn
            )

            if memo_text.strip():

                memo_map = parse_memo_answers(
                    memo_text,
                    subject,
                    grade
                )

        # ── Merge memos ─────────────────────────────────────

        norm_memo = {
            _normalise_qnum(k): v
            for k, v in memo_map.items()
        }

        for q in questions:

            norm = _normalise_qnum(
                q.get("question_number", "")
            )

            if norm in norm_memo:
                q["memo"] = norm_memo[norm]

        # ── Fallback question ───────────────────────────────

        if not questions:

            questions = [{
                "question_number": "1",
                "question":
                    "Questions could not be parsed.",
                "type": "open",
                "marks": 0,
                "options": None,
                "memo": None
            }]

        # ── Save exam ───────────────────────────────────────

        db.collection("exams").document(
            exam_id
        ).set({

            "title": title,
            "subject": subject,
            "grade": grade,

            "questionsExtracted": written > 0,

            "status": "ready",

            "uploadedBy":
                meta.get("uploadedBy", ""),

            "schoolId":
                meta.get("schoolId", ""),

            "teacherName":
                meta.get("teacherName", ""),

            "examStoragePath":
                meta.get("examStoragePath", ""),

            "memoStoragePath":
                meta.get("memoStoragePath", ""),

            "totalQuestions":
                len(questions),

            "memoMerged":
                bool(memo_map),

            "sourceUploadId":
                exam_id,

            "extractedAt":
                fs_admin.SERVER_TIMESTAMP
        })

        # ── Save questions ──────────────────────────────────
        # ─────────────────────────────────────────────
        # WRITE QUESTIONS (PRODUCTION SAFE)
        # ─────────────────────────────────────────────


        for i, q in enumerate(questions):

            # ── SAFE MARKS HANDLING ──────────────────
            raw_marks = q.get("marks", 1)

            try:
                if raw_marks is None:
                    marks = 1
                elif isinstance(raw_marks, str):
                    cleaned = raw_marks.strip()

                    # Handles "(2)" or "2 marks"
                    cleaned = re.sub(r"[^0-9]", "", cleaned)

                    marks = int(cleaned) if cleaned else 1
                else:
                    marks = int(raw_marks)

            except Exception:
                marks = 1

            # ── SAFE OPTIONS ─────────────────────────
            options = q.get("options")

            if not isinstance(options, dict):
                options = None

            # ── SAFE QUESTION NUMBER ─────────────────
            qnum = str(
                q.get("question_number")
                or i + 1
            )

            # ── SAFE QUESTION TEXT ───────────────────
            qtext = str(
                q.get("question")
                or ""
            ).strip()

            if not qtext:
                continue

            # ── FIRESTORE DOC ────────────────────────
            ref = db.collection(
                "exam_questions"
            ).document(
                f"{exam_id}_{i:04d}"
            )

            batch.set(ref, {

                "examId":
                    exam_id,

                "questionNumber":
                    qnum,

                "parentQuestion":
                    q.get(
                        "parent_question",
                        ""
                    ),

                "parentContext":
                    q.get(
                        "parent_context",
                        None
                    ),

                "section":
                    q.get(
                        "section",
                        "A"
                    ),

                "questionText":
                    qtext,

                "type":
                    q.get(
                        "type",
                        "open"
                    ),

                "marks":
                    marks,

                "options":
                    options,

                "columnA":
                    q.get("column_a"),

                "columnB":
                    q.get("column_b"),

                "memo":
                    str(
                        q.get("memo") or ""
                    ),

                "order":
                    i,

                "createdAt":
                    fs_admin.SERVER_TIMESTAMP,
            })

            written += 1

            # Firestore batch limit safety
            if written % 400 == 0:
                batch.commit()
                batch = db.batch()

        # Final commit
        batch.commit()

        print(
            f"[Pipeline] Successfully wrote "
            f"{written} questions"
        )

        # ── Complete ────────────────────────────────────────

        set_upload_status(
            "extracted",
            {
                "extractedAt":
                    datetime.utcnow().isoformat(),

                "totalQuestions": written,

                "memoMerged":
                    bool(memo_map)
            }
        )

        print(f"[Pipeline] COMPLETE {exam_id}")

    except Exception as e:

        traceback.print_exc()

        print(f"[Pipeline] FAILED {exam_id}: {e}")

        set_upload_status(
            "error",
            {
                "errorMessage": str(e)[:500]
            }
        )

        db.collection("exams").document(
            exam_id
        ).set({
            "status": "error",
            "errorMessage": str(e)[:500]
        }, merge=True)

    finally:
        _unmark_processing(exam_id)

# ═══════════════════════════════════════════════════════════════
# AUTO LISTENER
# ═══════════════════════════════════════════════════════════════

def _start_auto_extraction_listener():

    def on_snapshot(
        col_snapshot,
        changes,
        read_time
    ):

        for change in changes:

            if change.type.name not in (
                "ADDED",
                "MODIFIED"
            ):
                continue

            data = change.document.to_dict() or {}

            teacher_doc_id = change.document.id

            for upload in data.get("uploads", []):

                exam_id = (
                    upload.get("examId")
                    or upload.get("id")
                )

                if not exam_id:
                    continue

                if upload.get("status") != "pending_extraction":
                    continue

                if not (
                        upload.get("examStoragePath")
                        or upload.get("examStorageUrl")
                ):
                    continue

                if _is_already_processing(exam_id):
                    continue

                print(f"[Listener] {exam_id}")

                _launch_pipeline(
                    exam_id,
                    upload,
                    teacher_doc_id
                )

    db.collection(
        "teacherExamUploads"
    ).on_snapshot(on_snapshot)

    print("[Listener] ACTIVE")

# ═══════════════════════════════════════════════════════════════
# STARTUP SWEEP
# ═══════════════════════════════════════════════════════════════

def _sweep_pending_on_startup():

    print("[Startup] Sweep starting")

    launched = 0

    try:

        for doc in db.collection(
            "teacherExamUploads"
        ).stream():

            teacher_doc_id = doc.id

            uploads = doc.to_dict().get(
                "uploads",
                []
            )

            for upload in uploads:

                exam_id = (
                    upload.get("examId")
                    or upload.get("id")
                )

                if not exam_id:
                    continue

                if upload.get("status") != "pending_extraction":
                    continue

                if not upload.get("examStoragePath"):
                    continue

                if _is_already_processing(exam_id):
                    continue

                print(f"[Startup] {exam_id}")

                if _launch_pipeline(
                    exam_id,
                    upload,
                    teacher_doc_id
                ):
                    launched += 1

    except Exception as e:
        print(f"[Startup] Failed: {e}")

    print(f"[Startup] {launched} launched")

# ═══════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════

@app.route("/")
def home():

    return jsonify({
        "status": "online",
        "engine": "Eduket Extraction Engine"
    })

@app.route("/auto-extract", methods=["POST"])
def auto_extract():

    try:

        payload = request.json or {}

        exam_id = payload.get("examId")
        teacher_id = payload.get("teacherId")

        if not exam_id or not teacher_id:
            return jsonify({
                "error":
                    "Missing examId or teacherId"
            }), 400

        doc = db.collection(
            "teacherExamUploads"
        ).document(
            teacher_id
        ).get()

        if not doc.exists:
            return jsonify({
                "error":
                    "Teacher uploads not found"
            }), 404

        meta = None

        for upload in doc.to_dict().get(
            "uploads",
            []
        ):

            if (
                upload.get("examId") == exam_id
                or upload.get("id") == exam_id
            ):
                meta = upload
                break

        if not meta:
            return jsonify({
                "error":
                    "Upload metadata missing"
            }), 404

        launched = _launch_pipeline(
            exam_id,
            meta,
            teacher_id
        )

        return jsonify({
            "ok": True,
            "launched": launched,
            "examId": exam_id
        })

    except Exception as e:
        traceback.print_exc()

        return jsonify({
            "error": str(e)
        }), 500

# ═══════════════════════════════════════════════════════════════
# START SERVICES
# ═══════════════════════════════════════════════════════════════

try:
    _start_auto_extraction_listener()
    _sweep_pending_on_startup()
except Exception as e:
    print(f"[Startup] Services failed: {e}")

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 5000)
    )

    print(f"\n🚀 Eduket running on {port}")

    app.run(
        host="0.0.0.0",
        port=port,
        debug=True
    )