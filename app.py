"""
app.py — Eduket Production Exam Extraction & Marking API
═══════════════════════════════════════════════════════════════
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
import uuid

import requests as http_requests
import fitz  # PyMuPDF

from datetime import datetime, timezone
from difflib import SequenceMatcher

from flask import Flask, request, jsonify
from flask_cors import CORS

from extraction_engine import (
    extract_text_from_file,
    parse_questions_universal,
    extract_questions_from_file,
)

from groq import Groq
import firebase_admin
from firebase_admin import credentials, firestore as fs_admin, storage, auth as fb_auth


# ═══════════════════════════════════════════════════════════════
# FIREBASE INITIALIZATION
# ═══════════════════════════════════════════════════════════════

db     = None
bucket = None


def _init_firebase():
    global db, bucket

    raw = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON", "").strip()
    if not raw:
        raise ValueError("FIREBASE_SERVICE_ACCOUNT_JSON is not set")

    cred_dict = json.load(open(raw)) if os.path.exists(raw) else json.loads(raw)

    if "private_key" in cred_dict:
        cred_dict["private_key"] = cred_dict["private_key"].replace("\\n", "\n")

    missing = [k for k in ["type", "project_id", "private_key", "client_email"]
               if not cred_dict.get(k)]
    if missing:
        raise ValueError(f"Credential dict missing: {missing}")

    print(f"[Firebase] project_id: {cred_dict['project_id']}")
    print(f"[Firebase] client_email: {cred_dict['client_email']}")

    if not firebase_admin._apps:
        firebase_admin.initialize_app(
            credentials.Certificate(cred_dict),
            {"storageBucket": os.environ.get("FIREBASE_STORAGE_BUCKET")},
        )

    db     = fs_admin.client()
    bucket = storage.bucket()
    print("[Firebase] ✅ Ready")


def verify_request_token(req):
    auth_header = req.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return None, (jsonify({"error": "Missing or malformed Authorization header"}), 401)
    id_token = auth_header.split("Bearer ", 1)[1].strip()
    try:
        decoded = fb_auth.verify_id_token(id_token)
        return decoded["uid"], None
    except Exception as e:
        print(f"[verify_request_token] {e}")
        return None, (jsonify({"error": "Invalid or expired token"}), 401)


# ═══════════════════════════════════════════════════════════════
# TIER LIMITS
# ═══════════════════════════════════════════════════════════════

TIER_EXAM_LIMITS = {
    "free":     5,
    "silver":   30,
    "gold":     120,
    "platinum": 500,
    "diamond":  1000,
}


def get_exam_limit(tier_id):
    return TIER_EXAM_LIMITS.get(tier_id, TIER_EXAM_LIMITS["free"])


# ═══════════════════════════════════════════════════════════════
# FLASK APP + CORS
# Note: @app.before_request OPTIONS handler removed — Flask-CORS
# handles preflights. Having both caused CORS header conflicts.
# ═══════════════════════════════════════════════════════════════

app = Flask(__name__)

CORS(app, resources={r"/*": {
    "origins": [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
        "http://localhost:5176",
        "http://localhost:5177",
        "https://eduket.netlify.app",
        "https://eduket.tech",
    ],
    "methods":      ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    "allow_headers": ["Content-Type", "Authorization"],
}}, supports_credentials=False)

from billing_routes import billing_bp
app.register_blueprint(billing_bp)


# ═══════════════════════════════════════════════════════════════
# THREAD-SAFE PROCESSING TRACKER
# ═══════════════════════════════════════════════════════════════

_PROCESSING      = set()
_PROCESSING_LOCK = threading.Lock()


def _is_already_processing(exam_id):
    with _PROCESSING_LOCK:
        return exam_id in _PROCESSING


def _mark_processing(exam_id):
    with _PROCESSING_LOCK:
        _PROCESSING.add(exam_id)


def _unmark_processing(exam_id):
    with _PROCESSING_LOCK:
        _PROCESSING.discard(exam_id)


# ═══════════════════════════════════════════════════════════════
# FIREBASE STORAGE DOWNLOAD
# ═══════════════════════════════════════════════════════════════

def download_file_for_extraction(meta, file_type):
    filename     = meta.get(f"{file_type}FileName", f"{file_type}.pdf")
    storage_path = meta.get(f"{file_type}StoragePath")

    if storage_path:
        try:
            blob = bucket.blob(storage_path)
            if blob.exists():
                data = blob.download_as_bytes(timeout=120)
                print(f"[Storage] SDK OK: {storage_path} ({len(data)} bytes)")
                return data, filename
        except Exception as e:
            print(f"[Storage] SDK failed: {e}")

    storage_url = meta.get(f"{file_type}StorageUrl")
    if storage_url:
        try:
            res = http_requests.get(storage_url, timeout=120)
            if res.status_code == 200:
                print(f"[Storage] URL OK ({len(res.content)} bytes)")
                return res.content, filename
            print(f"[Storage] URL returned {res.status_code}")
        except Exception as e:
            print(f"[Storage] URL failed: {e}")

    print(f"[Storage] No source for {file_type}")
    return None, filename


# ═══════════════════════════════════════════════════════════════
# MEMO PARSER
# ═══════════════════════════════════════════════════════════════

def parse_memo_answers(memo_text, subject, grade):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    CHUNK  = 12000
    result = {}

    for idx, chunk in enumerate([memo_text[i:i+CHUNK]
                                  for i in range(0, len(memo_text), CHUNK)]):
        print(f"[Memo] Chunk {idx+1}")
        try:
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": f"""You are reading a South African CAPS/NSC exam MARKING MEMORANDUM.
Extract EVERY answer. Return ONLY a valid JSON object mapping question_number to answer.
For MCQ give just the letter. For True/False give "True" or "False".
No markdown, no explanation.
Example: {{"1.1": "C", "1.2": "True", "1.3": "RAM is volatile memory."}}
Subject: {subject} | Grade: {grade}
MEMO TEXT:
{chunk}"""}],
                temperature=0, max_tokens=8000,
            )
            raw   = resp.choices[0].message.content.strip()
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                chunk_result = json.loads(match.group())
                if isinstance(chunk_result, dict):
                    for k, v in chunk_result.items():
                        norm = _normalise_qnum(k)
                        if norm and norm not in result:
                            result[norm] = v
        except Exception as e:
            print(f"[Memo] Chunk {idx+1} failed: {e}")

    print(f"[Memo] Total answers: {len(result)}")
    return result


# ═══════════════════════════════════════════════════════════════
# MARKING ENGINE
# ═══════════════════════════════════════════════════════════════

def _normalise_text(v):
    return "" if v is None else str(v).strip().lower()


def _normalise_qnum(qn):
    s = str(qn).lower().strip()
    s = re.sub(r"^(question|q|ques|no|nr)[\s.\-]*", "", s)
    s = re.sub(r"[^a-z0-9]", "", s)
    return s


def _similarity(a, b):
    return SequenceMatcher(None, _normalise_text(a), _normalise_text(b)).ratio()


def mark_with_memo(student_answer, memo_answer, marks):
    s_norm = _normalise_text(student_answer)
    m_norm = _normalise_text(memo_answer)

    if not s_norm:
        return {"score": 0, "status": "missing",
                "feedback": "No answer provided.", "concept_gap": "Question not attempted."}

    if not m_norm:
        return None  # signal AI fallback

    if s_norm == m_norm:
        return {"score": marks, "status": "correct", "feedback": "Correct.", "concept_gap": ""}

    if len(m_norm) == 1 and m_norm.isalpha():
        if s_norm.startswith(m_norm):
            return {"score": marks, "status": "correct",
                    "feedback": "Correct option selected.", "concept_gap": ""}
        return {"score": 0, "status": "incorrect",
                "feedback": f"Incorrect. Correct answer: {memo_answer.upper()}.",
                "concept_gap": "Wrong option selected."}

    if m_norm in ("true", "false"):
        if s_norm.startswith(m_norm):
            return {"score": marks, "status": "correct", "feedback": "Correct.", "concept_gap": ""}
        return {"score": 0, "status": "incorrect",
                "feedback": f"Incorrect. Answer is {memo_answer}.",
                "concept_gap": "True/False answer incorrect."}

    sim = _similarity(s_norm, m_norm)
    if sim >= 0.75:
        return {"score": marks, "status": "correct", "feedback": "Correct.", "concept_gap": ""}
    return None  # AI fallback


def mark_with_ai(question, student_answer, marks, subject, memo=""):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    prompt = f"""You are a senior South African CAPS/NSC examiner for {subject}.
Mark fairly based on CONCEPTUAL UNDERSTANDING, not perfect wording.
IGNORE spelling mistakes and grammatical errors — focus on whether the student understands the concept.

QUESTION: {question}
MARKS AVAILABLE: {marks}
MEMO/EXPECTED ANSWER: {memo if memo else "Use your " + subject + " curriculum expertise"}
STUDENT ANSWER: {student_answer if student_answer.strip() else "No answer provided"}

Return ONLY this exact JSON:
{{
  "score": <number between 0 and {marks}>,
  "status": "<correct|partial|incorrect|missing>",
  "feedback": "<specific feedback>",
  "concept_gap": "<concept missed, or empty string if correct>",
  "model_answer": "<ideal answer>"
}}"""
    try:
        resp  = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=800,
        )
        raw   = re.sub(r"^```json\s*|^```\s*|```$", "",
                       resp.choices[0].message.content.strip(),
                       flags=re.MULTILINE).strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            result        = json.loads(match.group())
            result["score"] = max(0, min(float(result.get("score", 0)), marks))
            return result
    except Exception as e:
        print(f"[AI Mark] Failed: {e}", flush=True)

    return {"score": 0, "status": "incorrect",
            "feedback": "Could not mark — AI unavailable.",
            "concept_gap": "Unknown.", "model_answer": ""}


def generate_final_feedback(percentage, results, subject):
    wrong   = [r for r in results if r.get("status") in ("incorrect", "missing")]
    partial = [r for r in results if r.get("status") == "partial"]
    gaps    = list({r.get("concept_gap", "")
                    for r in results if r.get("concept_gap", "").strip()})

    if percentage >= 80:
        tone = f"Excellent work! Strong command of {subject}."
    elif percentage >= 60:
        tone = f"Good effort. A solid attempt at {subject}."
    elif percentage >= 40:
        tone = f"Average performance. More revision of {subject} is needed."
    else:
        tone = f"Below average. Serious revision of {subject} is required."

    lines = [tone]
    if wrong:
        lines.append(f"Questions needing attention: "
                     f"{', '.join(str(r.get('question_number','?')) for r in wrong[:8])}.")
    if partial:
        lines.append(f"Partially correct: "
                     f"{', '.join(str(r.get('question_number','?')) for r in partial[:5])} — "
                     f"expand your answers.")
    lines.append(f"Key concept gaps: {'; '.join(gaps[:5]) if gaps else 'None identified'}.")
    return " ".join(lines)


def generate_exam_analysis(subject, percentage, total_score, total_marks, results):
    client  = Groq(api_key=os.getenv("GROQ_API_KEY"))
    payload = [{"question": r.get("question", ""),
                "student_answer": r.get("student_answer", ""),
                "correct_answer": r.get("correct_answer", ""),
                "status": r.get("status", ""),
                "marks": r.get("marks", 0),
                "earned": r.get("earned", 0),
                "feedback": r.get("feedback", "")}
               for r in results]

    prompt = f"""You are an expert teacher and learning analyst.
Analyse the student's performance in {subject}. Identify conceptual strengths and weaknesses.
Student scored: {total_score}/{total_marks} ({percentage}%)
Return ONLY valid JSON:
{{
  "overallSummary":"","studentProfile":"",
  "strengths":[],"weaknesses":[],"misconceptions":[],
  "learningStyle":"",
  "cognitiveAnalysis":{{"remember":0,"understand":0,"apply":0,"analyse":0,"evaluate":0,"create":0}},
  "studyPlan":[],"teacherSummary":"","parentSummary":""
}}
Exam data:
{json.dumps(payload, indent=2)}"""

    try:
        resp  = client.chat.completions.create(
            model="qwen/qwen3-32b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2, max_tokens=2500,
        )
        raw   = re.sub(r"^```json\s*|^```\s*|```$", "",
                       resp.choices[0].message.content.strip(),
                       flags=re.MULTILINE).strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception as e:
        print(f"[Analysis] {e}")
    return {}


# ═══════════════════════════════════════════════════════════════
# EXTRACTION PIPELINE
# ═══════════════════════════════════════════════════════════════

def _get_subject_doc_ref(school_id, subject_name):
    return (db.collection("teacherExamUploads")
              .document(school_id)
              .collection("subjects")
              .document(subject_name))


def run_extraction_pipeline(exam_id, meta, school_id, subject_name):
    subject_ref = _get_subject_doc_ref(school_id, subject_name)

    def set_upload_status(status, extra={}):
        try:
            snap = subject_ref.get()
            if not snap.exists:
                return
            uploads = []
            for u in (snap.to_dict() or {}).get("uploads", []):
                if u.get("examId") == exam_id or u.get("id") == exam_id:
                    u["status"] = status
                    u.update(extra)
                uploads.append(u)
            subject_ref.update({"uploads": uploads})
        except Exception as e:
            print(f"[Status] Update failed: {e}")

    try:
        subject = meta.get("subject", subject_name or "General")
        grade   = meta.get("grade",   "12")
        title   = meta.get("title",   "Exam")

        print(f"\n[Pipeline] ═══ {exam_id} | {subject} Gr{grade}")
        set_upload_status("processing",
                          {"processingStartedAt": datetime.utcnow().isoformat()})

        # 1. Download exam file
        exam_bytes, exam_fn = download_file_for_extraction(meta, "exam")
        if not exam_bytes:
            raise ValueError("Could not download exam file.")

        # 2. Extract questions (render-first pipeline via extraction_engine)
        questions = extract_questions_from_file(
            exam_bytes, exam_fn, subject, grade,
            exam_id=exam_id,
            school_folder=meta.get("schoolFolder", school_id),
        )
        print(f"[Pipeline] Questions: {len(questions)}")
        if not questions:
            raise ValueError("No questions could be extracted. Check the file is a valid exam paper.")

        # 3. Download + parse memo
        memo_map   = {}
        memo_bytes, memo_fn = download_file_for_extraction(meta, "memo")
        if memo_bytes:
            memo_text = extract_text_from_file(memo_bytes, memo_fn, subject)
            if memo_text.strip():
                raw_memo = parse_memo_answers(memo_text, subject, grade)
                memo_map = {_normalise_qnum(k): v for k, v in raw_memo.items()}
                print(f"[Pipeline] Memo answers: {len(memo_map)}")

        # 4. Merge memo answers into questions
        for q in questions:
            qn = _normalise_qnum(q.get("question_number", ""))
            if qn and qn in memo_map and not q.get("memo"):
                q["memo"] = memo_map[qn]

        # 5. Write exam document
        db.collection("exams").document(exam_id).set({
            "title":             title,
            "subject":           subject,
            "grade":             grade,
            "year":              meta.get("year", ""),
            "curriculum":        meta.get("curriculum", "CAPS"),
            "teacherName":       meta.get("teacherName", ""),
            "uploadedBy":        meta.get("uploadedBy", ""),
            "schoolId":          meta.get("schoolId", school_id),
            "examDuration":      meta.get("examDuration", 0),
            "examStoragePath":   meta.get("examStoragePath", ""),
            "memoStoragePath":   meta.get("memoStoragePath", ""),
            "examStorageUrl":    meta.get("examStorageUrl", ""),
            "memoStorageUrl":    meta.get("memoStorageUrl", ""),
            "uploadedAt":        meta.get("uploadedAt", ""),
            "memoMerged":        bool(memo_map),
            "questionsExtracted": True,
            "status":            "ready",
            "totalQuestions":    len(questions),
            "extractedAt":       fs_admin.SERVER_TIMESTAMP,
            "sourceUploadId":    exam_id,
        }, merge=True)

        # 6. Write questions in batches of 400
        batch   = db.batch()
        written = 0
        for i, q in enumerate(questions):
            qtext = str(q.get("question") or "").strip()
            if not qtext:
                continue

            try:
                raw_marks = q.get("marks", 1)
                marks = int(re.sub(r"[^0-9]", "", str(raw_marks))) if raw_marks else 1
                marks = max(1, marks)
            except Exception:
                marks = 1

            options = q.get("options")
            if not isinstance(options, dict):
                options = None

            ref = db.collection("exam_questions").document(f"{exam_id}_{i:04d}")
            batch.set(ref, {
                "examId":          exam_id,
                "questionNumber":  str(q.get("question_number") or i + 1),
                "parentQuestion":  q.get("parent_question", ""),
                "parentContext":   q.get("parent_context"),
                "section":         q.get("section", "A"),
                "questionText":    qtext,
                "type":            q.get("type", "open"),
                "marks":           marks,
                "options":         options,
                "columnA":         q.get("column_a"),
                "columnB":         q.get("column_b"),
                "memo":            str(q.get("memo") or ""),
                "order":           i,
                "questionImageUrl": q.get("questionImageUrl"),
                "hasVisual":       bool(q.get("has_visual")),
                "questionLatex":   q.get("question_latex"),
                "questionTable":   q.get("question_table"),
            })
            written += 1
            if written % 400 == 0:
                batch.commit()
                batch = db.batch()

        batch.commit()
        print(f"[Pipeline] Done: {written} questions, {len(memo_map)} memo answers")
        set_upload_status("extracted", {
            "extractedAt":    datetime.utcnow().isoformat(),
            "totalQuestions": written,
            "memoMerged":     bool(memo_map),
        })

    except Exception as e:
        traceback.print_exc()
        print(f"[Pipeline] FAILED: {e}")
        set_upload_status("error", {"errorMessage": str(e)[:500]})
        try:
            db.collection("exams").document(exam_id).set(
                {"status": "error", "errorMessage": str(e)[:500]},
                merge=True,
            )
        except Exception:
            pass
    finally:
        _unmark_processing(exam_id)


def _launch_pipeline(exam_id, meta, school_id, subject_name):
    if _is_already_processing(exam_id):
        print(f"[Pipeline] Already processing: {exam_id}")
        return False

    try:
        snap = db.collection("exams").document(exam_id).get()
        if snap.exists and snap.to_dict().get("status") == "ready":
            print(f"[Pipeline] Already ready: {exam_id}")
            return False
    except Exception:
        pass

    _mark_processing(exam_id)
    db.collection("exams").document(exam_id).set(
        {"status": "processing", "startedAt": fs_admin.SERVER_TIMESTAMP},
        merge=True,
    )
    threading.Thread(
        target=run_extraction_pipeline,
        args=(exam_id, meta, school_id, subject_name),
        daemon=True,
    ).start()
    return True


# ═══════════════════════════════════════════════════════════════
# FIRESTORE LISTENER + STARTUP SWEEP
# ═══════════════════════════════════════════════════════════════

def _start_auto_extraction_listener():
    def on_snapshot(col_snapshot, changes, read_time):
        for change in changes:
            if change.type.name not in ("ADDED", "MODIFIED"):
                continue
            data         = change.document.to_dict() or {}
            subject_ref  = change.document.reference
            school_id    = subject_ref.parent.parent.id
            subject_name = change.document.id

            for upload in data.get("uploads", []):
                exam_id = upload.get("examId") or upload.get("id")
                if not exam_id:
                    continue
                if upload.get("status") != "pending_extraction":
                    continue
                if not (upload.get("examStoragePath") or upload.get("examStorageUrl")):
                    continue
                if _is_already_processing(exam_id):
                    continue
                print(f"[Listener] Pending: {school_id}/{subject_name}/{exam_id}")
                _launch_pipeline(exam_id, upload, school_id, subject_name)

    def on_error(e):
        print(f"[Listener] Error (collection group index may be missing): {e}")

    try:
        db.collection_group("subjects").on_snapshot(on_snapshot, on_error)
        print("[Listener] Active — watching all subjects")
    except Exception as e:
        print(f"[Listener] Failed to start: {e}")
        print("[Listener] Ensure a collection group index exists for 'subjects' in Firestore")


def _sweep_pending_on_startup():
    print("[Startup] Sweeping for pending extractions...")
    launched = 0
    try:
        for doc in db.collection_group("subjects").stream():
            data         = doc.to_dict() or {}
            school_id    = doc.reference.parent.parent.id
            subject_name = doc.id
            for upload in data.get("uploads", []):
                exam_id = upload.get("examId") or upload.get("id")
                if not exam_id:
                    continue
                if (upload.get("status") == "pending_extraction"
                        and (upload.get("examStoragePath") or upload.get("examStorageUrl"))
                        and not _is_already_processing(exam_id)):
                    print(f"[Startup] Claiming: {exam_id}")
                    if _launch_pipeline(exam_id, upload, school_id, subject_name):
                        launched += 1
    except Exception as e:
        print(f"[Startup] Sweep error: {e}")
        traceback.print_exc()
    print(f"[Startup] Sweep complete. Launched {launched} missed extraction(s)")


# ═══════════════════════════════════════════════════════════════
# SESSION HELPERS
# ═══════════════════════════════════════════════════════════════

def _save_session(sid, payload):
    db.collection("exam_sessions").document(sid).set(payload)


def _get_session(sid):
    if not sid:
        return None
    doc = db.collection("exam_sessions").document(sid).get()
    return doc.to_dict() if doc.exists else None


def _update_session_answers(sid, answers):
    db.collection("exam_sessions").document(sid).update({"answers": answers})


def _delete_session(sid):
    try:
        db.collection("exam_sessions").document(sid).delete()
    except Exception:
        pass


# ─── Load exam + questions ─────────────────────────────────────────────────
def _load_exam(exam_id):
    ref      = db.collection("exams").document(exam_id)
    exam_doc = ref.get()

    if not exam_doc.exists:
        return None, []

    meta = {**exam_doc.to_dict(), "id": exam_doc.id}

    if meta.get("status") != "ready":
        return meta, []

    raw_qs = list(
        db.collection("exam_questions")
          .where("examId", "==", exam_id)
          .stream()
    )
    raw_qs.sort(key=lambda d: d.to_dict().get("order", 0))

    questions = []
    for q in raw_qs:
        d       = q.to_dict()
        options = d.get("options")
        if isinstance(options, dict) and options:
            options = [{"key": k, "value": v} for k, v in sorted(options.items())]
        questions.append({
            "question_number": str(d.get("questionNumber", "")),
            "parent_question": d.get("parentQuestion", ""),
            "parent_context":  d.get("parentContext"),
            "section":         d.get("section", "A"),
            "question":        d.get("questionText", ""),
            "type":            d.get("type", "open").lower(),
            "options":         options,
            "column_a":        d.get("columnA"),
            "column_b":        d.get("columnB"),
            "marks":           d.get("marks", 1),
            "memo":            d.get("memo", ""),
            # ── Rich content fields ───────────────────────────────────────
            "questionImageUrl": d.get("questionImageUrl"),
            "hasVisual":        d.get("hasVisual", False),
            "questionLatex":    d.get("questionLatex"),
            "questionTable":    d.get("questionTable"),
        })

    return meta, questions


# ═══════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "Eduket Extraction & Marking API", "version": "4.0"})


@app.route("/exams/upload", methods=["POST", "OPTIONS"])
def upload_exam():
    if request.method == "OPTIONS":
        return "", 204
    try:
        uid, err = verify_request_token(request)
        if err:
            return err

        data = request.get_json() or {}

        user_doc = db.collection("users").document(uid).get()
        if not user_doc.exists:
            return jsonify({"error": "User profile not found"}), 404

        school_id = user_doc.to_dict().get("schoolId")
        if not school_id:
            return jsonify({"error": "No school associated with this account"}), 400

        school_doc = db.collection("schools").document(school_id).get()
        if not school_doc.exists:
            return jsonify({"error": "School not found"}), 404

        tier_id    = school_doc.to_dict().get("tier", "free")
        exam_limit = get_exam_limit(tier_id)

        now              = datetime.now(timezone.utc)
        start_of_month   = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
        current_count    = len(list(
            db.collection("exams")
              .where("schoolId",   "==", school_id)
              .where("uploadedAt", ">=", start_of_month.isoformat())
              .stream()
        ))

        if current_count >= exam_limit:
            return jsonify({
                "error":   "limit_reached",
                "message": (f"Your school has reached its monthly limit of {exam_limit} exam "
                            f"uploads on the {tier_id.capitalize()} plan."),
                "tier":    tier_id,
                "limit":   exam_limit,
                "used":    current_count,
            }), 403

        exam_id = data.get("examId") or f"{uid}_{int(now.timestamp() * 1000)}"
        subject = data.get("subject", "General")

        # ── Duplicate check — exam file path only, never memo path ────────
        # Checking memoStoragePath caused false positives: any two "skip memo"
        # uploads both have memoStoragePath = '' which incorrectly matched.
        subject_ref      = (db.collection("teacherExamUploads")
                              .document(school_id)
                              .collection("subjects")
                              .document(subject))
        subject_snap     = subject_ref.get()
        existing_uploads = (subject_snap.to_dict().get("uploads", [])
                            if subject_snap.exists else [])

        new_exam_path = data.get("examStoragePath", "")
        if new_exam_path:
            for u in existing_uploads:
                if u.get("examStoragePath") == new_exam_path:
                    print(f"[Upload] Duplicate detected: {new_exam_path}")
                    return jsonify({"examId": u.get("examId"), "duplicate": True})

        record = {
            "examId":          exam_id,
            "uploadedBy":      uid,
            "teacherName":     data.get("teacherName", "Teacher"),
            "schoolId":        school_id,
            "schoolName":      data.get("schoolName", school_id),
            "schoolFolder":    data.get("schoolFolder", school_id),
            "title":           data.get("title", ""),
            "year":            data.get("year", ""),
            "subject":         subject,
            "curriculum":      data.get("curriculum", "CAPS"),
            "grade":           data.get("grade", ""),
            "examDuration":    data.get("examDuration", 0),
            "examFileType":    data.get("examFileType", ""),
            "memoFileType":    data.get("memoFileType", ""),
            "examFileName":    data.get("examFileName", ""),
            "memoFileName":    data.get("memoFileName", ""),
            "examStorageUrl":  data.get("examStorageUrl", ""),
            "memoStorageUrl":  data.get("memoStorageUrl", ""),
            "examStoragePath": data.get("examStoragePath", ""),
            "memoStoragePath": data.get("memoStoragePath", ""),
            "aiMarkingOnly":   data.get("aiMarkingOnly", False),
            "status":          "pending_extraction",
            "questionsExtracted": False,
            "memoMerged":      False,
            "uploadedAt":      now.isoformat(),
            "extractedAt":     None,
        }

        # Write to all three Firestore locations
        db.collection("exams").document(exam_id).set(record)

        db.collection("teacherExamUploads").document(school_id).set({
            "schoolId":    school_id,
            "schoolName":  record["schoolName"],
            "schoolFolder": record["schoolFolder"],
            "updatedAt":   now.isoformat(),
        }, merge=True)

        subject_ref.set({
            "subject":   subject,
            "schoolId":  school_id,
            "uploads":   [{**record, "id": exam_id}] + existing_uploads,
            "updatedAt": now.isoformat(),
        }, merge=True)

        print(f"[Upload] Created exam {exam_id} for school {school_id}")

        # Trigger extraction immediately — don't depend on the Firestore
        # collection group listener, which requires a composite index.
        # The listener still runs as a catch-up mechanism for missed events.
        threading.Thread(
            target=_launch_pipeline,
            args=(exam_id, record, school_id, subject),
            daemon=True,
        ).start()
        print(f"[Upload] Extraction triggered for {exam_id}")

        return jsonify({"examId": exam_id, "duplicate": False})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/exams/usage", methods=["GET", "OPTIONS"])
def exam_usage():
    if request.method == "OPTIONS":
        return "", 204
    try:
        uid, err = verify_request_token(request)
        if err:
            return err

        user_doc = db.collection("users").document(uid).get()
        if not user_doc.exists:
            return jsonify({"error": "User profile not found"}), 404

        school_id = user_doc.to_dict().get("schoolId")
        if not school_id:
            return jsonify({"error": "No school associated with this account"}), 400

        school_doc = db.collection("schools").document(school_id).get()
        if not school_doc.exists:
            return jsonify({"error": "School not found"}), 404

        tier_id    = school_doc.to_dict().get("tier", "free")
        exam_limit = get_exam_limit(tier_id)

        now            = datetime.now(timezone.utc)
        start_of_month = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
        used = len(list(
            db.collection("exams")
              .where("schoolId",   "==", school_id)
              .where("uploadedAt", ">=", start_of_month.isoformat())
              .stream()
        ))

        return jsonify({
            "tier":      tier_id,
            "limit":     exam_limit,
            "used":      used,
            "remaining": max(0, exam_limit - used),
            "atLimit":   used >= exam_limit,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/exams", methods=["GET"])
def list_exams():
    exams = []
    try:
        for doc in db.collection("exams").where("status", "==", "ready").stream():
            d = doc.to_dict()
            exams.append({
                "id":           doc.id,
                "name":         d.get("title", doc.id),
                "subject":      d.get("subject", ""),
                "grade":        d.get("grade", ""),
                "year":         d.get("year", ""),
                "curriculum":   d.get("curriculum", "CAPS"),
                "memoMerged":   d.get("memoMerged", False),
                "examDuration": d.get("examDuration", 0),
            })
    except Exception as e:
        print(f"[list_exams] {e}")
    return jsonify({"exams": exams})


@app.route("/start_exam", methods=["POST"])
def start_exam():
    try:
        data       = request.get_json() or {}
        exam_id    = (data.get("exam_id") or data.get("exam") or
                      data.get("examId") or "").strip()
        student_id = data.get("student_id", "anonymous")

        if not exam_id:
            return jsonify({"error": "exam_id required"}), 400

        meta, questions = _load_exam(exam_id)

        if meta is None:
            return jsonify({"error": f"Exam '{exam_id}' not found"}), 404

        if not questions:
            return jsonify({"error": (
                "This exam has no questions yet — extraction may still be processing "
                f"(status: {meta.get('status', 'unknown')}). Please wait and try again."
            )}), 400

        sid = str(uuid.uuid4())
        _save_session(sid, {
            "exam_id":    exam_id,
            "exam":       meta.get("title", exam_id),
            "subject":    meta.get("subject", ""),
            "student_id": student_id,
            "questions":  questions,
            "answers":    {},
        })

        return jsonify({
            "session_id":            sid,
            "questions":             questions,
            "total_questions":       len(questions),
            "memo_merged":           meta.get("memoMerged", False),
            "subject":               meta.get("subject", ""),
            "title":                 meta.get("title", ""),
            "exam_duration_minutes": meta.get("examDuration", 0),
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/question", methods=["POST"])
def get_question():
    try:
        data    = request.get_json() or {}
        session = _get_session(data.get("session_id"))
        if not session:
            return jsonify({"error": "Invalid session"}), 400
        idx = int(data.get("index", 0))
        qs  = session.get("questions", [])
        if idx < 0 or idx >= len(qs):
            return jsonify({"error": "Index out of range"}), 400
        q = {**qs[idx]}
        q["saved_answer"] = session.get("answers", {}).get(str(idx), "")
        return jsonify(q)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/answer", methods=["POST"])
def save_answer():
    try:
        data    = request.get_json() or {}
        sid     = data.get("session_id")
        session = _get_session(sid)
        if not session:
            return jsonify({"error": "Invalid session"}), 400
        answers               = session.get("answers", {})
        answers[str(data.get("index"))] = data.get("answer", "")
        _update_session_answers(sid, answers)
        return jsonify({"status": "saved"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/submit", methods=["POST"])
def submit_exam():
    try:
        data       = request.get_json() or {}
        exam_id    = data.get("exam_id", "").strip()
        student_id = data.get("student_id", "anonymous")
        answers    = data.get("answers", {})

        if not exam_id:
            return jsonify({"error": "exam_id required"}), 400

        meta, questions = _load_exam(exam_id)
        if not questions:
            return jsonify({"error": "Exam not found or has no questions"}), 404

        subject     = meta.get("subject", "General")
        total_score = 0.0
        total_marks = 0.0
        results     = []

        for i, q in enumerate(questions):
            q_num       = q.get("question_number", f"Q{i+1}")
            q_type      = q.get("type", "open").lower()
            marks       = float(q.get("marks") or 1)
            memo        = q.get("memo", "")
            student_ans = str(answers.get(str(i), "")).strip()
            total_marks += marks

            options = q.get("options")
            if isinstance(options, list) and options and isinstance(options[0], dict):
                options = {o["key"]: o["value"] for o in options}

            marked = mark_with_memo(student_ans, memo, marks)
            if marked is None:
                marked = mark_with_ai(q.get("question", ""), student_ans,
                                      marks, subject, memo)

            earned       = float(marked.get("score", 0))
            total_score += earned

            correct_display = memo if memo else "Not available"
            if memo and q_type == "mcq" and isinstance(options, dict):
                letter          = str(memo).strip().upper()
                correct_display = (f"{letter}. {options.get(letter, '')}"
                                   if letter in options else letter)

            results.append({
                "question_number": q_num,
                "question":        q.get("question", ""),
                "type":            q_type,
                "marks":           marks,
                "earned":          earned,
                "score":           earned,
                "status":          marked.get("status", "incorrect"),
                "student_answer":  student_ans or "No answer",
                "correct_answer":  correct_display,
                "feedback":        marked.get("feedback", ""),
                "concept_gap":     marked.get("concept_gap", ""),
                "model_answer":    marked.get("model_answer", ""),
                # questionAnalysis removed — was hardcoded test data
            })

        percentage = round(total_score / total_marks * 100, 1) if total_marks else 0
        feedback   = generate_final_feedback(percentage, results, subject)
        analysis   = generate_exam_analysis(subject, percentage,
                                            total_score, total_marks, results)

        print(f"[submit] ✅ {total_score}/{total_marks} = {percentage}%", flush=True)
        return jsonify({
            "score":      total_score,
            "total":      total_marks,
            "percentage": percentage,
            "results":    results,
            "feedback":   feedback,
            "analysis":   analysis,
            "subject":    subject,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/results/<exam_id>/<student_id>", methods=["GET"])
def get_results(exam_id, student_id):
    try:
        docs = list(
            db.collection("exam_attempts")
              .where("examId",    "==", exam_id)
              .where("studentId", "==", student_id)
              .order_by("completedAt", direction="DESCENDING")
              .limit(1)
              .stream()
        )
        if not docs:
            return jsonify({"error": "Results not found"}), 404
        return jsonify({"success": True, "result": docs[0].to_dict()})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/autosave", methods=["POST", "OPTIONS"])
def autosave_exam():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    try:
        data       = request.get_json() or {}
        exam_id    = data.get("exam_id") or data.get("examId", "")
        student_id = data.get("student_id") or data.get("studentId", "")
        answers    = data.get("answers", {})
        if not exam_id or not student_id:
            return jsonify({"error": "Missing exam_id or student_id"}), 400
        db.collection("exam_autosaves").document(f"{exam_id}_{student_id}").set({
            "examId":    exam_id,
            "studentId": student_id,
            "answers":   answers,
            "updatedAt": fs_admin.SERVER_TIMESTAMP,
        }, merge=True)
        return jsonify({"success": True})
    except Exception as e:
        print(f"[autosave] {e}", flush=True)
        return jsonify({"error": str(e)}), 500


@app.route("/autosave/<exam_id>/<student_id>", methods=["GET"])
def load_autosave(exam_id, student_id):
    try:
        doc     = db.collection("exam_autosaves").document(f"{exam_id}_{student_id}").get()
        answers = doc.to_dict().get("answers", {}) if doc.exists else {}
        return jsonify({"success": True, "answers": answers})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/remark", methods=["POST", "OPTIONS"])
def remark():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    try:
        data    = request.get_json() or {}
        rows    = data.get("results", [])
        subject = data.get("subject", "General")
        updated = []
        for i, r in enumerate(rows):
            student_ans = r.get("student_answer", "").strip()
            memo        = r.get("correct_answer", "")
            marks       = float(r.get("marks", 1))
            question    = r.get("question", "")
            marked      = mark_with_memo(student_ans, memo, marks)
            if marked is None:
                marked  = mark_with_ai(question, student_ans, marks, subject, memo)
            updated.append({"idx": i, "earned": marked.get("score", 0),
                             "status": marked.get("status", "incorrect"),
                             "feedback": marked.get("feedback", "")})
        return jsonify({"results": updated})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/admin/extraction-status/<exam_id>", methods=["GET"])
def extraction_status(exam_id):
    try:
        doc = db.collection("exams").document(exam_id).get()
        if doc.exists:
            d       = doc.to_dict()
            q_count = sum(1 for _ in
                          db.collection("exam_questions")
                            .where("examId", "==", exam_id)
                            .stream())
            return jsonify({
                "status":             d.get("status"),
                "title":              d.get("title"),
                "subject":            d.get("subject"),
                "questions_in_db":    q_count,
                "memo_merged":        d.get("memoMerged", False),
                "student_accessible": d.get("status") == "ready" and q_count > 0,
            })
        return jsonify({"status": "not_found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/admin/trigger-extract/<exam_id>", methods=["GET"])
def trigger_extract(exam_id):
    try:
        exam_doc     = db.collection("exams").document(exam_id).get()
        meta         = None
        school_id    = "shared"
        subject_name = "General"

        if exam_doc.exists:
            meta         = exam_doc.to_dict()
            school_id    = meta.get("schoolId", "shared")
            subject_name = meta.get("subject",  "General")
        else:
            for doc in db.collection_group("subjects").stream():
                for upload in (doc.to_dict() or {}).get("uploads", []):
                    if upload.get("examId") == exam_id or upload.get("id") == exam_id:
                        meta         = upload
                        school_id    = doc.reference.parent.parent.id
                        subject_name = doc.id
                        break
                if meta:
                    break

        if not meta:
            return jsonify({"error": f"Exam {exam_id} not found"}), 404

        db.collection("exams").document(exam_id).set(
            {"status": "pending_extraction"}, merge=True
        )
        _unmark_processing(exam_id)
        threading.Thread(
            target=run_extraction_pipeline,
            args=(exam_id, meta, school_id, subject_name),
            daemon=True,
        ).start()
        return jsonify({"ok": True, "message": "Extraction started",
                        "poll": f"/admin/extraction-status/{exam_id}"})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/dashboard", methods=["POST", "OPTIONS"])
def dashboard():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    try:
        data       = request.get_json(silent=True) or {}
        student_id = data.get("student_id", "").strip()
        if not student_id:
            return jsonify({"error": "student_id required"}), 400

        attempts = []
        try:
            attempts = list(
                db.collection("exam_attempts")
                  .where("studentId", "==", student_id)
                  .stream()
            )
        except Exception as e:
            print(f"[dashboard] attempts fetch failed: {e}", flush=True)

        weak_map = {}
        for attempt in attempts:
            d = attempt.to_dict()
            for r in d.get("markedResults", []):
                if r.get("status") == "correct":
                    continue
                qnum = str(r.get("question_number", ""))
                if not qnum:
                    continue
                if qnum not in weak_map:
                    weak_map[qnum] = {"question_number": qnum,
                                      "question_text": r.get("question", ""),
                                      "q_type": r.get("type", "open"),
                                      "wrong_count": 0}
                weak_map[qnum]["wrong_count"] += 1

        weak       = sorted(weak_map.values(),
                            key=lambda x: x["wrong_count"], reverse=True)[:20]
        study_plan = None
        try:
            plan_doc = db.collection("study_plans").document(student_id).get()
            if plan_doc.exists:
                pd         = plan_doc.to_dict()
                study_plan = {"plan": pd.get("plan", ""),
                              "updated_at": str(pd.get("updatedAt", ""))}
        except Exception as e:
            print(f"[dashboard] study_plan failed: {e}", flush=True)

        return jsonify({"student_id": student_id, "weak": weak,
                        "study_plan": study_plan, "session_history": []})
    except Exception as e:
        print(f"[dashboard] ❌ {e}", flush=True)
        return jsonify({"error": str(e)}), 500


@app.route("/admin/cleanup-sessions", methods=["POST"])
def cleanup_sessions():
    from datetime import timedelta
    cutoff  = datetime.now(timezone.utc) - timedelta(hours=24)
    deleted = 0
    for doc in db.collection("exam_sessions").stream():
        created = doc.to_dict().get("createdAt")
        if created and created < cutoff:
            doc.reference.delete()
            deleted += 1
    return jsonify({"deleted": deleted})


@app.route("/agent-chat", methods=["POST"])
def agent_chat():
    try:
        data             = request.get_json(force=True)
        student_id       = data.get("student_id", "")
        student_message  = data.get("message", "").strip()
        learning_profile = data.get("learningProfile", {})
        latest_attempt   = data.get("latestAttempt", {})
        history          = data.get("history", [])

        if not student_message:
            return jsonify({"error": "Message cannot be empty."}), 400

        try:
            latest_questions = json.dumps([
                {"q": r.get("question_number"), "status": r.get("status"),
                 "topic": r.get("question", "")[:60]}
                for r in latest_attempt.get("markedResults", [])[:10]
            ])
        except Exception:
            latest_questions = "[]"

        try:
            subjects = ", ".join(learning_profile.get("subjects", ["Unknown"]))
        except Exception:
            subjects = "Unknown"

        try:
            weak_areas_full = json.dumps([
                {"question": w.get("question") or w.get("key", ""),
                 "timesWrong": w.get("timesWrong") or w.get("count", 0),
                 "type": w.get("type", ""), "text": w.get("text", "")[:80]}
                for w in learning_profile.get("weakAreas", [])[:8]
                if isinstance(w, dict)
            ])
        except Exception:
            weak_areas_full = "[]"

        system_prompt = f"""You are NextGen Skills AI Academic Coach — a brilliant, patient South African
CAPS/NSC curriculum tutor. Teach through natural conversation.
RULES: Never give everything at once. After every response, ask ONE question.
Keep each response SHORT (4–6 sentences max). Break topics into steps — teach step 1, then WAIT.
Student: {student_id} | Subjects: {subjects}
Average: {learning_profile.get('overallAverage', 'Unknown')}% | Weak areas: {weak_areas_full}"""

        user_context = f"""STUDENT MESSAGE: {student_message}
Latest exam: {latest_attempt.get('examTitle','N/A')} | {latest_attempt.get('percentage','?')}%
Latest questions: {latest_questions}"""

        messages = [{"role": "system", "content": system_prompt}]
        for item in (history[-10:] if isinstance(history, list) else []):
            if (isinstance(item, dict)
                    and item.get("role") in ("user", "assistant")
                    and item.get("content")):
                messages.append({"role": item["role"], "content": item["content"]})
        messages.append({"role": "user", "content": user_context})

        client     = Groq(api_key=os.getenv("GROQ_API_KEY"))
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages, temperature=0.4, max_tokens=600,
        )
        reply = completion.choices[0].message.content.strip()

        return jsonify({
            "success":    True,
            "coach":      "NextGen AI Academic Coach",
            "response":   reply,
            "student_id": student_id,
            "suggestions": [
                "Explain my weakest concept",
                "Create a personalised study timetable",
                "Generate 10 practice questions",
                "Test my understanding",
                "Explain my last exam mistakes",
                "How can I improve to distinction level?",
            ],
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e),
                        "response": "I couldn't process your request right now."}), 500


# ── Startup sequence ──────────────────────────────────────────
try:
    _init_firebase()
except Exception as e:
    traceback.print_exc()
    raise SystemExit(1)

_sweep_pending_on_startup()
_start_auto_extraction_listener()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)