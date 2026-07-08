"""
app.py — Eduket OS  Production API  v5
═══════════════════════════════════════════════════════════════════════════════
AI provider chain  (Groq → Gemini, automatic fallback)
────────────────────────────────────────────────────────
Every AI call goes through ai_text() which tries Groq first and falls back
to Gemini automatically when Groq returns a model error, rate limit, or any
other failure. No manual intervention required when Groq deprecates models.

Environment variables
──────────────────────
  GROQ_API_KEY                   — primary AI provider
  GEMINI_API_KEY                 — fallback (aistudio.google.com — free)
  FIREBASE_SERVICE_ACCOUNT_JSON  — Firebase Admin SDK credentials
  FIREBASE_STORAGE_BUCKET        — e.g. eduket.firebasestorage.app

Groq model auto-resolution
───────────────────────────
_resolve_groq_model() queries Groq's live /models endpoint on first call
and picks the first working model from _GROQ_MODEL_CANDIDATES. The result
is cached for the process lifetime. If a cached model is later decommissioned
the cache is invalidated and re-resolved on the next call.
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


# ══════════════════════════════════════════════════════════════════════════════
# AI PROVIDER LAYER  —  Groq → Gemini
# ══════════════════════════════════════════════════════════════════════════════

# Groq model priority list — first available wins.
# Checked against Groq's live /models endpoint on first call.
_GROQ_MODEL_CANDIDATES = [
    "llama-3.1-70b-versatile",
    "llama3-70b-8192",
    "llama-3.3-70b-specdec",
    # "llama-3.1-8b-instant",    # last resort — small context
]

_RESOLVED_GROQ_MODEL: str | None = None


def _resolve_groq_model() -> str:
    """
    Returns the first Groq model candidate that is currently available.
    Caches the result; invalidates cache if a decommissioned error fires.
    """
    global _RESOLVED_GROQ_MODEL
    if _RESOLVED_GROQ_MODEL:
        return _RESOLVED_GROQ_MODEL

    try:
        client    = Groq(api_key=os.getenv("GROQ_API_KEY"))
        available = {m.id for m in client.models.list().data}
        for candidate in _GROQ_MODEL_CANDIDATES:
            if candidate in available:
                print(f"[Model] Groq model resolved: {candidate}")
                _RESOLVED_GROQ_MODEL = candidate
                return candidate
    except Exception as e:
        print(f"[Model] Could not query Groq models: {e}")

    fallback = _GROQ_MODEL_CANDIDATES[-1]
    print(f"[Model] Groq fallback: {fallback}")
    _RESOLVED_GROQ_MODEL = fallback
    return fallback


def ai_text(prompt: str,
            max_tokens: int = 2000,
            temperature: float = 0.1) -> str:
    """
    Send a text prompt through the AI provider chain.

    Chain: Groq → Gemini
    Raises RuntimeError only when both providers fail.
    """
    global _RESOLVED_GROQ_MODEL
    last_error: Exception | None = None

    # ── 1. Groq ───────────────────────────────────────────────────────────
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        for attempt in range(2):
            try:
                client = Groq(api_key=groq_key)
                resp   = client.chat.completions.create(
                    model=_resolve_groq_model(),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                err = str(e)
                # Model decommissioned — invalidate cache and retry once
                if ("decommissioned" in err or "deprecated" in err) and attempt == 0:
                    _RESOLVED_GROQ_MODEL = None
                    print(f"[Groq] Model decommissioned — re-resolving")
                    continue
                last_error = e
                print(f"[Groq] Attempt {attempt + 1} failed: {err[:120]}")
                time.sleep(1)
                break

    # ── 2. Gemini ─────────────────────────────────────────────────────────
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model    = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(prompt)
            print("[AI] Gemini fallback responded")
            return response.text.strip()
        except Exception as e:
            last_error = e
            print(f"[Gemini] Failed: {str(e)[:120]}")

    raise RuntimeError(
        f"All AI providers failed. Last: {last_error}. "
        "Check GROQ_API_KEY and GEMINI_API_KEY."
    )


# ══════════════════════════════════════════════════════════════════════════════
# FIREBASE INITIALIZATION
# ══════════════════════════════════════════════════════════════════════════════

db     = None
bucket = None


def _init_firebase():
    global db, bucket

    raw = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON", "").strip()
    if not raw:
        raise ValueError("FIREBASE_SERVICE_ACCOUNT_JSON is not set")

    cred_dict = (
        json.load(open(raw))
        if os.path.exists(raw)
        else json.loads(raw)
    )
    if "private_key" in cred_dict:
        cred_dict["private_key"] = cred_dict["private_key"].replace("\\n", "\n")

    missing = [k for k in ["type", "project_id", "private_key", "client_email"]
               if not cred_dict.get(k)]
    if missing:
        raise ValueError(f"Credential dict missing keys: {missing}")

    print(f"[Firebase] project_id : {cred_dict['project_id']}")
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
    header = req.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        return None, (jsonify({"error": "Missing or malformed Authorization header"}), 401)
    try:
        decoded = fb_auth.verify_id_token(header.split("Bearer ", 1)[1].strip())
        return decoded["uid"], None
    except Exception as e:
        print(f"[Auth] Token verification failed: {e}")
        return None, (jsonify({"error": "Invalid or expired token"}), 401)


# ══════════════════════════════════════════════════════════════════════════════
# TIER LIMITS
# ══════════════════════════════════════════════════════════════════════════════

TIER_EXAM_LIMITS = {
    "free":     5,
    "silver":   30,
    "gold":     120,
    "platinum": 500,
    "diamond":  1000,
}


def get_exam_limit(tier_id: str) -> int:
    return TIER_EXAM_LIMITS.get(tier_id, TIER_EXAM_LIMITS["free"])


# ══════════════════════════════════════════════════════════════════════════════
# FLASK APP + CORS
# ══════════════════════════════════════════════════════════════════════════════
# Note: the @app.before_request OPTIONS handler was removed — it conflicted
# with Flask-CORS and caused CORS header mismatches on preflight responses.

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
        "https://eduket.tech",         # no trailing slash — browsers never send one
    ],
    "methods":       ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    "allow_headers": ["Content-Type", "Authorization"],
}}, supports_credentials=False)

from billing_routes import billing_bp
app.register_blueprint(billing_bp)


# ══════════════════════════════════════════════════════════════════════════════
# THREAD-SAFE PROCESSING TRACKER
# ══════════════════════════════════════════════════════════════════════════════

_PROCESSING      = set()
_PROCESSING_LOCK = threading.Lock()


def _is_processing(exam_id: str) -> bool:
    with _PROCESSING_LOCK:
        return exam_id in _PROCESSING


def _mark_processing(exam_id: str):
    with _PROCESSING_LOCK:
        _PROCESSING.add(exam_id)


def _unmark_processing(exam_id: str):
    with _PROCESSING_LOCK:
        _PROCESSING.discard(exam_id)


# ══════════════════════════════════════════════════════════════════════════════
# FIREBASE STORAGE DOWNLOAD
# ══════════════════════════════════════════════════════════════════════════════

def download_file_for_extraction(meta: dict, file_type: str):
    """
    Download an exam or memo file from Firebase Storage.
    Tries the Admin SDK blob path first, falls back to the download URL.
    Returns (file_bytes, filename) or (None, filename) on failure.
    """
    filename     = meta.get(f"{file_type}FileName", f"{file_type}.docx")
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

    print(f"[Storage] No source available for {file_type}")
    return None, filename


# ══════════════════════════════════════════════════════════════════════════════
# MEMO PARSER
# ══════════════════════════════════════════════════════════════════════════════

def parse_memo_answers(memo_text: str, subject: str, grade: str) -> dict:
    """
    Extract question_number → answer mappings from a marking memorandum.
    Processes in 6,000-char chunks to stay within provider token limits.
    Returns a dict keyed by normalised question number.
    """
    CHUNK   = 6_000
    result  = {}

    chunks = [memo_text[i:i + CHUNK] for i in range(0, len(memo_text), CHUNK)]

    for idx, chunk in enumerate(chunks):
        print(f"[Memo] Chunk {idx + 1}/{len(chunks)}")
        try:
            raw = ai_text(
                f"""You are reading a South African CAPS/NSC exam MARKING MEMORANDUM.
Extract EVERY answer. Return ONLY a valid JSON object mapping question_number to answer.
For MCQ: give just the letter (A/B/C/D). For True/False: "True" or "False".
No markdown, no explanation, no preamble.
Example: {{"1.1": "C", "1.2": "True", "1.3": "RAM is volatile memory."}}
Subject: {subject} | Grade: {grade}
MEMO TEXT:
{chunk}""",
                max_tokens=8000,
                temperature=0,
            )
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                chunk_result = json.loads(match.group())
                if isinstance(chunk_result, dict):
                    for k, v in chunk_result.items():
                        norm = _normalise_qnum(k)
                        if norm and norm not in result:
                            result[norm] = v
        except Exception as e:
            print(f"[Memo] Chunk {idx + 1} failed: {e}")

    print(f"[Memo] Total answers extracted: {len(result)}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# MARKING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _normalise_text(v) -> str:
    return "" if v is None else str(v).strip().lower()


def _normalise_qnum(qn: str) -> str:
    s = str(qn).lower().strip()
    s = re.sub(r"^(question|q|ques|no|nr)[\s.\-]*", "", s)
    s = re.sub(r"[^a-z0-9]", "", s)
    return s


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalise_text(a), _normalise_text(b)).ratio()


def mark_with_memo(student_answer: str, memo_answer: str,
                   marks: float) -> dict | None:
    """
    Rule-based marking against a memo answer.
    Returns None to signal AI fallback is needed.
    """
    s = _normalise_text(student_answer)
    m = _normalise_text(memo_answer)

    if not s:
        return {"score": 0, "status": "missing",
                "feedback": "No answer provided.", "concept_gap": "Question not attempted."}

    if not m:
        return None  # no memo — fall through to AI

    if s == m:
        return {"score": marks, "status": "correct",
                "feedback": "Correct.", "concept_gap": ""}

    # MCQ — single letter
    if len(m) == 1 and m.isalpha():
        if s.startswith(m):
            return {"score": marks, "status": "correct",
                    "feedback": "Correct option selected.", "concept_gap": ""}
        return {"score": 0, "status": "incorrect",
                "feedback": f"Incorrect. Correct answer: {memo_answer.upper()}.",
                "concept_gap": "Wrong option selected."}

    # True / False
    if m in ("true", "false"):
        if s.startswith(m):
            return {"score": marks, "status": "correct",
                    "feedback": "Correct.", "concept_gap": ""}
        return {"score": 0, "status": "incorrect",
                "feedback": f"Incorrect. Answer is {memo_answer}.",
                "concept_gap": "True/False answer incorrect."}

    # Fuzzy similarity for short answers
    sim = _similarity(s, m)
    if sim >= 0.75:
        return {"score": marks, "status": "correct",
                "feedback": "Correct.", "concept_gap": ""}

    return None  # below threshold — AI fallback


def mark_with_ai(question: str, student_answer: str,
                 marks: float, subject: str, memo: str = "") -> dict:
    """
    AI marking for open-ended, calculation, and essay questions.
    Focuses on conceptual understanding — spelling errors are forgiven.
    """
    prompt = f"""You are a senior South African CAPS/NSC examiner for {subject}.
Mark fairly based on CONCEPTUAL UNDERSTANDING — not perfect wording.
IGNORE spelling mistakes and grammatical errors. Focus on whether the student understands the concept.

QUESTION: {question}
MARKS AVAILABLE: {marks}
MEMO/EXPECTED ANSWER: {memo if memo else f"Use your {subject} curriculum expertise to determine correctness."}
STUDENT ANSWER: {student_answer if str(student_answer).strip() else "No answer provided."}

Return ONLY this exact JSON — no explanation, no preamble:
{{
  "score": <number 0 to {marks}>,
  "status": "<correct|partial|incorrect|missing>",
  "feedback": "<specific, constructive feedback>",
  "concept_gap": "<concept missed, or empty string if fully correct>",
  "model_answer": "<ideal answer in 1-2 sentences>"
}}"""

    try:
        raw   = ai_text(prompt, max_tokens=800, temperature=0.1)
        raw   = re.sub(r"^```json\s*|^```\s*|```$", "", raw,
                       flags=re.MULTILINE).strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            result          = json.loads(match.group())
            result["score"] = max(0.0, min(float(result.get("score", 0)), marks))
            return result
    except Exception as e:
        print(f"[AI Mark] Failed: {e}")

    return {"score": 0, "status": "incorrect",
            "feedback": "Could not mark — AI unavailable.",
            "concept_gap": "Unknown.", "model_answer": ""}


def generate_final_feedback(percentage: float, results: list,
                            subject: str) -> str:
    """Generate a concise overall feedback string for the student."""
    wrong   = [r for r in results if r.get("status") in ("incorrect", "missing")]
    partial = [r for r in results if r.get("status") == "partial"]
    gaps    = list({r.get("concept_gap", "") for r in results
                    if r.get("concept_gap", "").strip()})

    if   percentage >= 80: tone = f"Excellent work! Strong command of {subject}."
    elif percentage >= 60: tone = f"Good effort. A solid attempt at {subject}."
    elif percentage >= 40: tone = f"Average performance. More revision of {subject} is needed."
    else:                  tone = f"Below average. Serious revision of {subject} is required."

    lines = [tone]
    if wrong:
        nums = ", ".join(str(r.get("question_number", "?")) for r in wrong[:8])
        lines.append(f"Questions needing attention: {nums}.")
    if partial:
        nums = ", ".join(str(r.get("question_number", "?")) for r in partial[:5])
        lines.append(f"Partially correct: {nums} — expand your answers.")
    lines.append(
        f"Key concept gaps: {'; '.join(gaps[:5]) if gaps else 'None identified'}."
    )
    return " ".join(lines)


def generate_exam_analysis(subject: str, percentage: float,
                           total_score: float, total_marks: float,
                           results: list) -> dict:
    """
    Generate a deep cognitive and learning-style analysis for the student.
    Used to populate the results dashboard and parent/teacher reports.
    """
    payload = [
        {
            "question":       r.get("question", ""),
            "student_answer": r.get("student_answer", ""),
            "correct_answer": r.get("correct_answer", ""),
            "status":         r.get("status", ""),
            "marks":          r.get("marks", 0),
            "earned":         r.get("earned", 0),
            "feedback":       r.get("feedback", ""),
        }
        for r in results
    ]

    prompt = f"""You are an expert teacher and learning analyst for {subject}.
Analyse this student's performance and identify conceptual strengths and weaknesses.
Student scored: {total_score}/{total_marks} ({percentage}%)

Return ONLY valid JSON — no markdown, no explanation:
{{
  "overallSummary":   "",
  "studentProfile":   "",
  "strengths":        [],
  "weaknesses":       [],
  "misconceptions":   [],
  "learningStyle":    "",
  "cognitiveAnalysis": {{
    "remember":0,"understand":0,"apply":0,"analyse":0,"evaluate":0,"create":0
  }},
  "studyPlan":        [],
  "teacherSummary":   "",
  "parentSummary":    ""
}}

Exam data:
{json.dumps(payload, indent=2)}"""

    try:
        raw   = ai_text(prompt, max_tokens=2500, temperature=0.2)
        raw   = re.sub(r"^```json\s*|^```\s*|```$", "", raw,
                       flags=re.MULTILINE).strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception as e:
        print(f"[Analysis] {e}")
    return {}


# ══════════════════════════════════════════════════════════════════════════════
# EXTRACTION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def _subject_doc_ref(school_id: str, subject_name: str):
    return (db.collection("teacherExamUploads")
              .document(school_id)
              .collection("subjects")
              .document(subject_name))


def run_extraction_pipeline(exam_id: str, meta: dict,
                             school_id: str, subject_name: str):
    """
    Full extraction pipeline for one exam:
      1. Download exam file from Firebase Storage
      2. extract_questions_from_file (render-first via extraction_engine)
      3. Download + parse memo (if provided)
      4. Merge memo answers into questions
      5. Write exams/{exam_id} document
      6. Write exam_questions in Firestore batches
      7. Update status → "extracted"
    """
    subject_ref = _subject_doc_ref(school_id, subject_name)

    def set_status(status: str, extra: dict = {}):
        """Update the upload record's status field inside the subjects array."""
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
        set_status("processing",
                   {"processingStartedAt": datetime.utcnow().isoformat()})

        # 1. Download exam file
        exam_bytes, exam_fn = download_file_for_extraction(meta, "exam")
        if not exam_bytes:
            raise ValueError("Could not download exam file from Firebase Storage.")

        # 2. Extract questions — render-first (LibreOffice + vision)
        questions = extract_questions_from_file(
            exam_bytes, exam_fn, subject, grade,
            exam_id=exam_id,
            school_folder=meta.get("schoolFolder", school_id),
        )
        print(f"[Pipeline] Questions extracted: {len(questions)}")
        if not questions:
            raise ValueError(
                "No questions could be extracted. "
                "Check the file is a valid exam paper and that LibreOffice is installed."
            )

        # 3. Download + parse memo (skipped when aiMarkingOnly=True)
        memo_map: dict = {}
        if not meta.get("aiMarkingOnly"):
            memo_bytes, memo_fn = download_file_for_extraction(meta, "memo")
            if memo_bytes:
                memo_text = extract_text_from_file(memo_bytes, memo_fn, subject)
                if memo_text.strip():
                    raw_memo = parse_memo_answers(memo_text, subject, grade)
                    memo_map = {_normalise_qnum(k): v for k, v in raw_memo.items()}
                    print(f"[Pipeline] Memo answers: {len(memo_map)}")

        # 4. Merge memo answers into questions where not already set
        for q in questions:
            qn = _normalise_qnum(q.get("question_number", ""))
            if qn and qn in memo_map and not q.get("memo"):
                q["memo"] = memo_map[qn]

        # 5. Write top-level exams document
        db.collection("exams").document(exam_id).set({
            "title":              title,
            "subject":            subject,
            "grade":              grade,
            "year":               meta.get("year", ""),
            "curriculum":         meta.get("curriculum", "CAPS"),
            "teacherName":        meta.get("teacherName", ""),
            "uploadedBy":         meta.get("uploadedBy", ""),
            "schoolId":           meta.get("schoolId", school_id),
            "examDuration":       meta.get("examDuration", 0),
            "examStoragePath":    meta.get("examStoragePath", ""),
            "memoStoragePath":    meta.get("memoStoragePath", ""),
            "examStorageUrl":     meta.get("examStorageUrl", ""),
            "memoStorageUrl":     meta.get("memoStorageUrl", ""),
            "uploadedAt":         meta.get("uploadedAt", ""),
            "memoMerged":         bool(memo_map),
            "questionsExtracted": True,
            "status":             "ready",
            "totalQuestions":     len(questions),
            "extractedAt":        fs_admin.SERVER_TIMESTAMP,
            "sourceUploadId":     exam_id,
        }, merge=True)

        # 6. Write exam_questions in Firestore (batch 400 at a time)
        batch   = db.batch()
        written = 0

        for i, q in enumerate(questions):
            qtext = str(q.get("question") or "").strip()
            if not qtext:
                continue

            try:
                marks = int(re.sub(r"[^0-9]", "", str(q.get("marks", 1))) or "1")
                marks = max(1, marks)
            except Exception:
                marks = 1

            options = q.get("options")
            if not isinstance(options, dict):
                options = None

            ref = db.collection("exam_questions").document(f"{exam_id}_{i:04d}")
            batch.set(ref, {
                "examId":           exam_id,
                "questionNumber":   str(q.get("question_number") or i + 1),
                "parentQuestion":   q.get("parent_question", ""),
                "parentContext":    q.get("parent_context"),
                "section":          q.get("section", "A"),
                "questionText":     qtext,
                "type":             q.get("type", "open"),
                "marks":            marks,
                "options":          options,
                "columnA":          q.get("column_a"),
                "columnB":          q.get("column_b"),
                "memo":             str(q.get("memo") or ""),
                "order":            i,
                # Rich content — populated by extraction_engine vision pipeline
                "questionImageUrl": q.get("questionImageUrl"),
                "hasVisual":        bool(q.get("has_visual")),
                "questionLatex":    q.get("question_latex"),
                "questionTable":    q.get("question_table"),
            })
            written += 1

            # Commit every 400 documents (Firestore batch limit)
            if written % 400 == 0:
                batch.commit()
                batch = db.batch()

        batch.commit()
        print(f"[Pipeline] ✓ Done — {written} questions, {len(memo_map)} memo answers")

        set_status("extracted", {
            "extractedAt":    datetime.utcnow().isoformat(),
            "totalQuestions": written,
            "memoMerged":     bool(memo_map),
        })

    except Exception as e:
        traceback.print_exc()
        print(f"[Pipeline] FAILED: {e}")
        set_status("error", {"errorMessage": str(e)[:500]})
        try:
            db.collection("exams").document(exam_id).set(
                {"status": "error", "errorMessage": str(e)[:500]},
                merge=True,
            )
        except Exception:
            pass
    finally:
        _unmark_processing(exam_id)


def _launch_pipeline(exam_id: str, meta: dict,
                     school_id: str, subject_name: str) -> bool:
    if _is_processing(exam_id):
        print(f"[Pipeline] Already processing: {exam_id}")
        return False

    # Skip if already successfully extracted
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


# ══════════════════════════════════════════════════════════════════════════════
# FIRESTORE LISTENER + STARTUP SWEEP
# ══════════════════════════════════════════════════════════════════════════════

def _start_auto_extraction_listener():
    """
    Watch all subjects sub-collections for new pending_extraction documents.
    Acts as a catch-up mechanism — primary triggering happens directly in
    upload_exam() via _launch_pipeline().
    """
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
                if _is_processing(exam_id):
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
        print("[Listener] Ensure a collection group index exists for 'subjects'")


def _sweep_pending_on_startup():
    """Re-queue any extractions that were pending before the last server restart."""
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
                        and not _is_processing(exam_id)):
                    print(f"[Startup] Claiming: {exam_id}")
                    if _launch_pipeline(exam_id, upload, school_id, subject_name):
                        launched += 1
    except Exception as e:
        print(f"[Startup] Sweep error: {e}")
        traceback.print_exc()
    print(f"[Startup] Sweep complete — {launched} extraction(s) launched")


# ══════════════════════════════════════════════════════════════════════════════
# SESSION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _save_session(sid: str, payload: dict):
    db.collection("exam_sessions").document(sid).set(payload)


def _get_session(sid: str) -> dict | None:
    if not sid:
        return None
    doc = db.collection("exam_sessions").document(sid).get()
    return doc.to_dict() if doc.exists else None


def _update_session_answers(sid: str, answers: dict):
    db.collection("exam_sessions").document(sid).update({"answers": answers})


def _delete_session(sid: str):
    try:
        db.collection("exam_sessions").document(sid).delete()
    except Exception:
        pass


def _load_exam(exam_id: str) -> tuple[dict | None, list]:
    """Load exam metadata and all questions from Firestore."""
    ref      = db.collection("exams").document(exam_id)
    exam_doc = ref.get()
    if not exam_doc.exists:
        return None, []

    meta = {**exam_doc.to_dict(), "id": exam_doc.id}
    if meta.get("status") != "ready":
        return meta, []

    raw_qs = sorted(
        db.collection("exam_questions").where("examId", "==", exam_id).stream(),
        key=lambda d: d.to_dict().get("order", 0),
    )

    questions = []
    for q in raw_qs:
        d       = q.to_dict()
        options = d.get("options")
        if isinstance(options, dict) and options:
            options = [{"key": k, "value": v} for k, v in sorted(options.items())]
        questions.append({
            "question_number":  str(d.get("questionNumber", "")),
            "parent_question":  d.get("parentQuestion", ""),
            "parent_context":   d.get("parentContext"),
            "section":          d.get("section", "A"),
            "question":         d.get("questionText", ""),
            "type":             d.get("type", "open").lower(),
            "options":          options,
            "column_a":         d.get("columnA"),
            "column_b":         d.get("columnB"),
            "marks":            d.get("marks", 1),
            "memo":             d.get("memo", ""),
            # Rich content fields — populated by render-first pipeline
            "questionImageUrl": d.get("questionImageUrl"),
            "hasVisual":        d.get("hasVisual", False),
            "questionLatex":    d.get("questionLatex"),
            "questionTable":    d.get("questionTable"),
        })

    return meta, questions


# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status":  "ok",
        "service": "Eduket Extraction & Marking API",
        "version": "5.0",
    })


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

        # Monthly upload count — requires composite index: schoolId ASC, uploadedAt ASC
        now            = datetime.now(timezone.utc)
        start_of_month = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
        current_count  = len(list(
            db.collection("exams")
              .where("schoolId",   "==", school_id)
              .where("uploadedAt", ">=", start_of_month.isoformat())
              .stream()
        ))

        if current_count >= exam_limit:
            return jsonify({
                "error":   "limit_reached",
                "message": (f"Your school has reached its monthly limit of {exam_limit} "
                            f"exam uploads on the {tier_id.capitalize()} plan."),
                "tier":  tier_id,
                "limit": exam_limit,
                "used":  current_count,
            }), 403

        exam_id = data.get("examId") or f"{uid}_{int(now.timestamp() * 1000)}"
        subject = data.get("subject", "General")

        # ── Duplicate check — exam file path only ─────────────────────────
        # Checking memoStoragePath caused false positives: any two "skip memo"
        # uploads both have memoStoragePath="" which incorrectly matched.
        subject_ref      = _subject_doc_ref(school_id, subject)
        subject_snap     = subject_ref.get()
        existing_uploads = (
            subject_snap.to_dict().get("uploads", [])
            if subject_snap.exists else []
        )

        new_exam_path = data.get("examStoragePath", "")
        if new_exam_path:
            for u in existing_uploads:
                if u.get("examStoragePath") == new_exam_path:
                    print(f"[Upload] Duplicate detected: {new_exam_path}")
                    return jsonify({"examId": u.get("examId"), "duplicate": True})

        record = {
            "examId":           exam_id,
            "uploadedBy":       uid,
            "teacherName":      data.get("teacherName", "Teacher"),
            "schoolId":         school_id,
            "schoolName":       data.get("schoolName", school_id),
            "schoolFolder":     data.get("schoolFolder", school_id),
            "title":            data.get("title", ""),
            "year":             data.get("year", ""),
            "subject":          subject,
            "curriculum":       data.get("curriculum", "CAPS"),
            "grade":            data.get("grade", ""),
            "examDuration":     data.get("examDuration", 0),
            "examFileType":     data.get("examFileType", ""),
            "memoFileType":     data.get("memoFileType", ""),
            "examFileName":     data.get("examFileName", ""),
            "memoFileName":     data.get("memoFileName", ""),
            "examStorageUrl":   data.get("examStorageUrl", ""),
            "memoStorageUrl":   data.get("memoStorageUrl", ""),
            "examStoragePath":  data.get("examStoragePath", ""),
            "memoStoragePath":  data.get("memoStoragePath", ""),
            "aiMarkingOnly":    data.get("aiMarkingOnly", False),
            "status":           "pending_extraction",
            "questionsExtracted": False,
            "memoMerged":       False,
            "uploadedAt":       now.isoformat(),
            "extractedAt":      None,
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

        print(f"[Upload] Created {exam_id} for school {school_id}")

        # ── Trigger extraction immediately ────────────────────────────────
        # Don't rely solely on the Firestore listener — trigger directly so
        # the exam appears in the audit trail as fast as possible.
        threading.Thread(
            target=_launch_pipeline,
            args=(exam_id, record, school_id, subject),
            daemon=True,
        ).start()

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
        tier_id    = school_doc.to_dict().get("tier", "free") if school_doc.exists else "free"
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
        exam_id    = (data.get("exam_id") or data.get("examId") or "").strip()
        student_id = data.get("student_id", "anonymous")

        if not exam_id:
            return jsonify({"error": "exam_id required"}), 400

        meta, questions = _load_exam(exam_id)
        if meta is None:
            return jsonify({"error": f"Exam '{exam_id}' not found"}), 404

        if not questions:
            return jsonify({"error": (
                f"This exam has no questions yet (status: {meta.get('status', 'unknown')}). "
                "Extraction may still be running — please wait and try again."
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
        answers              = session.get("answers", {})
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

            # Resolve MCQ options dict for display
            options = q.get("options")
            if isinstance(options, list) and options and isinstance(options[0], dict):
                options = {o["key"]: o["value"] for o in options}

            # Mark: rule-based first, AI fallback when memo absent/inconclusive
            marked = mark_with_memo(student_ans, memo, marks)
            if marked is None:
                marked = mark_with_ai(q.get("question", ""), student_ans,
                                      marks, subject, memo)

            earned       = float(marked.get("score", 0))
            total_score += earned

            correct_display = memo if memo else "Not available"
            if memo and q_type == "mcq" and isinstance(options, dict):
                letter          = str(memo).strip().upper()
                correct_display = (
                    f"{letter}. {options.get(letter, '')}"
                    if letter in options else letter
                )

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
            })

        percentage = round(total_score / total_marks * 100, 1) if total_marks else 0
        feedback   = generate_final_feedback(percentage, results, subject)
        analysis   = generate_exam_analysis(subject, percentage,
                                            total_score, total_marks, results)

        print(f"[Submit] ✓ {total_score}/{total_marks} = {percentage}%")
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
        db.collection("exam_autosaves").document(f"{exam_id}_{student_id}").set(
            {"examId": exam_id, "studentId": student_id,
             "answers": answers, "updatedAt": fs_admin.SERVER_TIMESTAMP},
            merge=True,
        )
        return jsonify({"success": True})
    except Exception as e:
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
            updated.append({
                "idx":      i,
                "earned":   marked.get("score", 0),
                "status":   marked.get("status", "incorrect"),
                "feedback": marked.get("feedback", ""),
            })
        return jsonify({"results": updated})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


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
            subjects = ", ".join(learning_profile.get("subjects", ["Unknown"]))
        except Exception:
            subjects = "Unknown"

        try:
            weak_areas = json.dumps([
                {"question":   w.get("question") or w.get("key", ""),
                 "timesWrong": w.get("timesWrong") or w.get("count", 0),
                 "type":       w.get("type", "")}
                for w in learning_profile.get("weakAreas", [])[:8]
                if isinstance(w, dict)
            ])
        except Exception:
            weak_areas = "[]"

        try:
            latest_qs = json.dumps([
                {"q":      r.get("question_number"),
                 "status": r.get("status"),
                 "topic":  r.get("question", "")[:60]}
                for r in latest_attempt.get("markedResults", [])[:10]
            ])
        except Exception:
            latest_qs = "[]"

        system = (
            f"You are NextGen Skills AI Academic Coach — a brilliant, patient South African "
            f"CAPS/NSC curriculum tutor. Teach through natural conversation.\n"
            f"RULES: Never give everything at once. After every response ask ONE question. "
            f"Keep responses SHORT (4-6 sentences). Break topics into steps — teach step 1 then WAIT.\n"
            f"Student: {student_id} | Subjects: {subjects} | "
            f"Average: {learning_profile.get('overallAverage','?')}% | "
            f"Weak areas: {weak_areas}"
        )
        user_ctx = (
            f"STUDENT MESSAGE: {student_message}\n"
            f"Latest exam: {latest_attempt.get('examTitle','N/A')} "
            f"({latest_attempt.get('percentage','?')}%)\n"
            f"Latest questions: {latest_qs}"
        )

        # Build conversation history for context
        messages = [{"role": "system", "content": system}]
        for item in (history[-10:] if isinstance(history, list) else []):
            if isinstance(item, dict) and item.get("role") in ("user", "assistant"):
                messages.append({"role": item["role"], "content": item.get("content", "")})
        messages.append({"role": "user", "content": user_ctx})

        # Use Groq directly for streaming-friendly chat; Gemini as fallback
        reply = ""
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            try:
                client     = Groq(api_key=groq_key)
                completion = client.chat.completions.create(
                    model=_resolve_groq_model(),
                    messages=messages,
                    temperature=0.4,
                    max_tokens=600,
                )
                reply = completion.choices[0].message.content.strip()
            except Exception as e:
                print(f"[Chat] Groq failed: {e} — trying Gemini")

        if not reply:
            try:
                gemini_key = os.getenv("GEMINI_API_KEY")
                if gemini_key:
                    import google.generativeai as genai
                    genai.configure(api_key=gemini_key)
                    model = genai.GenerativeModel("gemini-2.0-flash",
                        system_instruction=system)
                    result = model.generate_content(user_ctx)
                    reply  = result.text.strip()
            except Exception as e:
                print(f"[Chat] Gemini also failed: {e}")
                reply = "I'm having trouble connecting right now. Please try again in a moment."

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
        return jsonify({
            "success":  False,
            "error":    str(e),
            "response": "I couldn't process your request right now.",
        }), 500


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
            print(f"[dashboard] attempts fetch failed: {e}")

        weak_map: dict = {}
        for attempt in attempts:
            for r in attempt.to_dict().get("markedResults", []):
                if r.get("status") == "correct":
                    continue
                qnum = str(r.get("question_number", ""))
                if not qnum:
                    continue
                if qnum not in weak_map:
                    weak_map[qnum] = {
                        "question_number": qnum,
                        "question_text":   r.get("question", ""),
                        "q_type":          r.get("type", "open"),
                        "wrong_count":     0,
                    }
                weak_map[qnum]["wrong_count"] += 1

        weak       = sorted(weak_map.values(),
                            key=lambda x: x["wrong_count"], reverse=True)[:20]
        study_plan = None
        try:
            plan_doc = db.collection("study_plans").document(student_id).get()
            if plan_doc.exists:
                pd         = plan_doc.to_dict()
                study_plan = {"plan":       pd.get("plan", ""),
                              "updated_at": str(pd.get("updatedAt", ""))}
        except Exception as e:
            print(f"[dashboard] study_plan: {e}")

        return jsonify({
            "student_id":      student_id,
            "weak":            weak,
            "study_plan":      study_plan,
            "session_history": [],
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/admin/extraction-status/<exam_id>", methods=["GET"])
def extraction_status(exam_id):
    try:
        doc = db.collection("exams").document(exam_id).get()
        if not doc.exists:
            return jsonify({"status": "not_found"}), 404
        d       = doc.to_dict()
        q_count = sum(
            1 for _ in db.collection("exam_questions")
                         .where("examId", "==", exam_id)
                         .stream()
        )
        return jsonify({
            "status":             d.get("status"),
            "title":              d.get("title"),
            "subject":            d.get("subject"),
            "questions_in_db":    q_count,
            "memo_merged":        d.get("memoMerged", False),
            "student_accessible": d.get("status") == "ready" and q_count > 0,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/admin/trigger-extract/<exam_id>", methods=["GET"])
def trigger_extract(exam_id):
    """
    Manually re-trigger extraction for a stuck or failed exam.
    Use when status is "error" or "pending_extraction" for too long.
    """
    try:
        meta         = None
        school_id    = "shared"
        subject_name = "General"

        exam_doc = db.collection("exams").document(exam_id).get()
        if exam_doc.exists:
            meta         = exam_doc.to_dict()
            school_id    = meta.get("schoolId", "shared")
            subject_name = meta.get("subject",  "General")
        else:
            # Search the subjects collection group
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
        _unmark_processing(exam_id)  # allow re-processing

        threading.Thread(
            target=run_extraction_pipeline,
            args=(exam_id, meta, school_id, subject_name),
            daemon=True,
        ).start()

        return jsonify({
            "ok":      True,
            "message": "Extraction started",
            "poll":    f"/admin/extraction-status/{exam_id}",
        })
    except Exception as e:
        traceback.print_exc()
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


# ══════════════════════════════════════════════════════════════════════════════
# STARTUP SEQUENCE
# ══════════════════════════════════════════════════════════════════════════════

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

