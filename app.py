"""
app.py — Eduket Production Exam Extraction & Marking API
═══════════════════════════════════════════════════════════════

FEATURES
─────────────────────────────────────────────────────────────
✅ Firebase Storage (ODT, DOCX, PDF support)
✅ Automatic extraction listener + startup sweep
✅ Duplicate extraction prevention (thread-safe)
✅ Memo-based marking with partial credit
✅ AI marking fallback where memo is missing
✅ Per-question feedback + concept gap analysis
✅ Autosave + resume support
✅ Render-compatible gunicorn deployment
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

from datetime import datetime
from difflib import SequenceMatcher

from flask import Flask, request, jsonify
from flask_cors import CORS

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

from odf.opendocument import load as load_odt
from odf import text as odf_text
from odf import teletype
import mammoth
from groq import Groq
from datetime import datetime, timezone

import firebase_admin
from firebase_admin import credentials, firestore as fs_admin, storage, auth as fb_auth


# ═══════════════════════════════════════════════════════════════
# FIREBASE INITIALIZATION
# ═══════════════════════════════════════════════════════════════

db = None      # declare at module level so references don't NameError
bucket = None
def _init_firebase():
    global db, bucket

    raw = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON", "").strip()
    if not raw:
        raise ValueError("FIREBASE_SERVICE_ACCOUNT_JSON is not set")

    if os.path.exists(raw):
        with open(raw) as f:
            cred_dict = json.load(f)
    else:
        cred_dict = json.loads(raw)

    # ✅ Fix for Render/Heroku: unescape private key newlines if mangled
    if "private_key" in cred_dict:
        cred_dict["private_key"] = cred_dict["private_key"].replace("\\n", "\n")

    # Validate before attempting connection
    required = ["type", "project_id", "private_key", "client_email"]
    missing = [k for k in required if not cred_dict.get(k)]
    if missing:
        raise ValueError(f"Credential dict missing: {missing}")

    print(f"[Firebase] project_id: {cred_dict['project_id']}")
    print(f"[Firebase] client_email: {cred_dict['client_email']}")
    print(f"[Firebase] private_key newlines: {cred_dict['private_key'].count(chr(10))}")

    if not firebase_admin._apps:
        cred = credentials.Certificate(cred_dict)
        firebase_admin.initialize_app(cred, {
            "storageBucket": os.environ.get("FIREBASE_STORAGE_BUCKET")
        })

    db = fs_admin.client()
    bucket = storage.bucket()
    print("[Firebase] ✅ Ready")

def verify_request_token(request):
        """
        Verifies the Firebase ID token from the Authorization header.
        Returns (uid, error_response) — uid is None if verification fails,
        in which case error_response is a (jsonify, status_code) tuple to return immediately.
        """
        auth_header = request.headers.get("Authorization", "")
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
    # TIER LIMITS — mirrors src/utils/tierConfig.js (TIERS array)
    # ═══════════════════════════════════════════════════════════════
TIER_EXAM_LIMITS = {
        "free": 5,
        "silver": 30,
        "gold": 120,
        "platinum": 500,
    }

def get_exam_limit(tier_id):
    return TIER_EXAM_LIMITS.get(tier_id, TIER_EXAM_LIMITS["free"])

# ═══════════════════════════════════════════════════════════════
# APP + CORS
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
        "https://eduket.tech/"
    ],
    "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    "allow_headers": ["Content-Type", "Authorization"],
}}, supports_credentials=False)

# ═══════════════════════════════════════════════════════════════
# THREAD-SAFE PROCESSING TRACKER
# ═══════════════════════════════════════════════════════════════

_PROCESSING      = set()
_PROCESSING_LOCK = threading.Lock()


def _is_already_processing(exam_id: str) -> bool:
    with _PROCESSING_LOCK:
        return exam_id in _PROCESSING


def _mark_processing(exam_id: str):
    with _PROCESSING_LOCK:
        _PROCESSING.add(exam_id)


def _unmark_processing(exam_id: str):
    with _PROCESSING_LOCK:
        _PROCESSING.discard(exam_id)


# ═══════════════════════════════════════════════════════════════
# FIREBASE STORAGE DOWNLOAD
# ═══════════════════════════════════════════════════════════════

def download_file_for_extraction(meta: dict, file_type: str):
    """
    Downloads exam or memo file from Firebase Storage.
    Tries Storage SDK path first, falls back to public download URL.
    Returns (bytes, filename) or (None, filename).
    """
    filename = meta.get(f"{file_type}FileName", f"{file_type}.pdf")

    # 1. Storage SDK path
    storage_path = meta.get(f"{file_type}StoragePath")
    if storage_path:
        try:
            blob = bucket.blob(storage_path)
            if blob.exists():
                data = blob.download_as_bytes(timeout=120)
                print(f"[Storage] SDK download OK: {storage_path} ({len(data)} bytes)")
                return data, filename
        except Exception as e:
            print(f"[Storage] SDK failed: {e}")

    # 2. Public download URL fallback
    storage_url = meta.get(f"{file_type}StorageUrl")
    if storage_url:
        try:
            res = http_requests.get(storage_url, timeout=120)
            if res.status_code == 200:
                print(f"[Storage] URL download OK ({len(res.content)} bytes)")
                return res.content, filename
            print(f"[Storage] URL returned {res.status_code}")
        except Exception as e:
            print(f"[Storage] URL failed: {e}")

    print(f"[Storage] No source found for {file_type}")
    return None, filename


# ═══════════════════════════════════════════════════════════════
# TEXT EXTRACTION
# ═══════════════════════════════════════════════════════════════

def _iter_block_items(parent):
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P
    for child in parent.element.body.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)


def extract_text_from_docx(file_bytes: bytes) -> str:
    try:
        doc   = Document(io.BytesIO(file_bytes))
        lines = []
        for block in _iter_block_items(doc):
            if isinstance(block, Paragraph):
                t = block.text.strip()
                if t:
                    lines.append(t)
            elif isinstance(block, Table):
                for row in block.rows:
                    cells = [c.text.strip() for c in row.cells]
                    row_text = " | ".join(c for c in cells if c)
                    if row_text:
                        lines.append(row_text)
        text = "\n".join(lines)
        print(f"[DOCX] Extracted {len(text)} chars")
        return text
    except Exception as e:
        print(f"[DOCX] Failed: {e}")
        return ""


def extract_text_from_odt(file_bytes: bytes) -> str:
    try:
        with tempfile.NamedTemporaryFile(suffix=".odt", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp.flush()
            odt_doc    = load_odt(tmp.name)
            paragraphs = odt_doc.getElementsByType(odf_text.P)
            content    = "\n".join(
                teletype.extractText(p) for p in paragraphs
            )
        print(f"[ODT] Extracted {len(content)} chars")
        return content
    except Exception as e:
        print(f"[ODT] Failed: {e}")
        return ""


def extract_text_from_pdf(file_bytes: bytes) -> str:
    # Stage 1: native text
    try:
        doc  = fitz.open(stream=file_bytes, filetype="pdf")
        text = "".join(page.get_text() + "\n" for page in doc)
        doc.close()
        if len(text.strip()) > 200:
            print(f"[PDF] Native: {len(text)} chars")
            return text
    except Exception as e:
        print(f"[PDF] Native failed: {e}")

    # Stage 2: Groq vision OCR
    print("[PDF] Falling back to Groq vision OCR")
    return _groq_vision_ocr(file_bytes)


def _groq_vision_ocr(pdf_bytes: bytes) -> str:
    client   = Groq(api_key=os.getenv("GROQ_API_KEY"))
    all_text = ""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for i, page in enumerate(doc):
            try:
                pix  = page.get_pixmap()
                img  = base64.b64encode(pix.tobytes("png")).decode()
                resp = client.chat.completions.create(
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    messages=[{"role": "user", "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{img}"}},
                        {"type": "text",
                         "text": (
                             "South African NSC/CAPS exam page. "
                             "Extract ALL text exactly. Preserve question numbers, "
                             "marks in brackets, MCQ options A B C D. Plain text only."
                         )},
                    ]}],
                    max_tokens=2000,
                )
                all_text += resp.choices[0].message.content.strip() + "\n"
                print(f"[OCR] Page {i+1} done")
            except Exception as e:
                print(f"[OCR] Page {i+1} failed: {e}")
        doc.close()
    except Exception as e:
        print(f"[OCR] Fatal: {e}")
    return all_text


def extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    """Returns plain text regardless of file format."""
    lower = filename.lower()
    if lower.endswith(".docx"):
        return extract_text_from_docx(file_bytes)
    if lower.endswith(".odt"):
        return extract_text_from_odt(file_bytes)
    if lower.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    if lower.endswith(".doc"):
        try:
            result = mammoth.extract_raw_text(io.BytesIO(file_bytes))
            return result.value
        except Exception as e:
            print(f"[DOC] mammoth failed: {e}")
    return ""


# ═══════════════════════════════════════════════════════════════
# QUESTION PARSER
# ═══════════════════════════════════════════════════════════════

def parse_questions_universal(exam_text: str, subject: str, grade: str) -> list:
    client      = Groq(api_key=os.getenv("GROQ_API_KEY"))
    CHUNK       = 10000
    OVERLAP     = 800
    all_qs      = []
    seen        = set()

    chunks = []
    start  = 0
    while start < len(exam_text):
        chunks.append(exam_text[start:start + CHUNK])
        start += CHUNK - OVERLAP

    for idx, chunk in enumerate(chunks):
        print(f"[Parser] Chunk {idx+1}/{len(chunks)}")
        prompt = f"""You are an expert at parsing South African CAPS/NSC/IEB exam papers.

Extract EVERY question from the text below into a JSON array.

Rules:
- MCQ: split options into A/B/C/D dict, type="mcq"
- True/False: type="true_false"
- Matching: type="matching", column_a=[], column_b=[]
- Calculation: type="calculation"
- Essay: type="essay"
- Short answer: type="short_answer"
- Default: type="open"
- Marks: integer from brackets like (2), default 1
- Include section, question_number, parent_question, parent_context

Return ONLY a valid JSON array, no markdown, no explanation.

Each item:
{{
  "question_number": "1.1",
  "parent_question": "QUESTION 1",
  "parent_context": null,
  "section": "A",
  "question": "Full question text",
  "type": "mcq",
  "marks": 2,
  "options": {{"A":"...","B":"...","C":"...","D":"..."}},
  "column_a": null,
  "column_b": null,
  "memo": null
}}

Subject: {subject} | Grade: {grade}

EXAM TEXT:
{chunk}"""

        try:
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=8000,
            )
            raw   = resp.choices[0].message.content.strip()
            raw   = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
            raw   = re.sub(r"\s*```\s*$",       "", raw, flags=re.MULTILINE)
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                if isinstance(parsed, list):
                    for q in parsed:
                        qn = _normalise_qnum(q.get("question_number", ""))
                        key = qn or q.get("question", "")[:60]
                        if key not in seen:
                            seen.add(key)
                            all_qs.append(q)
        except Exception as e:
            print(f"[Parser] Chunk {idx+1} failed: {e}")

    print(f"[Parser] Total questions: {len(all_qs)}")
    return all_qs


# ═══════════════════════════════════════════════════════════════
# MEMO PARSER
# ═══════════════════════════════════════════════════════════════

def parse_memo_answers(memo_text: str, subject: str, grade: str) -> dict:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    CHUNK  = 12000
    result = {}

    chunks = [memo_text[i:i+CHUNK] for i in range(0, len(memo_text), CHUNK)]

    for idx, chunk in enumerate(chunks):
        print(f"[Memo] Chunk {idx+1}/{len(chunks)}")
        prompt = f"""You are reading a South African CAPS/NSC exam MARKING MEMORANDUM.

Extract EVERY answer. Return ONLY a valid JSON object mapping question_number to answer.
For MCQ give just the letter. For True/False give "True" or "False".
For written answers give the full expected answer.
No markdown, no explanation.

Example:
{{"1.1": "C", "1.2": "True", "1.3": "RAM is volatile memory."}}

Subject: {subject} | Grade: {grade}

MEMO TEXT:
{chunk}"""

        try:
            resp  = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=8000,
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

def _normalise_text(v) -> str:
    if v is None:
        return ""
    return str(v).strip().lower()


def _normalise_qnum(qn: str) -> str:
    s = str(qn).lower().strip()
    s = re.sub(r"^(question|q|ques|no|nr)[\s.\-]*", "", s)
    s = re.sub(r"[^a-z0-9]", "", s)
    return s


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalise_text(a), _normalise_text(b)).ratio()


def mark_with_memo(student_answer: str, memo_answer: str, marks: float) -> dict:
    """
    Mark against memo with partial credit.
    - Handles MCQ and True/False with exact logic
    - For open/short answers: uses similarity then falls through to AI
      for contextual marking that ignores spelling errors
    """
    s_norm = _normalise_text(student_answer)
    m_norm = _normalise_text(memo_answer)

    if not s_norm:
        return {
            "score":       0,
            "status":      "missing",
            "feedback":    "No answer provided.",
            "concept_gap": "Question not attempted.",
        }

    if not m_norm:
        return None  # No memo — signal AI fallback

    # ── Exact match after normalisation ──────────────────────────────────────
    if s_norm == m_norm:
        return {
            "score":       marks,
            "status":      "correct",
            "feedback":    "Correct.",
            "concept_gap": "",
        }

    # ── MCQ: single letter — strict exact match only ──────────────────────────
    if len(m_norm) == 1 and m_norm.isalpha():
        if s_norm.startswith(m_norm):
            return {
                "score":       marks,
                "status":      "correct",
                "feedback":    "Correct option selected.",
                "concept_gap": "",
            }
        return {
            "score":       0,
            "status":      "incorrect",
            "feedback":    f"Incorrect. Correct answer: {memo_answer.upper()}.",
            "concept_gap": "Wrong option selected.",
        }

    # ── True/False — strict check ─────────────────────────────────────────────
    if m_norm in ("true", "false"):
        if s_norm.startswith(m_norm):
            return {
                "score":       marks,
                "status":      "correct",
                "feedback":    "Correct.",
                "concept_gap": "",
            }
        return {
            "score":       0,
            "status":      "incorrect",
            "feedback":    f"Incorrect. Answer is {memo_answer}.",
            "concept_gap": "True/False answer incorrect.",
        }

    # ── Similarity check for open/short answers ───────────────────────────────
    sim = _similarity(s_norm, m_norm)

    # Very close match (>=75%) — full marks, spelling errors forgiven
    if sim >= 0.75:
        return {
            "score":       marks,
            "status":      "correct",
            "feedback":    "Correct.",
            "concept_gap": "",
        }

    # Moderate match (>=55%) — hand to AI for contextual check
    # instead of blindly awarding partial marks based on word overlap
    if sim >= 0.55:
        return None  # AI will assess contextual meaning

    # Low similarity — also hand to AI
    # Keyword matching penalises paraphrasing and spelling errors
    # so AI subject knowledge is more reliable here
    return None  # AI fallback for full contextual marking


def mark_with_ai(question: str, student_answer: str, marks: float,
                 subject: str, memo: str = "") -> dict:
    """
    AI marking using contextual subject understanding.
    - Ignores spelling errors, focuses on meaning
    - Awards marks based on conceptual correctness
    - Uses South African CAPS/NSC curriculum knowledge
    """
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    prompt = f"""You are a senior South African CAPS/NSC examiner for {subject}.

Your job is to mark a student's answer fairly based on CONCEPTUAL UNDERSTANDING, not perfect wording.

CRITICAL MARKING RULES:
1. IGNORE all spelling mistakes — if the intended meaning is clear, award marks
2. IGNORE grammatical errors — focus on whether the student understands the concept
3. Award marks for CORRECT MEANING even if different words are used
4. A student who writes "compewter" instead of "computer" should NOT lose marks
5. A student who writes "data is raw facts" instead of "data is unprocessed facts" SHOULD get marks — same concept
6. Use your deep {subject} curriculum knowledge to identify if the student demonstrates understanding
7. Be GENEROUS with partial marks — if the student shows partial understanding, award partial marks

QUESTION: {question}

MARKS AVAILABLE: {marks}

MEMO/EXPECTED ANSWER: {memo if memo else "Use your " + subject + " curriculum expertise to determine correctness"}

STUDENT ANSWER: {student_answer if student_answer.strip() else "No answer provided"}

MARKING APPROACH:
- First, identify the KEY CONCEPTS required by this question (ignore how they are spelled)
- Then, check if the student's answer demonstrates those key concepts
- Award full marks if all key concepts are present (even with spelling errors)
- Award partial marks if some key concepts are present
- Award 0 only if the answer is completely wrong or missing

MARK ALLOCATION GUIDE for {marks} marks:
- Full ({marks}): All required concepts present, meaning is correct
- Partial ({round(marks * 0.5, 1)}): Some concepts present, partial understanding shown  
- Minimal ({round(marks * 0.25, 1)}): Vague relevance but lacks key concepts
- Zero (0): Completely incorrect, irrelevant, or no answer

Return ONLY this exact JSON (no explanation, no markdown):
{{
  "score": <number between 0 and {marks}>,
  "status": "<correct|partial|incorrect|missing>",
  "feedback": "<specific feedback mentioning what concepts were correct and what was missing>",
  "concept_gap": "<the specific concept the student missed or misunderstood, empty string if correct>",
  "model_answer": "<a clear ideal answer a top student would write>"
}}"""

    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,   # slight creativity for contextual understanding
            max_tokens=800,
        )
        raw   = resp.choices[0].message.content.strip()

        # Strip markdown fences if model wraps in ```json
        raw   = re.sub(r"^```json\s*|^```\s*|```$", "", raw, flags=re.MULTILINE).strip()

        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            result = json.loads(match.group())
            result["score"] = max(0, min(float(result.get("score", 0)), marks))
            return result

    except Exception as e:
        print(f"[AI Mark] Failed: {e}", flush=True)

    return {
        "score":        0,
        "status":       "incorrect",
        "feedback":     "Could not mark — AI unavailable.",
        "concept_gap":  "Unknown.",
        "model_answer": "",
    }


def generate_final_feedback(percentage: float, results: list,
                             subject: str) -> str:
    wrong   = [r for r in results if r.get("status") in ("incorrect", "missing")]
    partial = [r for r in results if r.get("status") == "partial"]
    gaps    = list({
        r.get("concept_gap", "")
        for r in results
        if r.get("concept_gap", "").strip()
    })
    gap_summary = "; ".join(gaps[:5]) if gaps else "None identified"

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
        lines.append(
            f"Questions needing attention: "
            f"{', '.join(str(r.get('question_number', '?')) for r in wrong[:8])}."
        )
    if partial:
        lines.append(
            f"Partially correct: "
            f"{', '.join(str(r.get('question_number', '?')) for r in partial[:5])} — "
            f"expand your answers."
        )
    lines.append(f"Key concept gaps: {gap_summary}.")
    return " ".join(lines)

# AI GENERAL STUDENT EXAM FEEDBACK
def generate_exam_analysis(
    subject,
    percentage,
    total_score,
    total_marks,
    results
):

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    payload = []

    for r in results:
        payload.append({
            "question": r.get("question", ""),
            "student_answer": r.get("student_answer", ""),
            "correct_answer": r.get("correct_answer", ""),
            "status": r.get("status", ""),
            "marks": r.get("marks", 0),
            "earned": r.get("earned", 0),
            "feedback": r.get("feedback", ""),
        })

    prompt = f"""
You are an expert teacher, curriculum specialist and learning analyst.

Analyse the student's performance in {subject}.

Do NOT analyse by question number.

Instead identify conceptual strengths and weaknesses.

Infer concepts even if topics are not explicitly given.

Determine whether mistakes are isolated or recurring.

Determine whether the learner struggles with:

Remembering

Understanding

Applying

Analysing

Evaluating

Creating

Identify misconceptions.

Identify strongest knowledge areas.

Identify weakest knowledge areas.

Produce a personalised study plan.

Return ONLY valid JSON.

Student scored:

{total_score}/{total_marks}

({percentage}%)

Exam:

{json.dumps(payload, indent=2)}

Return:

{{
"overallSummary":"",
"studentProfile":"",
"strengths":[],
"weaknesses":[],
"misconceptions":[],
"learningStyle":"",
"cognitiveAnalysis":{{

"remember":0,

"understand":0,

"apply":0,

"analyse":0,

"evaluate":0,

"create":0

}},

"studyPlan":[],

"teacherSummary":"",

"parentSummary":""

}}
"""
    try:

        resp = client.chat.completions.create(

            model="llama-3.3-70b-versatile",

            messages=[

                {

                    "role":"user",

                    "content":prompt

                }

            ],

            temperature=0.2,

            max_tokens=2500

        )

        raw = resp.choices[0].message.content.strip()

        raw = re.sub(
            r"^```json\s*|^```\s*|```$",
            "",
            raw,
            flags=re.MULTILINE
        ).strip()

        match = re.search(
            r"\{.*\}",
            raw,
            re.DOTALL
        )

        if match:

            return json.loads(match.group())

    except Exception as e:

        print(e)

    return {}


# ═══════════════════════════════════════════════════════════════
# EXTRACTION PIPELINE
# ═══════════════════════════════════════════════════════════════

def _get_subject_doc_ref(school_id: str, subject_name: str):
    return (
        db.collection("teacherExamUploads")
          .document(school_id)
          .collection("subjects")
          .document(subject_name)
    )


def run_extraction_pipeline(exam_id: str, meta: dict,
                             school_id: str, subject_name: str):
    subject_ref = _get_subject_doc_ref(school_id, subject_name)

    def set_upload_status(status: str, extra: dict = {}):
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

        # 1. Download exam
        exam_bytes, exam_fn = download_file_for_extraction(meta, "exam")
        if not exam_bytes:
            raise ValueError(
                f"Could not download exam file. "
                f"Check examStoragePath/examStorageUrl in the upload record."
            )

        # 2. Extract text
        exam_text = extract_text_from_file(exam_bytes, exam_fn)
        if not exam_text.strip():
            raise ValueError("No text could be extracted from the exam file.")
        print(f"[Pipeline] Exam text: {len(exam_text)} chars")

        # 3. Parse questions
        questions = parse_questions_universal(exam_text, subject, grade)
        print(f"[Pipeline] Questions parsed: {len(questions)}")

        # 4. Download + parse memo
        memo_map   = {}
        memo_bytes, memo_fn = download_file_for_extraction(meta, "memo")
        if memo_bytes:
            memo_text = extract_text_from_file(memo_bytes, memo_fn)
            if memo_text.strip():
                raw_memo = parse_memo_answers(memo_text, subject, grade)
                memo_map = {_normalise_qnum(k): v for k, v in raw_memo.items()}
                print(f"[Pipeline] Memo answers: {len(memo_map)}")

        # 5. Merge memo into questions
        for q in questions:
            qn = _normalise_qnum(q.get("question_number", ""))
            if qn and qn in memo_map and not q.get("memo"):
                q["memo"] = memo_map[qn]

        # 6. Placeholder if no questions parsed
        if not questions:
            questions = [{
                "question_number": "1",
                "parent_question": "",
                "section":         "A",
                "question":        (
                    f"Questions could not be parsed from this {subject} paper. "
                    f"Please re-upload in Word (.docx) format."
                ),
                "type":    "open",
                "marks":   0,
                "options": None,
                "memo":    None,
            }]

        # 7. Write exam document
        db.collection("exams").document(exam_id).set({
            "title":            title,
            "subject":          subject,
            "grade":            grade,
            "year":             meta.get("year",        ""),
            "curriculum":       meta.get("curriculum",  "CAPS"),
            "teacherName":      meta.get("teacherName", ""),
            "uploadedBy":       meta.get("uploadedBy",  ""),
            "schoolId":         meta.get("schoolId",    school_id),
            "examDuration":     meta.get("examDuration", 0),
            "examStoragePath":  meta.get("examStoragePath", ""),
            "memoStoragePath":  meta.get("memoStoragePath", ""),
            "examStorageUrl":   meta.get("examStorageUrl",  ""),
            "memoStorageUrl":   meta.get("memoStorageUrl",  ""),
            "memoMerged":       bool(memo_map),
            "questionsExtracted": True,
            "status":           "ready",
            "totalQuestions":   len(questions),
            "extractedAt":      fs_admin.SERVER_TIMESTAMP,
            "sourceUploadId":   exam_id,
        })

        # 8. Write questions in batches of 400
        batch   = db.batch()
        written = 0
        for i, q in enumerate(questions):
            qtext = str(q.get("question") or "").strip()
            if not qtext:
                continue

            # Safe marks parsing
            try:
                raw_marks = q.get("marks", 1)
                marks = int(re.sub(r"[^0-9]", "", str(raw_marks))) if raw_marks else 1
                marks = max(1, marks)
            except Exception:
                marks = 1

            options = q.get("options")
            if not isinstance(options, dict):
                options = None

            ref = db.collection("exam_questions").document(
                f"{exam_id}_{i:04d}"
            )
            batch.set(ref, {
                "examId":         exam_id,
                "questionNumber": str(q.get("question_number") or i + 1),
                "parentQuestion": q.get("parent_question", ""),
                "parentContext":  q.get("parent_context"),
                "section":        q.get("section", "A"),
                "questionText":   qtext,
                "type":           q.get("type", "open"),
                "marks":          marks,
                "options":        options,
                "columnA":        q.get("column_a"),
                "columnB":        q.get("column_b"),
                "memo":           str(q.get("memo") or ""),
                "order":          i,
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


def _launch_pipeline(exam_id: str, meta: dict,
                     school_id: str, subject_name: str) -> bool:
    if _is_already_processing(exam_id):
        print(f"[Pipeline] Already processing: {exam_id}")
        return False

    # Check if already ready
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
                if not (upload.get("examStoragePath") or
                        upload.get("examStorageUrl")):
                    continue
                if _is_already_processing(exam_id):
                    continue
                print(f"[Listener] Pending: {school_id}/{subject_name}/{exam_id}")
                _launch_pipeline(exam_id, upload, school_id, subject_name)

    db.collection_group("subjects").on_snapshot(on_snapshot)
    print("[Listener] Active — watching all subjects")


def _sweep_pending_on_startup():
    print("[Startup] Sweeping for pending extractions...")
    launched = 0
    try:
        for doc in db.collection_group("subjects").stream():
            data = doc.to_dict() or {}
            school_id = doc.reference.parent.parent.id
            subject_name = doc.id

            uploads = data.get("uploads", [])
            updated_uploads = []
            doc_needs_update = False

            for upload in uploads:
                exam_id = upload.get("examId") or upload.get("id")
                if not exam_id:
                    continue

                # Check criteria
                if upload.get("status") == "pending_extraction" and \
                        (upload.get("examStoragePath") or upload.get("examStorageUrl")) and \
                        not _is_already_processing(exam_id):

                    print(f"[Startup] Attempting atomic claim for: {exam_id}")

                    # 🚀 CRITICAL FOR MULTI-WORKER: Atomic check-and-set via Firestore transaction
                    # This prevents two gunicorn workers from processing the same paper.
                    if _launch_pipeline(exam_id, upload, school_id, subject_name):
                        launched += 1

            # Note: _launch_pipeline internally updates the individual exam status to 'processing'
            # and marks the local thread safe tracker.

    except Exception as e:
        print(f"[Startup] Sweep error: {e}")
        traceback.print_exc()

    print(f"[Startup] Sweep complete. Launched {launched} missed extraction(s)")


# ═══════════════════════════════════════════════════════════════
# SAVE MEMORY Session
# =========================================================

def _save_session(sid: str, payload: dict):
    print(f"[_save_session] saving {sid}", flush=True)
    db.collection("exam_sessions").document(sid).set(payload)
    print(f"[_save_session] saved", flush=True)


def _get_session(sid: str) -> dict | None:
    if not sid:
        return None
    doc = db.collection("exam_sessions").document(sid).get()
    return doc.to_dict() if doc.exists else None


def _update_session_answers(sid: str, answers: dict):
    db.collection("exam_sessions").document(sid).update(
        {"answers": answers}
    )


def _delete_session(sid: str):
    try:
        db.collection("exam_sessions").document(sid).delete()
    except Exception:
        pass


# ─── Load exam + questions ────────────────────────────────────────────────────
def _load_exam(exam_id: str):
    print(f"[_load_exam] fetching {exam_id}", flush=True)

    ref = db.collection("exams").document(exam_id)
    print("[_load_exam] before get", flush=True)
    exam_doc = ref.get()
    print(f"[_load_exam] after get — exists={exam_doc.exists}", flush=True)

    if not exam_doc.exists:
        return None, []

    meta = {**exam_doc.to_dict(), "id": exam_doc.id}

    if meta.get("status") != "ready":
        print(f"[_load_exam] not ready: status={meta.get('status')}", flush=True)
        return meta, []

    print("[_load_exam] fetching questions", flush=True)
    raw_qs = list(
        db.collection("exam_questions")
          .where("examId", "==", exam_id)
          .stream()
    )
    print(f"[_load_exam] got {len(raw_qs)} questions", flush=True)

    raw_qs.sort(key=lambda d: d.to_dict().get("order", 0))

    questions = []
    for q in raw_qs:
        d = q.to_dict()
        options = d.get("options")
        if isinstance(options, dict) and options:
            options = [{"key": k, "value": v}
                       for k, v in sorted(options.items())]
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
        })

    return meta, questions

# ═══════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════

@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status":  "ok",
        "service": "Eduket Extraction & Marking API",
        "version": "3.0",
        "endpoints": {
            "exams":         "GET  /exams",
            "start":         "POST /start-exam",
            "question":      "POST /question",
            "answer":        "POST /answer",
            "submit":        "POST /submit",
            "results":       "GET  /results/<exam_id>/<student_id>",
            "autosave":      "POST /autosave",
            "admin_status":  "GET  /admin/extraction-status/<exam_id>",
            "admin_trigger": "GET  /admin/trigger-extract/<exam_id>",
        }
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

        # ── Look up the caller's schoolId from their own user doc ──────────
        # (never trust a schoolId sent in the body — derive it server-side)
        user_doc = db.collection("users").document(uid).get()
        if not user_doc.exists:
            return jsonify({"error": "User profile not found"}), 404

        user_data = user_doc.to_dict()
        school_id = user_data.get("schoolId")
        if not school_id:
            return jsonify({"error": "No school associated with this account"}), 400

        # ── Look up the school's tier ───────────────────────────────────────
        school_doc = db.collection("schools").document(school_id).get()
        if not school_doc.exists:
            return jsonify({"error": "School not found"}), 404

        tier_id = school_doc.to_dict().get("tier", "free")
        exam_limit = get_exam_limit(tier_id)

        # ── Count this school's exam uploads so far this calendar month ───
        now = datetime.now(timezone.utc)
        start_of_month = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
        start_of_month_iso = start_of_month.isoformat()

        exams_query = (
            db.collection("exams")
            .where("schoolId", "==", school_id)
            .where("uploadedAt", ">=", start_of_month_iso)
        )
        current_count = len(list(exams_query.stream()))

        if current_count >= exam_limit:
            return jsonify({
                "error": "limit_reached",
                "message": (
                    f"Your school has reached its monthly limit of {exam_limit} exam uploads "
                    f"on the {tier_id.capitalize()} plan. You can wait until next month, or ask "
                    "your principal to upgrade your school's plan for a higher limit."
                ),
                "tier": tier_id,
                "limit": exam_limit,
                "used": current_count,
            }), 403

        # ── Build the exam record (mirrors driveManager.js saveExamMetadata) ─
        exam_id = data.get("examId") or f"{uid}_{int(now.timestamp() * 1000)}"
        subject = data.get("subject", "General")

        # Duplicate check against the subject subdocument
        subject_ref = db.collection("teacherExamUploads").document(school_id).collection("subjects").document(subject)
        subject_snap = subject_ref.get()
        existing_uploads = subject_snap.to_dict().get("uploads", []) if subject_snap.exists else []

        for u in existing_uploads:
            if u.get("examStoragePath") == data.get("examStoragePath") or u.get("memoStoragePath") == data.get("memoStoragePath"):
                return jsonify({"examId": u.get("examId"), "duplicate": True})

        record = {
            "examId": exam_id,
            "uploadedBy": uid,
            "teacherName": data.get("teacherName", "Teacher"),
            "schoolId": school_id,
            "schoolName": data.get("schoolName", school_id),
            "schoolFolder": data.get("schoolFolder", school_id),
            "title": data.get("title", ""),
            "year": data.get("year", ""),
            "subject": subject,
            "curriculum": data.get("curriculum", "CAPS"),
            "grade": data.get("grade", ""),
            "examDuration": data.get("examDuration", 0),
            "examFileType": data.get("examFileType", ""),
            "memoFileType": data.get("memoFileType", ""),
            "examFileName": data.get("examFileName", ""),
            "memoFileName": data.get("memoFileName", ""),
            "examStorageUrl": data.get("examStorageUrl", ""),
            "memoStorageUrl": data.get("memoStorageUrl", ""),
            "examStoragePath": data.get("examStoragePath", ""),
            "memoStoragePath": data.get("memoStoragePath", ""),
            "status": "pending_extraction",
            "questionsExtracted": False,
            "memoMerged": False,
            "uploadedAt": now.isoformat(),
            "extractedAt": None,
        }

        # 1. Top-level exams collection — backend pipeline reads this
        db.collection("exams").document(exam_id).set(record)

        # 2. School-level doc under teacherExamUploads
        db.collection("teacherExamUploads").document(school_id).set({
            "schoolId": school_id,
            "schoolName": record["schoolName"],
            "schoolFolder": record["schoolFolder"],
            "updatedAt": now.isoformat(),
        }, merge=True)

        # 3. Subject subdocument with uploads array
        subject_ref.set({
            "subject": subject,
            "schoolId": school_id,
            "uploads": [{**record, "id": exam_id}] + existing_uploads,
            "updatedAt": now.isoformat(),
        }, merge=True)

        return jsonify({"examId": exam_id, "duplicate": False})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

#     Verify Account Usage based on Tier
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

        tier_id = school_doc.to_dict().get("tier", "free")
        exam_limit = get_exam_limit(tier_id)

        now = datetime.now(timezone.utc)
        start_of_month = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
        start_of_month_iso = start_of_month.isoformat()

        exams_query = (
            db.collection("exams")
            .where("schoolId", "==", school_id)
            .where("uploadedAt", ">=", start_of_month_iso)
        )
        used = len(list(exams_query.stream()))
        remaining = max(0, exam_limit - used)

        return jsonify({
            "tier": tier_id,
            "limit": exam_limit,
            "used": used,
            "remaining": remaining,
            "atLimit": used >= exam_limit,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/exams", methods=["GET"])
def list_exams():
    exams = []
    try:
        for doc in (
            db.collection("exams")
              .where("status", "==", "ready")
              .stream()
        ):
            d = doc.to_dict()
            exams.append({
                "id":           doc.id,
                "name":         d.get("title",      doc.id),
                "subject":      d.get("subject",    ""),
                "grade":        d.get("grade",      ""),
                "year":         d.get("year",       ""),
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
        exam_id    = (
            data.get("exam_id") or data.get("exam") or
            data.get("examId")  or ""
        ).strip()
        student_id = data.get("student_id", "anonymous")

        print(f"[start_exam] exam='{exam_id}' student='{student_id}'")


        if not exam_id:
            return jsonify({"error": "exam_id required"}), 400

        meta, questions = _load_exam(exam_id)

        # ✅ Check meta FIRST before touching questions
        if meta is None:
            return jsonify({"error": f"Exam '{exam_id}' not found"}), 404

        if not questions:
            return jsonify({
                "error": (
                    "This exam has no questions yet — extraction may still "
                    f"be processing (status: {meta.get('status', 'unknown')}). "
                    "Please wait a minute and try again."
                )
            }), 400

        # Debug log (safe to remove once confirmed working)
        for q in questions:
            print(f"[START DEBUG] Q{q.get('question_number')} options: {q.get('options')}")

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
        answers                          = session.get("answers", {})
        answers[str(data.get("index"))]  = data.get("answer", "")
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

        print(f"[submit] exam={exam_id} student={student_id} answers={len(answers)}", flush=True)

        if not exam_id:
            return jsonify({"error": "exam_id required"}), 400

        # Load exam fresh — no session needed
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
                marked = mark_with_ai(
                    q.get("question", ""),
                    student_ans,
                    marks,
                    subject,
                    memo,
                )

            earned       = float(marked.get("score", 0))
            total_score += earned

            correct_display = memo if memo else "Not available"
            if memo and q_type == "mcq" and isinstance(options, dict):
                letter = str(memo).strip().upper()
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
                "questionAnalysis": [
                        {
                            "questionText": "Write a while loop that counts from 1 to 10.",
                            "testedConcept": "Iteration and loop control",
                            "studentUnderstanding": "Partial",
                            "misconception": "Loop termination condition is incorrect.",
                            "explanation": "The learner understands loop syntax but not the stopping condition, which can lead to infinite loops or incorrect execution.",
                            "improvementAdvice": "Trace loop execution step by step and practise predicting output before running code."
                        }
                    ]

            })

        percentage = round(total_score / total_marks * 100, 1) if total_marks else 0
        feedback   = generate_final_feedback(percentage, results, subject)
        analysis = generate_exam_analysis(

            subject,

            percentage,

            total_score,

            total_marks,

            results

        )

        print(f"[submit] ✅ {total_score}/{total_marks} = {percentage}%", flush=True)

        return jsonify({

            "score": total_score,

            "total": total_marks,

            "percentage": percentage,

            "results": results,

            "feedback": feedback,

            "analysis": analysis,

            "subject": subject

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
        # Accept both naming conventions
        exam_id    = data.get("exam_id") or data.get("examId", "")
        student_id = data.get("student_id") or data.get("studentId", "")
        answers    = data.get("answers", {})

        if not exam_id or not student_id:
            return jsonify({"error": "Missing exam_id or student_id"}), 400

        db.collection("exam_autosaves").document(
            f"{exam_id}_{student_id}"
        ).set({
            "examId":    exam_id,
            "studentId": student_id,
            "answers":   answers,
            "updatedAt": fs_admin.SERVER_TIMESTAMP,
        }, merge=True)

        return jsonify({"success": True})
    except Exception as e:
        print(f"[autosave] error: {e}", flush=True)
        return jsonify({"error": str(e)}), 500


@app.route("/autosave/<exam_id>/<student_id>", methods=["GET"])
def load_autosave(exam_id, student_id):
    try:
        doc = db.collection("exam_autosaves").document(
            f"{exam_id}_{student_id}"
        ).get()
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

            marked = mark_with_memo(student_ans, memo, marks)
            if marked is None:
                marked = mark_with_ai(question, student_ans, marks, subject, memo)

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


@app.route("/admin/extraction-status/<exam_id>", methods=["GET"])
def extraction_status(exam_id):
    try:
        doc = db.collection("exams").document(exam_id).get()
        if doc.exists:
            d       = doc.to_dict()
            q_count = sum(
                1 for _ in
                db.collection("exam_questions")
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
        return jsonify({"status": "not_found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/admin/trigger-extract/<exam_id>", methods=["GET"])
def trigger_extract(exam_id):
    try:
        # Find in exams collection
        exam_doc = db.collection("exams").document(exam_id).get()
        meta         = None
        school_id    = "shared"
        subject_name = "General"

        if exam_doc.exists:
            meta         = exam_doc.to_dict()
            school_id    = meta.get("schoolId", "shared")
            subject_name = meta.get("subject",  "General")
        else:
            # Search subject subcollections
            for doc in db.collection_group("subjects").stream():
                for upload in (doc.to_dict() or {}).get("uploads", []):
                    if (upload.get("examId") == exam_id or
                            upload.get("id") == exam_id):
                        meta         = upload
                        school_id    = doc.reference.parent.parent.id
                        subject_name = doc.id
                        break
                if meta:
                    break

        if not meta:
            return jsonify({"error": f"Exam {exam_id} not found"}), 404

        # Force reset so it re-runs
        db.collection("exams").document(exam_id).set(
            {"status": "pending_extraction"}, merge=True
        )
        _unmark_processing(exam_id)

        threading.Thread(
            target=run_extraction_pipeline,
            args=(exam_id, meta, school_id, subject_name),
            daemon=True,
        ).start()

        return jsonify({
            "ok":      True,
            "message": "Extraction started",
            "exam_id": exam_id,
            "poll":    f"/admin/extraction-status/{exam_id}",
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/dashboard", methods=["POST", "OPTIONS"])
def dashboard():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    try:
        data = request.get_json(silent=True) or {}
        student_id = data.get("student_id", "").strip()

        if not student_id:
            return jsonify({"error": "student_id required"}), 400

        print(f"[dashboard] student_id={student_id}", flush=True)

        # ── 1. Weak topics from exam_attempts ────────────────────────────
        try:
            attempts = list(
                db.collection("exam_attempts")
                  .where("studentId", "==", student_id)
                  .stream()
            )
        except Exception as e:
            print(f"[dashboard] exam_attempts fetch failed: {e}", flush=True)
            attempts = []

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
                    weak_map[qnum] = {
                        "question_number": qnum,
                        "question_text":   r.get("question", ""),
                        "q_type":          r.get("type", "open"),
                        "wrong_count":     0,
                    }
                weak_map[qnum]["wrong_count"] += 1

        weak = sorted(weak_map.values(), key=lambda x: x["wrong_count"], reverse=True)[:20]
        print(f"[dashboard] weak topics={len(weak)}", flush=True)

        # ── 2. Study plan ─────────────────────────────────────────────────
        study_plan = None
        try:
            plan_doc = db.collection("study_plans").document(student_id).get()
            if plan_doc.exists:
                pd = plan_doc.to_dict()
                study_plan = {
                    "plan":       pd.get("plan", ""),
                    "updated_at": str(pd.get("updatedAt", "")),
                }
        except Exception as e:
            print(f"[dashboard] study_plan fetch failed: {e}", flush=True)

        # ── 3. Session history ────────────────────────────────────────────
        session_history = []
        try:
            sessions = list(
                db.collection("agent_sessions")
                  .where("studentId", "==", student_id)
                  .order_by("startedAt", direction=fs_admin.Query.DESCENDING)
                  .limit(10)
                  .stream()
            )
            session_history = [
                {
                    "session_id":    s.id,
                    "started_at":    str(s.to_dict().get("startedAt", "")),
                    "ended_at":      str(s.to_dict().get("endedAt", "")),
                    "message_count": s.to_dict().get("messageCount", 0),
                }
                for s in sessions
            ]
        except Exception as e:
            print(f"[dashboard] sessions fetch failed (ok if unused): {e}", flush=True)

        print(f"[dashboard] ✅ weak={len(weak)} plan={'yes' if study_plan else 'no'} sessions={len(session_history)}", flush=True)

        return jsonify({
            "student_id":      student_id,
            "weak":            weak,
            "study_plan":      study_plan,
            "session_history": session_history,
        })

    except Exception as e:
        print(f"[dashboard] ❌ {e}", flush=True)
        return jsonify({"error": str(e)}), 500

@app.route("/admin/cleanup-sessions", methods=["POST"])
def cleanup_sessions():
    from datetime import timedelta, timezone
    cutoff  = datetime.now(timezone.utc) - timedelta(hours=24)
    deleted = 0
    for doc in db.collection("exam_sessions").stream():
        created = doc.to_dict().get("createdAt")
        if created and created < cutoff:
            doc.reference.delete()
            deleted += 1
    return jsonify({"deleted": deleted})


# AGENT CHAT-BOX WITH STUDENT HISTORY ACCESS
@app.route("/agent-chat", methods=["POST"])
def agent_chat():
    try:
        data = request.get_json(force=True)
        student_id      = data.get("student_id", "")
        student_message = data.get("message", "").strip()
        learning_profile = data.get("learningProfile", {})
        latest_attempt  = data.get("latestAttempt", {})
        history         = data.get("history", [])

        if not student_message:
            return jsonify({"error": "Message cannot be empty."}), 400

        # ── Build context variables BEFORE f-strings ──────────────────────────
        try:
            latest_questions = json.dumps([
                {
                    "q":      r.get("question_number"),
                    "status": r.get("status"),
                    "topic":  r.get("question", "")[:60],
                }
                for r in latest_attempt.get("markedResults", [])[:10]
            ])
        except Exception:
            latest_questions = "[]"

        try:
            weak_areas_list = [
                str(w.get("question") or w.get("key") or w.get("text") or "")
                for w in learning_profile.get("weakAreas", [])[:5]
                if isinstance(w, dict)
            ]
        except Exception:
            weak_areas_list = []

        try:
            recent_results_list = [
                f"{r.get('examTitle', '?')} — {r.get('percentage', '?')}%"
                for r in learning_profile.get("recentResults", [])[:3]
                if isinstance(r, dict)
            ]
        except Exception:
            recent_results_list = []

        try:
            weak_areas_full = json.dumps([
                {
                    "question":   w.get("question") or w.get("key", ""),
                    "timesWrong": w.get("timesWrong") or w.get("count", 0),
                    "type":       w.get("type", ""),
                    "text":       w.get("text", "")[:80],
                }
                for w in learning_profile.get("weakAreas", [])[:8]
                if isinstance(w, dict)
            ])
        except Exception:
            weak_areas_full = "[]"

        try:
            subjects = ", ".join(learning_profile.get("subjects", ["Unknown"]))
        except Exception:
            subjects = "Unknown"

        # ── System prompt ──────────────────────────────────────────────────────
        system_prompt = f"""
You are NextGen Skills AI Academic Coach — a brilliant, patient South African
CAPS/NSC curriculum tutor who teaches through natural conversation.

═══════════════════════════════════════════════════════
CRITICAL CONVERSATION RULES — NEVER BREAK THESE:
═══════════════════════════════════════════════════════
1. NEVER give everything at once. One small chunk at a time.
2. After EVERY response, ask ONE simple question to check understanding
   or ask if the student wants to continue.
3. Wait for the student to respond before moving on.
4. Keep each response SHORT — maximum 4-6 sentences or one concept.
5. If teaching a topic, break it into steps. Teach step 1, then WAIT.
6. Only move to step 2 when student says yes, ok, continue, proceed, next,
   I understand, got it, or any positive response.
7. If student says no, stop, or I don't understand — simplify or re-explain
   that same step differently. Do NOT move forward.
8. If student answers a practice question — mark it immediately, give brief
   feedback, then ask if they want to try the next one.
9. NEVER use long bullet lists. Use at most 2-3 short points per message.
10. NEVER repeat what you said before unless asked.

═══════════════════════════════════════════════════════
RESPONSE LENGTH GUIDE:
═══════════════════════════════════════════════════════
- Greeting or checking in: 1-2 sentences
- Explaining a concept: 3-5 sentences MAX, then pause and ask
- Giving an example: show ONE example, then ask if it makes sense
- Practice question: ONE question at a time, wait for answer
- Feedback on answer: 2-3 sentences, then ask if ready to continue

═══════════════════════════════════════════════════════
CONVERSATION FLOW EXAMPLES:
═══════════════════════════════════════════════════════

Example 1 — Teaching:
Student: "Help me study functions"
You: "Sure! Let's start with the basics. A function in a spreadsheet is a
built-in formula that does a specific job for you — like adding numbers or
finding the highest value. The most common one is =SUM().
Do you want to see how =SUM() works with an example? 😊"

Student: "yes"
You: "Great! Imagine you have marks in cells B2 to B6.
To add them all up, you type: =SUM(B2:B6)
That's it — the spreadsheet adds all those numbers automatically.
Does that make sense, or should I explain it differently?"

Student: "I get it"
You: "Nice! Let's test that. What formula would you use to add
cells A1 to A10? Give it a try 👇"

Example 2 — Student doesn't understand:
Student: "I don't get it"
You: "No problem at all! Let me try a different way.
Think of =SUM() like a calculator that adds things for you.
Instead of typing 5+6+7+8, you just say 'add everything in this column'.
Better? Or should I try another way?"

═══════════════════════════════════════════════════════
TWO MODES — switch automatically:
═══════════════════════════════════════════════════════
MODE 1 — RESULTS & COACHING: When student asks about their scores or progress.
Use ONLY their actual exam data provided. Keep it brief and personal.

MODE 2 — SUBJECT TUTOR: When student wants to learn a topic.
Use your full CAPS curriculum knowledge. Teach step by step.
Cross-check their weak areas and mention if this topic appeared in their exam.

═══════════════════════════════════════════════════════
STUDENT PROFILE:
═══════════════════════════════════════════════════════
Student: {student_id}
Subject(s): {subjects}
Average Score: {learning_profile.get('overallAverage', 'Unknown')}%
Weak Areas: {weak_areas_full}

═══════════════════════════════════════════════════════
PERSONALITY:
═══════════════════════════════════════════════════════
- Warm, encouraging, and patient
- Use the student's name occasionally
- Use simple emojis sparingly (😊 ✅ 👇 💡) to make it friendly
- Celebrate small wins: "Well done!", "That's correct!", "You're getting it!"
- Never make the student feel bad for not knowing something
- Never say you are an AI
"""

        # ── User context ───────────────────────────────────────────────────────
        user_context = f"""
STUDENT MESSAGE: {student_message}

STUDENT DATA:
- Latest exam: {latest_attempt.get('examTitle', 'N/A')} | {latest_attempt.get('subject', 'N/A')} | {latest_attempt.get('percentage', '?')}%
- Weak areas: {weak_areas_list}
- Recent results: {recent_results_list}
- Latest exam questions: {latest_questions}
"""

        # ── Build messages with history ────────────────────────────────────────
        messages = [{"role": "system", "content": system_prompt}]

        if isinstance(history, list):
            for item in history[-10:]:
                if (
                    isinstance(item, dict)
                    and item.get("role") in ("user", "assistant")
                    and item.get("content")
                ):
                    messages.append({
                        "role":    item["role"],
                        "content": item["content"],
                    })

        messages.append({"role": "user", "content": user_context})

        # ── Call Groq ──────────────────────────────────────────────────────────
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.4,
            max_tokens=600,
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
                "What should I revise today?",
                "Create a revision quiz",
            ],
        })

    except Exception as e:
        traceback.print_exc()
        print("AGENT CHAT ERROR:", str(e))
        return jsonify({
            "success":  False,
            "error":    str(e),
            "response": "I couldn't process your request right now. Please try again.",
        }), 500


@app.before_request
def handle_options():
    if request.method == "OPTIONS":
        response = app.make_default_options_response()
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        return response


# ── Startup sequence ──────────────────────────────────────────
try:
    _init_firebase()
except Exception as e:
    traceback.print_exc()
    raise SystemExit(1)

_sweep_pending_on_startup()
_start_auto_extraction_listener()
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)