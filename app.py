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


# ═══════════════════════════════════════════════════════════════
# FIREBASE INITIALIZATION
# ═══════════════════════════════════════════════════════════════

import firebase_admin
from firebase_admin import credentials, firestore as fs_admin, storage



def _init_firebase():
    raw = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON", "").strip()

    if not raw:
        raise ValueError("FIREBASE_SERVICE_ACCOUNT_JSON is not set")

    # If it's a file path, read it
    if os.path.exists(raw):
        with open(raw) as f:
            cred_dict = json.load(f)
    else:
        cred_dict = json.loads(raw)  # assume raw JSON string

    cred = credentials.Certificate(cred_dict)
    firebase_admin.initialize_app(cred, {
        "storageBucket": os.environ.get("FIREBASE_STORAGE_BUCKET")
    })

_init_firebase()
db     = fs_admin.client()
bucket = storage.bucket()

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
    Returns dict with score, status, feedback.
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

    # Exact match
    if s_norm == m_norm:
        return {
            "score":       marks,
            "status":      "correct",
            "feedback":    "Correct.",
            "concept_gap": "",
        }

    sim = _similarity(s_norm, m_norm)

    # Very close (>=85%) — full marks
    if sim >= 0.85:
        return {
            "score":       marks,
            "status":      "correct",
            "feedback":    "Correct (closely matches memo).",
            "concept_gap": "",
        }

    # Partial (60-84%) — half marks
    if sim >= 0.60:
        partial = round(marks * 0.5, 1)
        return {
            "score":       partial,
            "status":      "partial",
            "feedback":    f"Partially correct ({int(sim*100)}% match). "
                           f"Memo: {memo_answer}",
            "concept_gap": "Incomplete or imprecise answer.",
        }

    # MCQ: single letter check
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

    # True/False check
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

    # Keyword check — award partial if key words present
    memo_words    = set(re.findall(r"\b\w{4,}\b", m_norm))
    student_words = set(re.findall(r"\b\w{4,}\b", s_norm))
    if memo_words:
        keyword_match = len(memo_words & student_words) / len(memo_words)
        if keyword_match >= 0.70:
            return {
                "score":       round(marks * keyword_match, 1),
                "status":      "partial",
                "feedback":    f"Contains key concepts but not complete. "
                               f"Memo: {memo_answer}",
                "concept_gap": "Missing some key points.",
            }
        if keyword_match >= 0.40:
            return {
                "score":       round(marks * 0.25, 1),
                "status":      "partial",
                "feedback":    f"Some relevant content but mostly incorrect. "
                               f"Memo: {memo_answer}",
                "concept_gap": "Significant gaps in understanding.",
            }

    return {
        "score":       0,
        "status":      "incorrect",
        "feedback":    f"Incorrect. Expected: {memo_answer}",
        "concept_gap": "Core concept not demonstrated.",
    }


def mark_with_ai(question: str, student_answer: str, marks: float,
                 subject: str, memo: str = "") -> dict:
    """
    AI marking fallback used when no memo is available.
    Uses Groq to assess answer quality against subject knowledge.
    Supports partial marks allocation.
    """
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    prompt = f"""You are an expert South African CAPS/NSC exam marker for {subject}.

Mark the student answer fairly and strictly. Award partial marks where deserved.

QUESTION: {question}
MARKS AVAILABLE: {marks}
MEMO/EXPECTED ANSWER: {memo if memo else "Not provided — use your subject expertise"}
STUDENT ANSWER: {student_answer if student_answer.strip() else "No answer provided"}

Instructions:
- Award full marks for complete correct answers
- Award partial marks (e.g. {round(marks*0.5,1)} for {marks} marks) for partially correct answers
- Award 0 for incorrect or missing answers
- Be specific in feedback — mention what was right and what was missing
- Concept gap: what key idea did the student miss?

Return ONLY valid JSON:
{{
  "score": <number between 0 and {marks}>,
  "status": "<correct|partial|incorrect|missing>",
  "feedback": "<specific examiner feedback>",
  "concept_gap": "<what was missing or misunderstood>",
  "model_answer": "<ideal answer>"
}}"""

    try:
        resp  = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=600,
        )
        raw   = resp.choices[0].message.content.strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            result = json.loads(match.group())
            # Clamp score to valid range
            result["score"] = max(0, min(float(result.get("score", 0)), marks))
            return result
    except Exception as e:
        print(f"[AI Mark] Failed: {e}")

    return {
        "score":        0,
        "status":       "incorrect",
        "feedback":     "Could not mark — AI unavailable.",
        "concept_gap":  "Unknown.",
        "model_answer": "",
    }


def generate_final_feedback(percentage: float, results: list,
                             subject: str) -> str:
    """Generate personalised exam feedback summary."""
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
            f"{', '.join(r['questionNumber'] for r in wrong[:8])}."
        )
    if partial:
        lines.append(
            f"Partially correct: "
            f"{', '.join(r['questionNumber'] for r in partial[:5])} — "
            f"expand your answers."
        )
    lines.append(f"Key concept gaps: {gap_summary}.")
    return " ".join(lines)


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
            data         = doc.to_dict() or {}
            school_id    = doc.reference.parent.parent.id
            subject_name = doc.id
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
                print(f"[Startup] Launching: {exam_id}")
                if _launch_pipeline(exam_id, upload, school_id, subject_name):
                    launched += 1
    except Exception as e:
        print(f"[Startup] Sweep error: {e}")
    print(f"[Startup] Launched {launched} missed extraction(s)")


_start_auto_extraction_listener()
_sweep_pending_on_startup()


# ═══════════════════════════════════════════════════════════════
# EXAM SESSION (in-memory, backed by Firestore)
# ═══════════════════════════════════════════════════════════════

def _save_session(sid: str, data: dict):
    db.collection("exam_sessions").document(sid).set(
        {**data, "createdAt": fs_admin.SERVER_TIMESTAMP}
    )


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


import concurrent.futures


def _load_exam(exam_id: str):
    """Load exam metadata + questions from Firestore."""

    def _fetch():
        exam_doc = db.collection("exams").document(exam_id).get()
        if not exam_doc.exists:
            return None, []
        meta = {**exam_doc.to_dict(), "id": exam_doc.id}

        raw_qs = list(
            db.collection("exam_questions")
            .where("examId", "==", exam_id)
            .stream()
        )
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
                "parent_context": d.get("parentContext"),
                "section": d.get("section", "A"),
                "question": d.get("questionText", ""),
                "type": d.get("type", "open").lower(),
                "options": options,
                "column_a": d.get("columnA"),
                "column_b": d.get("columnB"),
                "marks": d.get("marks", 1),
                "memo": d.get("memo", ""),
            })
        return meta, questions

    print(f"[_load_exam] fetching exam_id={exam_id}")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(_fetch)
        try:
            result = future.result(timeout=15)
            print(f"[_load_exam] success — {len(result[1])} questions")
            return result
        except concurrent.futures.TimeoutError:
            print("[_load_exam] TIMEOUT — Firestore unreachable after 15s")
            raise Exception("Database timeout — please try again")
        except Exception as e:
            print(f"[_load_exam] ERROR: {e}")
            raise


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
        sid        = data.get("session_id")
        student_id = data.get("student_id", "anonymous")
        session    = _get_session(sid)

        if not session:
            return jsonify({"error": "Invalid or expired session"}), 400

        subject   = session.get("subject", "General")
        questions = session.get("questions", [])
        answers   = session.get("answers", {})
        exam_id   = session.get("exam_id", "")

        total_score = 0.0
        total_marks = 0.0
        results     = []

        for i, q in enumerate(questions):
            q_num        = q.get("question_number", f"Q{i+1}")
            q_type       = q.get("type", "open").lower()
            marks        = float(q.get("marks") or 1)
            memo         = q.get("memo", "")
            student_ans  = answers.get(str(i), "").strip()
            total_marks += marks

            # Resolve options for MCQ display
            options = q.get("options")
            if isinstance(options, list) and options and isinstance(options[0], dict):
                options = {o["key"]: o["value"] for o in options}

            # ── Mark the answer ───────────────────────────────────────
            marked = mark_with_memo(student_ans, memo, marks)

            # No memo or no match — use AI
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

            # Format correct answer for display
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
            })

        percentage = round(total_score / total_marks * 100, 1) if total_marks else 0
        feedback   = generate_final_feedback(percentage, results, subject)

        # Save attempt to Firestore
        try:
            db.collection("exam_attempts").add({
                "examId":      exam_id,
                "studentId":   student_id,
                "subject":     subject,
                "score":       total_score,
                "total":       total_marks,
                "percentage":  percentage,
                "results":     results,
                "feedback":    feedback,
                "completedAt": fs_admin.SERVER_TIMESTAMP,
            })
        except Exception as e:
            print(f"[submit] Could not save attempt: {e}")

        _delete_session(sid)

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
        return jsonify({"error": str(e)}), 500


@app.route("/results/<exam_id>/<student_id>", methods=["GET"])
def get_results(exam_id, student_id):
    try:
        docs = list(
            db.collection("exam_attempts")
              .where("examId",    "==", exam_id)
              .where("studentId", "==", student_id)
              .order_by("completedAt", direction=fs_admin.Query.DESCENDING)
              .limit(1)
              .stream()
        )
        if not docs:
            return jsonify({"error": "Results not found"}), 404
        return jsonify({"success": True, "result": docs[0].to_dict()})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/autosave", methods=["POST"])
def autosave_exam():
    try:
        data       = request.get_json() or {}
        exam_id    = data.get("examId")
        student_id = data.get("studentId")
        answers    = data.get("answers", {})
        if not exam_id or not student_id:
            return jsonify({"error": "Missing examId or studentId"}), 400
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

@app.before_request
def handle_options():
    if request.method == "OPTIONS":
        response = app.make_default_options_response()
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response

if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)