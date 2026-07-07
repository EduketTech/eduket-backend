"""
extraction_engine.py — v4 (render-first, images stored in Firebase)
─────────────────────────────────────────────────────────────────────
Every DOCX/ODT is converted to PDF by LibreOffice, rendered page-by-page
at 200 DPI, and each page image is uploaded to Firebase Storage.

Groq vision then reads each page and produces:
  - Structured question JSON
  - questionImageUrl  → page PNG URL (set when question has a visual element)
  - questionLatex     → LaTeX string for maths equations (render with MathJax)
  - questionTable     → markdown-format table for Accounting (render as HTML table)

This means students see the ACTUAL diagram/graph/circuit/table, not a
prose description, because the stored PNG is served directly in the exam UI.

─── SETUP ────────────────────────────────────────────────────────────────────
Render build command (Settings → Build Command):
  apt-get install -y libreoffice --no-install-recommends && pip install -r requirements.txt

─── app.py changes needed ────────────────────────────────────────────────────
1. Import:
     from extraction_engine import extract_questions_from_file, extract_text_from_file, parse_questions_universal

2. In run_extraction_pipeline, replace:
     exam_text = extract_text_from_file(exam_bytes, exam_fn)
     ...
     questions = parse_questions_universal(exam_text, subject, grade)
   with:
     questions = extract_questions_from_file(
         exam_bytes, exam_fn, subject, grade,
         exam_id=exam_id,
         school_folder=meta.get("schoolFolder", school_id)
     )

3. In the batch.set() for exam_questions, add:
     "questionImageUrl":  q.get("questionImageUrl"),
     "hasVisual":         q.get("has_visual", False),
     "questionLatex":     q.get("question_latex"),
     "questionTable":     q.get("question_table"),

─── Frontend changes needed (ExamDisplay / question renderer) ────────────────
For each question, render in order:
  1. If questionLatex:  <MathJax>{questionLatex}</MathJax>
  2. If questionTable:  parse markdown table → <table>
  3. If questionImageUrl: <img src={questionImageUrl} />
  4. Question text (may also contain [DIAGRAM: ...] for accessibility)
"""

from __future__ import annotations  # makes type hints Python 3.8+ compatible

import io
import os
import re
import json
import uuid
import base64
import shutil
import logging
import subprocess
import tempfile
from pathlib import Path

import fitz
import mammoth
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from odf.opendocument import load as load_odt
from odf import text as odf_text
from odf import teletype
from groq import Groq

logger = logging.getLogger(__name__)

_VISION_MODEL        = "meta-llama/llama-4-scout-17b-16e-instruct"
_PARSER_MODEL        = "llama-3.3-70b-versatile"
_PAGE_DPI            = 200          # 200 DPI = clear A4 page, good for diagrams
_MAX_TOKENS_PER_PAGE = 4096


# ══════════════════════════════════════════════════════════════════════════════
# 1. SUBJECT CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

_SUBJECT_MAP = {
    "accounting":    {"accounting", "acc", "financial accounting", "financial management"},
    "mathematics":   {"mathematics", "maths", "math", "mathematical literacy",
                      "maths lit", "calculus", "statistics", "pure maths"},
    "sciences":      {"physical sciences", "physics", "chemistry", "natural sciences",
                      "physical science"},
    "life_sciences": {"life sciences", "biology", "life science"},
    "geography":     {"geography", "geo"},
    "business":      {"business studies", "business", "economics", "entrepreneurship"},
    "language":      {"english", "afrikaans", "isizulu", "isixhosa", "setswana",
                      "sesotho", "language", "home language", "first additional"},
    "cat_it":        {"computer applications technology", "cat",
                      "information technology", "it"},
    "history":       {"history"},
}

def _cat(subject: str) -> str:
    s = subject.lower().strip()
    for cat, kw in _SUBJECT_MAP.items():
        if any(k in s for k in kw):
            return cat
    return "general"


# ══════════════════════════════════════════════════════════════════════════════
# 2. FIREBASE STORAGE — page image upload
# ══════════════════════════════════════════════════════════════════════════════

def _upload_page_image(school_folder: str, exam_id: str,
                        page_num: int, png_bytes: bytes) -> str | None:
    """
    Upload a rendered page PNG to Firebase Storage.
    Returns a download URL with token, or None on failure.
    Path: exam_pages/{schoolFolder}/{examId}/page_{nnn}.png
    """
    try:
        from firebase_admin import storage as fb_storage
        bucket = fb_storage.bucket()
        token  = str(uuid.uuid4())
        path   = f"exam_pages/{school_folder}/{exam_id}/page_{page_num:03d}.png"
        blob   = bucket.blob(path)
        blob.metadata = {"firebaseStorageDownloadTokens": token}
        blob.upload_from_string(png_bytes, content_type="image/png")
        blob.patch()
        bucket_name  = bucket.name
        encoded_path = path.replace("/", "%2F")
        url = (
            f"https://firebasestorage.googleapis.com/v0/b/{bucket_name}"
            f"/o/{encoded_path}?alt=media&token={token}"
        )
        logger.info("[Storage] Uploaded page %d → %s", page_num, path)
        return url
    except Exception as exc:
        logger.error("[Storage] Upload failed p%d: %s", page_num, exc)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 3. LIBREOFFICE — DOCX/ODT → PDF
# ══════════════════════════════════════════════════════════════════════════════

def _lo_available() -> bool:
    return bool(shutil.which("libreoffice") or shutil.which("soffice"))


def _lo_cmd() -> str:
    return shutil.which("libreoffice") or shutil.which("soffice") or "libreoffice"


def _convert_to_pdf(file_bytes: bytes, filename: str) -> bytes | None:
    """
    Convert DOCX/ODT → PDF using LibreOffice headless.
    LibreOffice renders fonts, OMML equations, embedded images, and table
    formatting exactly as Word/Calc would display them — this is the only
    reliable way to preserve complex document structure.
    Returns PDF bytes or None if LibreOffice unavailable.
    """
    if not _lo_available():
        logger.warning("[LibreOffice] Not installed — falling back to text path")
        return None

    with tempfile.TemporaryDirectory() as tmp:
        inp = os.path.join(tmp, filename)
        with open(inp, "wb") as f:
            f.write(file_bytes)
        try:
            subprocess.run(
                [_lo_cmd(), "--headless", "--convert-to", "pdf",
                 "--outdir", tmp, inp],
                check=True, timeout=120, capture_output=True,
            )
            pdf = os.path.join(tmp, Path(filename).stem + ".pdf")
            if os.path.exists(pdf):
                data = open(pdf, "rb").read()
                logger.info("[LibreOffice] %s → PDF (%d bytes)", filename, len(data))
                return data
            logger.error("[LibreOffice] PDF not produced for %s", filename)
        except subprocess.TimeoutExpired:
            logger.error("[LibreOffice] Timeout: %s", filename)
        except subprocess.CalledProcessError as e:
            logger.error("[LibreOffice] %s", e.stderr.decode(errors="replace"))
        except Exception as e:
            logger.error("[LibreOffice] %s", e)
    return None


# ══════════════════════════════════════════════════════════════════════════════
# 4. PAGE RENDERING — PDF → PNG
# ══════════════════════════════════════════════════════════════════════════════

def _render_pages(pdf_bytes: bytes) -> list[tuple[int, bytes]]:
    """
    Render each PDF page as a PNG at _PAGE_DPI.
    Returns [(page_num_1based, png_bytes), ...].
    """
    pages = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        mat = fitz.Matrix(_PAGE_DPI / 72, _PAGE_DPI / 72)
        for i, page in enumerate(doc):
            try:
                pix  = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
                png  = pix.tobytes("png")
                pages.append((i + 1, png))
                logger.info("[Render] Page %d: %d bytes", i + 1, len(png))
            except Exception as e:
                logger.error("[Render] Page %d: %s", i + 1, e)
        doc.close()
    except Exception as e:
        logger.error("[Render] %s", e)
    return pages


# ══════════════════════════════════════════════════════════════════════════════
# 5. VISION EXTRACTION — one page → question JSON
# ══════════════════════════════════════════════════════════════════════════════

_SUBJECT_PROMPTS = {
    "accounting": """
ACCOUNTING EXAM RULES — follow precisely:
• Financial statements (Income Statement, Balance Sheet, Cash Flow, Notes) are ONE question.
  Reproduce the complete table in the "question_table" field using GitHub markdown:
  | Account | Debit | Credit |
  |---------|-------|--------|
  | Sales   |       | 50000  |
  Preserve EVERY row, column header, and numerical value exactly.
• T-accounts: include both debit AND credit columns in question_table.
• Trial balance: every account name and its balance in question_table.
• Journal entries: date, account, debit, credit in question_table.
• type="accounting_statement" for statement preparation questions.
• has_visual=true for any question showing a financial table or diagram.
""",
    "mathematics": """
MATHEMATICS EXAM RULES — follow precisely:
• For every equation, formula, or mathematical expression: transcribe it in LaTeX
  inside the question_latex field. Example: "Calculate x if \\\\frac{x^2+1}{2} = 5"
  becomes question_latex: "$\\\\frac{x^2+1}{2} = 5$"
• Use display math $$...$$ for standalone equations, inline $...$ for inline.
• Geometric figures, graphs, number lines: set has_visual=true so the page image is shown.
• type="proof" for Show that / Prove that questions.
• type="calculation" for numerical answer questions.
• Preserve ALL marks notation, e.g. "(3)" at end of question text.
""",
    "sciences": """
PHYSICAL SCIENCES RULES:
• Circuit diagrams, force diagrams, velocity-time graphs, wave diagrams:
  set has_visual=true so students see the actual diagram from the page image.
  Also describe the diagram in the question text: [DIAGRAM: resistor R1=10Ω...].
• Preserve ALL SI units exactly: m·s⁻², N, J, Pa, mol·dm⁻³, kPa, etc.
• Mathematical equations: use question_latex for formulas like $v^2 = u^2 + 2as$.
• Scenario / "given information" block before sub-questions → parent_context.
• type="practical" for investigation/experiment questions.
• type="calculation" for formula-application questions.
""",
    "life_sciences": """
LIFE SCIENCES RULES:
• Biological diagrams (cells, organs, food webs, plant cross-sections):
  set has_visual=true, also write [DIAGRAM: labelled diagram of...] in question text.
• Data tables: reproduce as markdown in question_table.
• "Label parts A, B, C" → diagram description is parent_context, has_visual=true.
• type="practical" for investigation questions.
""",
    "geography": """
GEOGRAPHY RULES:
• Maps, climate graphs, cross-sections, topographic extracts:
  set has_visual=true so students see the actual image from the page.
  Also describe: [MAP: showing Gauteng region with N1 highway...].
• Stimulus text / case study is parent_context for all sub-questions below it.
• Preserve all statistics, co-ordinates, and place names exactly.
• Data tables → question_table in markdown format.
""",
    "business": """
BUSINESS STUDIES / ECONOMICS RULES:
• Case study or scenario text → parent_context (don't repeat per sub-question).
• type="essay" for discuss/critically analyse/evaluate questions (20–40 marks).
• type="short_answer" for define/identify/list questions (2–4 marks).
• Financial data tables → question_table in markdown format.
""",
    "language": """
LANGUAGE (ENGLISH / AFRIKAANS / etc.) RULES:
• Reading/comprehension passage → parent_context for ALL sub-questions below it.
• Poetry / quoted text that a question asks about → include in question field directly.
• type="mcq" for vocabulary/grammar multiple-choice.
• type="essay" for creative writing / formal essay / summary.
• type="short_answer" for contextual questions.
""",
    "cat_it": """
CAT / IT RULES:
• Code snippets must be preserved EXACTLY including indentation.
  Wrap in triple backticks with language: ```python\\n    x = 5\\n```.
• Scenario text → parent_context.
• Spreadsheet cell references like B2:B10 or $A$1 preserved exactly.
• type="practical" for spreadsheet/database/word processing tasks.
• type="calculation" for algorithm/pseudocode/trace table questions.
""",
    "history": """
HISTORY RULES:
• Source text (Document A, Cartoon B) → parent_context for questions referring to it.
• type="essay" for "to what extent" / "discuss" extended writing questions.
• Preserve all dates, names, and quoted text exactly.
• has_visual=true for questions that reference a cartoon, photograph, or map.
""",
    "general": "",
}


def _extract_page_questions(page_b64: str, page_url: str | None,
                             subject: str, grade: str,
                             page_num: int, client: Groq) -> list[dict]:
    """
    Extract questions from one rendered page image via Groq vision.
    Returns a list of question dicts with full schema including image URLs.
    """
    cat        = _cat(subject)
    subj_rules = _SUBJECT_PROMPTS.get(cat, "")
    is_acct    = cat == "accounting"
    is_maths   = cat == "mathematics"

    prompt = f"""You are a professional South African exam paper parser reading page {page_num} of a {subject} Grade {grade} exam.

Extract EVERY question on this page into a JSON array. Be thorough — include all sub-questions.

OUTPUT SCHEMA (include all fields for every question):
{{
  "question_number":  "1.1"           // exactly as printed
  "parent_question":  "QUESTION 1"    // section heading
  "parent_context":   null            // scenario/passage shared by sub-questions, or null
  "section":          "A"             // visible section letter, default "A"
  "question":         "..."           // FULL question text
  "type":             "short_answer"  // mcq|true_false|calculation|essay|short_answer|matching|practical|accounting_statement|open
  "marks":            2               // integer from (2) or [2], default 1
  "options":          null            // MCQ only: {{"A":"...","B":"...","C":"...","D":"..."}}
  "column_a":         null            // matching left column list
  "column_b":         null            // matching right column list
  "memo":             null            // always null for question papers
  "has_visual":       false           // TRUE if this question includes a diagram/graph/figure/image/map/circuit/table
  "question_latex":   null            // LaTeX string for maths equations, e.g. "$x^2 + 1 = 0$"
  "question_table":   null            // GitHub markdown table for accounting/data tables
}}

GENERAL RULES:
- For ANY diagram, graph, figure, map, circuit, or image visible on this page near a question:
  • Set has_visual=true on that question
  • Also describe it in the question text: [DIAGRAM: all labels, axes, values, component names]
  • The stored page image will be shown to students automatically when has_visual=true
- Preserve ALL marks notation "(3)" at the end of question text
- MCQ: A/B/C/D options must be COMPLETE answer text, not just letters

{subj_rules}

IMPORTANT FOR MATHEMATICS:
- Every equation MUST appear in question_latex using proper LaTeX syntax
- Fractions: \\frac{{num}}{{den}}   Powers: x^{{2}}   Roots: \\sqrt{{x}}
- Greek: \\alpha \\beta \\theta      Trig: \\sin \\cos \\tan
- Both question text AND question_latex should contain the equation

IMPORTANT FOR ACCOUNTING:
- Every financial table MUST appear in question_table as a full markdown table
- Include EVERY row including totals, subtotals, and blank/formatting rows
- Column headers must match what is shown on the page exactly

Return ONLY a valid JSON array. No markdown fences. No explanation.
Return [] for cover pages, instruction pages, or formula sheets with no questions."""

    try:
        resp = client.chat.completions.create(
            model=_VISION_MODEL,
            messages=[{"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{page_b64}"}},
                {"type": "text", "text": prompt},
            ]}],
            max_tokens=_MAX_TOKENS_PER_PAGE,
            temperature=0,
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\s*```\s*$",        "", raw, flags=re.MULTILINE)
        m   = re.search(r"\[.*\]", raw, re.DOTALL)
        if m:
            parsed = json.loads(m.group())
            if isinstance(parsed, list):
                # Attach the page image URL to every question that has a visual
                if page_url:
                    for q in parsed:
                        if q.get("has_visual"):
                            q["questionImageUrl"] = page_url
                logger.info("[Vision] Page %d → %d questions", page_num, len(parsed))
                return parsed
    except json.JSONDecodeError as e:
        logger.error("[Vision] Page %d JSON error: %s", page_num, e)
    except Exception as e:
        logger.error("[Vision] Page %d: %s", page_num, e)
    return []


# ══════════════════════════════════════════════════════════════════════════════
# 6. MERGE + DEDUPLICATE
# ══════════════════════════════════════════════════════════════════════════════

def _normalise_qnum(qn: str) -> str:
    s = str(qn).lower().strip()
    s = re.sub(r"^(question|q|ques|no|nr)[\s.\-]*", "", s)
    s = re.sub(r"[^a-z0-9]", "", s)
    return s


def _merge(page_results: list[list[dict]]) -> list[dict]:
    """Merge question lists from all pages, deduplicating by question number."""
    seen:  dict[str, int] = {}
    final: list[dict]     = []
    for page_qs in page_results:
        for q in page_qs:
            key = _normalise_qnum(q.get("question_number", "")) or q.get("question", "")[:60]
            if key and key in seen:
                # Keep later (more complete) version
                final[seen[key]] = q
            else:
                if key:
                    seen[key] = len(final)
                final.append(q)
    logger.info("[Merge] %d unique questions", len(final))
    return final


# ══════════════════════════════════════════════════════════════════════════════
# 7. PRIMARY PUBLIC API — QUESTION EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_questions_from_file(file_bytes: bytes, filename: str,
                                 subject: str, grade: str,
                                 exam_id: str = "",
                                 school_folder: str = "shared") -> list[dict]:
    """
    PRIMARY entry point for question paper extraction.

    Pipeline:
      DOCX/ODT → LibreOffice → PDF → PyMuPDF renders pages at 200 DPI
      → pages uploaded to Firebase Storage
      → Groq vision per page → questions with questionImageUrl, questionLatex, questionTable
      → merged and deduplicated question list

    Falls back to text extraction + LLM parse if LibreOffice is unavailable.
    """
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    lower  = filename.lower()

    # ── Convert to PDF ──────────────────────────────────────────────────────
    if lower.endswith(".pdf"):
        pdf_bytes = file_bytes
        logger.info("[Extract] PDF received directly: %s", filename)
    else:
        logger.info("[Extract] Converting %s → PDF (LibreOffice)", filename)
        pdf_bytes = _convert_to_pdf(file_bytes, filename)

    if pdf_bytes:
        # ── Render pages ───────────────────────────────────────────────────
        raw_pages = _render_pages(pdf_bytes)
        if not raw_pages:
            logger.error("[Extract] No pages rendered from %s", filename)
        else:
            # ── Upload page images + extract questions ─────────────────────
            page_results: list[list[dict]] = []
            for page_num, png_bytes in raw_pages:
                # Upload page PNG to Firebase Storage
                page_url = (
                    _upload_page_image(school_folder, exam_id, page_num, png_bytes)
                    if exam_id else None
                )
                # Vision extract
                page_b64 = base64.b64encode(png_bytes).decode()
                questions = _extract_page_questions(
                    page_b64, page_url, subject, grade, page_num, client
                )
                page_results.append(questions)

            questions = _merge(page_results)
            if questions:
                logger.info("[Extract] ✓ %d questions with images via render-first",
                            len(questions))
                return questions
            logger.warning("[Extract] Vision returned 0 questions — trying fallback")

    # ── Fallback: text extraction + LLM parse ──────────────────────────────
    logger.info("[Extract] Fallback: text + LLM parse (%s)", filename)
    text = extract_text_from_file(file_bytes, filename, subject)
    if text.strip():
        return parse_questions_universal(text, subject, grade)

    logger.error("[Extract] All methods failed for %s", filename)
    return []


# ══════════════════════════════════════════════════════════════════════════════
# 8. TEXT EXTRACTION (for memos only — no vision needed)
# ══════════════════════════════════════════════════════════════════════════════

def _docx_text(file_bytes: bytes) -> str:
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P
    try:
        doc, lines = Document(io.BytesIO(file_bytes)), []
        for child in doc.element.body.iterchildren():
            if isinstance(child, CT_P):
                t = Paragraph(child, doc).text.strip()
                if t: lines.append(t)
            elif isinstance(child, CT_Tbl):
                for row in Table(child, doc).rows:
                    cells = [c.text.strip() for c in row.cells]
                    r = " | ".join(c for c in cells if c)
                    if r: lines.append(r)
        return "\n".join(lines)
    except Exception as e:
        logger.error("[DOCX text] %s", e); return ""


def _odt_text(file_bytes: bytes) -> str:
    try:
        with tempfile.NamedTemporaryFile(suffix=".odt", delete=False) as tmp:
            tmp.write(file_bytes); tmp.flush()
            odt_doc = load_odt(tmp.name)
            return "\n".join(
                teletype.extractText(p)
                for p in odt_doc.getElementsByType(odf_text.P)
                if teletype.extractText(p).strip()
            )
    except Exception as e:
        logger.error("[ODT text] %s", e); return ""


def _pdf_text(file_bytes: bytes, subject: str) -> str:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    try:
        doc  = fitz.open(stream=file_bytes, filetype="pdf")
        text = "".join(page.get_text() + "\n" for page in doc)
        doc.close()
        if len(text.strip()) > 100:
            return text
        # Scanned PDF — OCR
        doc, all_text = fitz.open(stream=file_bytes, filetype="pdf"), ""
        for i, page in enumerate(doc):
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
            img = base64.b64encode(pix.tobytes("png")).decode()
            try:
                r = client.chat.completions.create(
                    model=_VISION_MODEL,
                    messages=[{"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}},
                        {"type": "text", "text": f"{subject} exam memo. Extract ALL text. Plain text only."},
                    ]}],
                    max_tokens=2500,
                )
                all_text += r.choices[0].message.content.strip() + "\n\n"
            except Exception as e:
                logger.error("[PDF OCR] p%d: %s", i + 1, e)
        doc.close()
        return all_text
    except Exception as e:
        logger.error("[PDF text] %s", e); return ""


def extract_text_from_file(file_bytes: bytes, filename: str,
                           subject: str = "General") -> str:
    """Text-only extraction for memos. No vision, no images."""
    lower = filename.lower()
    if lower.endswith(".docx"): return _docx_text(file_bytes)
    if lower.endswith(".odt"):  return _odt_text(file_bytes)
    if lower.endswith(".pdf"):  return _pdf_text(file_bytes, subject)
    if lower.endswith(".doc"):
        try: return mammoth.extract_raw_text(io.BytesIO(file_bytes)).value
        except Exception as e: logger.error("[DOC] %s", e)
    return ""


# ══════════════════════════════════════════════════════════════════════════════
# 9. TEXT FALLBACK PARSER (when LibreOffice unavailable)
# ══════════════════════════════════════════════════════════════════════════════

def parse_questions_universal(exam_text: str, subject: str, grade: str) -> list[dict]:
    """LLM parser for plain text. Fallback only."""
    client, cat = Groq(api_key=os.getenv("GROQ_API_KEY")), _cat(subject)
    all_qs, seen = [], set()
    CHUNK, OVERLAP = 10_000, 800

    chunks, start = [], 0
    while start < len(exam_text):
        chunks.append(exam_text[start:start + CHUNK])
        start += CHUNK - OVERLAP

    for idx, chunk in enumerate(chunks):
        prompt = f"""Parse this {subject} Grade {grade} exam text into a JSON array.
MCQ→type="mcq"+options. true_false. calculation. essay. short_answer. Default: open.
Marks from (2)/[2]. Include question_number, parent_question, parent_context, section.
Return ONLY valid JSON array. Each item:
{{"question_number":"1.1","parent_question":"QUESTION 1","parent_context":null,
"section":"A","question":"...","type":"open","marks":1,"options":null,
"column_a":null,"column_b":null,"memo":null,"has_visual":false,
"question_latex":null,"question_table":null}}
TEXT:
{chunk}"""
        try:
            resp = client.chat.completions.create(
                model=_PARSER_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0, max_tokens=8000,
            )
            raw = re.sub(r"^```(?:json)?\s*", "",
                         resp.choices[0].message.content.strip(), flags=re.MULTILINE)
            raw = re.sub(r"\s*```\s*$", "", raw, flags=re.MULTILINE)
            m   = re.search(r"\[.*\]", raw, re.DOTALL)
            if m:
                for q in json.loads(m.group()):
                    key = _normalise_qnum(q.get("question_number","")) or q.get("question","")[:60]
                    if key not in seen:
                        seen.add(key); all_qs.append(q)
        except Exception as e:
            logger.error("[Parser] Chunk %d: %s", idx + 1, e)

    logger.info("[Parser] %d questions (text fallback)", len(all_qs))
    return all_qs