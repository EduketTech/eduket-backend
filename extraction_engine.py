"""
extraction_engine.py — Eduket OS  v5  (multi-provider, render-first)
═══════════════════════════════════════════════════════════════════════════════
Architecture
──────────────
Every uploaded DOCX / ODT is converted to PDF by LibreOffice, rendered
page-by-page at 200 DPI, each page is uploaded to Firebase Storage, and
Groq vision reads the page image to extract a structured question JSON.

This "render-first" approach handles everything in one pass:
  • Diagrams, graphs, maps, circuit diagrams  — visible in the page render
  • OMML equations (Word math)               — rendered faithfully by LibreOffice
  • Accounting tables / financial statements — exact layout preserved
  • Any font, any formatting                 — LibreOffice renders it all

AI provider chain (automatic fallback)
──────────────────────────────────────
Text tasks (parsing, marking, memo extraction):
  1. Groq    — fastest, generous free tier
  2. Gemini  — 1,500 free calls/day, 1M context
  3. Together AI — $25 credit covers months of testing

Vision tasks (page image → question JSON):
  1. Groq vision  (llama-4-scout)
  2. Gemini vision (gemini-2.0-flash)

Public API
──────────
  extract_questions_from_file(file_bytes, filename, subject, grade,
                              exam_id, school_folder) → list[dict]
      Primary entry point for question papers.

  extract_text_from_file(file_bytes, filename, subject) → str
      Fast text-only path — used for memo extraction.

  parse_questions_universal(text, subject, grade) → list[dict]
      LLM text parser — fallback when vision pipeline unavailable.

app.py integration
──────────────────
  In run_extraction_pipeline, replace:
      exam_text = extract_text_from_file(exam_bytes, exam_fn)
      questions = parse_questions_universal(exam_text, subject, grade)
  with:
      questions = extract_questions_from_file(
          exam_bytes, exam_fn, subject, grade,
          exam_id=exam_id,
          school_folder=meta.get("schoolFolder", school_id),
      )

Render build command
─────────────────────
  apt-get install -y libreoffice --no-install-recommends && pip install -r requirements.txt

Environment variables required
────────────────────────────────
  GROQ_API_KEY      — primary AI provider
  GEMINI_API_KEY    — fallback AI provider (aistudio.google.com — free)
  TOGETHER_API_KEY  — final fallback (together.ai — $25 free credit)
"""

from __future__ import annotations   # Python 3.8+ type hint compatibility

import io
import os
import re
import json
import uuid
import time
import base64
import shutil
import logging
import subprocess
import tempfile
from pathlib import Path

import fitz          # PyMuPDF
import mammoth
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from odf.opendocument import load as load_odt
from odf import text as odf_text
from odf import teletype
from groq import Groq

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# Vision model — handles page image → question JSON.
# llama-4-scout supports image input and is not in the deprecation list.
_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# Text model priority list — tried in order, first available wins.
# Update this list after checking console.groq.com/docs/models.
_GROQ_MODEL_CANDIDATES = [
    "llama-3.1-70b-versatile",
    "llama3-70b-8192",
    "llama-3.3-70b-specdec",
    # "llama-3.1-8b-instant",     # last resort — small context, use only if others gone
]

_PAGE_DPI            = 200     # 200 DPI = sharp enough for diagrams and equations
_MAX_TOKENS_PER_PAGE = 4096    # Groq vision response budget per page
_CHUNK_SIZE          = 6_000   # chars per text chunk (≈1,500 tokens + prompt overhead)
_CHUNK_OVERLAP       = 400     # overlap to avoid splitting questions across chunks

# Module-level cache — resolved on first call, reused for all subsequent calls
_RESOLVED_GROQ_MODEL: str | None = None


# ══════════════════════════════════════════════════════════════════════════════
# MULTI-PROVIDER AI LAYER
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_groq_model() -> str:
    """
    Query Groq's /models endpoint and return the first candidate that
    is currently available. Caches the result so the API is only called once
    per process lifetime.
    """
    global _RESOLVED_GROQ_MODEL
    if _RESOLVED_GROQ_MODEL:
        return _RESOLVED_GROQ_MODEL

    try:
        client    = Groq(api_key=os.getenv("GROQ_API_KEY"))
        available = {m.id for m in client.models.list().data}
        for candidate in _GROQ_MODEL_CANDIDATES:
            if candidate in available:
                logger.info("[Model] Groq parser model resolved: %s", candidate)
                _RESOLVED_GROQ_MODEL = candidate
                return candidate
    except Exception as e:
        logger.warning("[Model] Could not query Groq models: %s", e)

    fallback = _GROQ_MODEL_CANDIDATES[-1]
    logger.warning("[Model] Falling back to last-resort Groq model: %s", fallback)
    _RESOLVED_GROQ_MODEL = fallback
    return fallback


def ai_text(prompt: str, max_tokens: int = 2000, temperature: float = 0.1) -> str:
    """
    Send a text prompt to the AI provider chain.
    Tries Groq → Gemini → Together AI in order.
    Raises RuntimeError only if all three fail.
    """
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
                err_str = str(e)
                if "decommissioned" in err_str or "deprecated" in err_str:
                    # Invalidate cached model so next call re-resolves
                    global _RESOLVED_GROQ_MODEL
                    _RESOLVED_GROQ_MODEL = None
                    logger.warning("[Groq] Model decommissioned — re-resolving")
                    continue
                if "413" in err_str and attempt == 0:
                    # Token limit — will be handled by caller with smaller chunks
                    raise
                last_error = e
                logger.warning("[Groq] Attempt %d failed: %s", attempt + 1, err_str[:120])
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
            logger.info("[AI] Gemini responded successfully")
            return response.text.strip()
        except Exception as e:
            last_error = e
            logger.warning("[Gemini] Failed: %s", str(e)[:120])

    raise RuntimeError(
        f"All AI providers failed. Last error: {last_error}. "
        "Check GROQ_API_KEY or GEMINI_API_KEY in environment."
    )


def ai_vision(image_b64: str, prompt: str) -> str:
    """
    Send a page image + prompt to the vision provider chain.
    Tries Groq vision → Gemini vision.
    """
    # ── 1. Groq vision ────────────────────────────────────────────────────
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        try:
            client = Groq(api_key=groq_key)
            resp   = client.chat.completions.create(
                model=_VISION_MODEL,
                messages=[{"role": "user", "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                    {"type": "text", "text": prompt},
                ]}],
                max_tokens=_MAX_TOKENS_PER_PAGE,
                temperature=0,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.warning("[Vision] Groq failed: %s", str(e)[:120])

    # ── 2. Gemini vision ──────────────────────────────────────────────────
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model  = genai.GenerativeModel("gemini-2.0-flash")
            img    = {"inline_data": {"mime_type": "image/png", "data": image_b64}}
            result = model.generate_content([img, prompt])
            logger.info("[Vision] Gemini vision responded")
            return result.text.strip()
        except Exception as e:
            logger.warning("[Vision] Gemini failed: %s", str(e)[:120])

    raise RuntimeError("All vision providers failed. Check API keys.")


# ══════════════════════════════════════════════════════════════════════════════
# SUBJECT CLASSIFICATION + PARSING HINTS
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
                      "sesotho", "language", "home language", "first additional",
                      "life skills", "life orientation"},
    "cat_it":        {"computer applications technology", "cat",
                      "information technology", "it"},
    "history":       {"history"},
}

def _subject_category(subject: str) -> str:
    s = subject.lower().strip()
    for cat, keywords in _SUBJECT_MAP.items():
        if any(k in s for k in keywords):
            return cat
    return "general"


_PARSING_HINTS: dict[str, str] = {
    "accounting": """
ACCOUNTING: Financial statements (Income Statement, Balance Sheet, Cash Flow, Notes)
are ONE question. Reproduce the COMPLETE table in question_table as GitHub markdown:
| Account | Debit (R) | Credit (R) |
|---------|-----------|------------|
| ...     | ...       | ...        |
Preserve EVERY row, header, subtotal, and total line exactly as shown.
T-accounts: capture both debit AND credit columns in question_table.
Trial balance: every account name and balance in question_table.
Journal entries: date, account, debit, credit in question_table.
type="accounting_statement" for statement preparation questions.
has_visual=true for any question containing a financial table or diagram.""",

    "mathematics": """
MATHEMATICS: Every equation and formula must appear in question_latex using LaTeX syntax.
Fractions: \\frac{num}{den}  Powers: x^{2}  Roots: \\sqrt{x}
Subscripts: x_{n}  Greek: \\alpha \\beta \\theta \\pi
Trig: \\sin \\cos \\tan  Limits: \\lim_{x \\to 0}
Use inline $...$ for formulas within sentences, display $$...$$ for standalone equations.
type="proof" for Show that / Prove that questions.
type="calculation" for numerical answer questions.
has_visual=true for geometric figures, graphs, number lines.""",

    "sciences": """
PHYSICAL SCIENCES: Preserve ALL SI units exactly (m·s⁻², N, J, Pa, mol·dm⁻³).
Circuit diagrams, force diagrams, velocity-time graphs: has_visual=true.
Also describe in question text: [DIAGRAM: resistor R1=10Ω connected in series...].
Equations: use question_latex e.g. $v^2 = u^2 + 2as$.
Scenario / "given information" block before sub-questions → parent_context.
type="practical" for investigation questions.
type="calculation" for formula application questions.""",

    "life_sciences": """
LIFE SCIENCES: Biological diagrams (cells, organs, food webs): has_visual=true.
Describe all labelled structures in question text: [DIAGRAM: plant cell showing...].
Data tables: reproduce as markdown in question_table.
"Label parts A, B, C" → diagram description in parent_context, has_visual=true.
type="practical" for investigation questions.""",

    "geography": """
GEOGRAPHY: Maps, climate graphs, cross-sections, topographic extracts: has_visual=true.
Describe: [MAP: Gauteng region showing N1 highway and surrounding municipalities].
Stimulus / case study text → parent_context for all sub-questions below it.
Data tables → question_table in markdown.
Preserve all statistics, co-ordinates, and place names exactly.""",

    "business": """
BUSINESS STUDIES / ECONOMICS:
Case study or scenario text → parent_context (never repeat it per sub-question).
type="essay" for discuss / critically analyse / evaluate questions (20-40 marks).
type="short_answer" for define / identify / list questions (2-4 marks).
Financial data tables → question_table in markdown.""",

    "language": """
LANGUAGE / LIFE SKILLS / LIFE ORIENTATION:
Reading passage / scenario text → parent_context for ALL sub-questions below it.
type="mcq" for vocabulary / grammar / comprehension multiple-choice.
type="essay" for creative writing / formal essay / summary tasks.
Quoted lines of poetry or prose that a question asks about → include in question field.
For Life Orientation case studies: scenario text is always parent_context.""",

    "cat_it": """
CAT / IT: Code snippets MUST be preserved EXACTLY including indentation.
Wrap in triple backticks: ```python\\n    x = 5\\n```.
Scenario text → parent_context.
Spreadsheet cell references like B2:B10 or $A$1 preserved exactly.
type="practical" for spreadsheet / database / word-processing tasks.
type="calculation" for algorithm / pseudocode / trace table questions.""",

    "history": """
HISTORY: Source text (Document A, Cartoon B, photograph) → parent_context.
type="essay" for "to what extent" / "discuss" questions.
has_visual=true for cartoons, photographs, or maps.
Preserve all dates, names, and quoted text exactly.""",

    "general": "",
}


# ══════════════════════════════════════════════════════════════════════════════
# FIREBASE STORAGE — page image upload
# ══════════════════════════════════════════════════════════════════════════════

def _upload_page_image(school_folder: str, exam_id: str,
                        page_num: int, png_bytes: bytes) -> str | None:
    """
    Upload a rendered page PNG to Firebase Storage.
    Returns a tokenized download URL or None on failure.
    Path: exam_pages/{schoolFolder}/{examId}/page_{nnn}.png
    The Admin SDK bypasses Firestore rules — no permission issues.
    """
    try:
        from firebase_admin import storage as fb_storage
        bucket       = fb_storage.bucket()
        token        = str(uuid.uuid4())
        path         = f"exam_pages/{school_folder}/{exam_id}/page_{page_num:03d}.png"
        blob         = bucket.blob(path)
        blob.metadata = {"firebaseStorageDownloadTokens": token}
        blob.upload_from_string(png_bytes, content_type="image/png")
        blob.patch()
        encoded = path.replace("/", "%2F")
        url = (
            f"https://firebasestorage.googleapis.com/v0/b/{bucket.name}"
            f"/o/{encoded}?alt=media&token={token}"
        )
        logger.info("[Storage] Page %d uploaded → %s", page_num, path)
        return url
    except Exception as e:
        logger.error("[Storage] Upload failed p%d: %s", page_num, e)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# LIBREOFFICE CONVERSION — DOCX / ODT → PDF
# ══════════════════════════════════════════════════════════════════════════════

def _lo_available() -> bool:
    return bool(shutil.which("libreoffice") or shutil.which("soffice"))


def _convert_to_pdf(file_bytes: bytes, filename: str) -> bytes | None:
    """
    Convert DOCX / ODT / DOC / RTF → PDF using LibreOffice headless.
    LibreOffice renders fonts, OMML equations, embedded images, and table
    formatting exactly as Microsoft Word/Calc would display them.
    Returns PDF bytes or None if LibreOffice is unavailable or conversion fails.
    """
    if not _lo_available():
        logger.warning("[LibreOffice] Not installed — falling back to text extraction")
        return None

    cmd = shutil.which("libreoffice") or shutil.which("soffice")
    with tempfile.TemporaryDirectory() as tmp:
        inp = os.path.join(tmp, filename)
        with open(inp, "wb") as f:
            f.write(file_bytes)
        try:
            subprocess.run(
                [cmd, "--headless", "--convert-to", "pdf", "--outdir", tmp, inp],
                check=True, timeout=120, capture_output=True,
            )
            pdf_path = os.path.join(tmp, Path(filename).stem + ".pdf")
            if os.path.exists(pdf_path):
                data = open(pdf_path, "rb").read()
                logger.info("[LibreOffice] %s → PDF (%d bytes)", filename, len(data))
                return data
            logger.error("[LibreOffice] PDF not produced for %s", filename)
        except subprocess.TimeoutExpired:
            logger.error("[LibreOffice] Timeout converting %s", filename)
        except subprocess.CalledProcessError as e:
            logger.error("[LibreOffice] %s", e.stderr.decode(errors="replace")[:300])
        except Exception as e:
            logger.error("[LibreOffice] %s", e)
    return None


# ══════════════════════════════════════════════════════════════════════════════
# PAGE RENDERING — PDF → PNG
# ══════════════════════════════════════════════════════════════════════════════

def _render_pages(pdf_bytes: bytes) -> list[tuple[int, bytes]]:
    """
    Render each PDF page as a PNG at _PAGE_DPI (200).
    Returns [(page_number_1based, png_bytes), ...].
    """
    pages: list[tuple[int, bytes]] = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        mat = fitz.Matrix(_PAGE_DPI / 72, _PAGE_DPI / 72)
        for i, page in enumerate(doc):
            try:
                pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
                png = pix.tobytes("png")
                pages.append((i + 1, png))
                logger.info("[Render] Page %d: %d bytes", i + 1, len(png))
            except Exception as e:
                logger.error("[Render] Page %d: %s", i + 1, e)
        doc.close()
    except Exception as e:
        logger.error("[Render] PDF open failed: %s", e)
    return pages


# ══════════════════════════════════════════════════════════════════════════════
# VISION EXTRACTION — rendered page → question JSON
# ══════════════════════════════════════════════════════════════════════════════

def _extract_page_questions(page_b64: str, page_url: str | None,
                             subject: str, grade: str,
                             page_num: int) -> list[dict]:
    """
    Send one rendered page image to the vision AI and get back a structured
    list of question dicts. The vision model sees exactly what the teacher's
    Word document looks like — diagrams, equations, and tables are all visible.
    """
    cat   = _subject_category(subject)
    hints = _PARSING_HINTS.get(cat, "")

    prompt = f"""You are a professional South African exam parser reading page {page_num} of a {subject} Grade {grade} exam paper.

Extract EVERY question on this page into a JSON array. Be thorough — include all sub-questions.

OUTPUT SCHEMA (all fields required for every question):
{{
  "question_number":  "1.1",
  "parent_question":  "QUESTION 1",
  "parent_context":   null,
  "section":          "A",
  "question":         "Full question text including any [DIAGRAM: ...] descriptions",
  "type":             "short_answer",
  "marks":            2,
  "options":          null,
  "column_a":         null,
  "column_b":         null,
  "memo":             null,
  "has_visual":       false,
  "question_latex":   null,
  "question_table":   null
}}

FIELD RULES:
- question_number: exactly as printed — "1.1", "2.3.1", "QUESTION 3"
- parent_question: the section heading above this question — "QUESTION 1"
- parent_context: scenario / passage / source text shared by sub-questions (null if none)
- section: visible letter (A, B, C) or "A" as default
- question: FULL text. For ANY diagram, graph, circuit, map, or figure visible
  near this question: describe it inline → [DIAGRAM: all labels, axes, units, values]
- type: mcq | true_false | calculation | proof | essay | short_answer |
        matching | practical | accounting_statement | open
- marks: integer from (2) or [2], default 1
- options: MCQ only — {{"A":"...","B":"...","C":"...","D":"..."}} — full answer text
- has_visual: true if this question has a diagram, graph, figure, table, or image
- question_latex: LaTeX string for any equation e.g. "$x^2 + 1 = 0$" (null if none)
- question_table: full GitHub markdown table for accounting / data tables (null if none)
- memo: always null for question papers

{hints}

IMPORTANT:
- Include EVERY question on this page including sub-questions.
- Marks notation "(3)" at end of line must be captured in the marks field.
- MCQ options must contain the full answer text, not just the letter.
- Return [] for cover pages, instruction pages, or formula sheets.

Return ONLY a valid JSON array. No markdown fences. No explanation."""

    try:
        raw = ai_vision(page_b64, prompt)
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\s*```\s*$",        "", raw, flags=re.MULTILINE)
        m   = re.search(r"\[.*\]", raw, re.DOTALL)
        if m:
            parsed = json.loads(m.group())
            if isinstance(parsed, list):
                # Attach page image URL to questions with visual elements
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
# MERGE + DEDUPLICATE
# ══════════════════════════════════════════════════════════════════════════════

def _normalise_qnum(qn: str) -> str:
    """Normalise question number for deduplication comparison."""
    s = str(qn).lower().strip()
    s = re.sub(r"^(question|q|ques|no|nr)[\s.\-]*", "", s)
    s = re.sub(r"[^a-z0-9]", "", s)
    return s


def _merge_pages(page_results: list[list[dict]]) -> list[dict]:
    """
    Merge question arrays from all pages, deduplicating by question number.
    When the same question number appears on two pages (carry-over), the later
    version wins — it is likely more complete.
    """
    seen:  dict[str, int] = {}
    final: list[dict]     = []

    for page_qs in page_results:
        for q in page_qs:
            key = (
                _normalise_qnum(q.get("question_number", ""))
                or q.get("question", "")[:60].strip()
            )
            if key and key in seen:
                final[seen[key]] = q        # update with fuller later version
            else:
                if key:
                    seen[key] = len(final)
                final.append(q)

    logger.info("[Merge] %d unique questions from %d pages",
                len(final), len(page_results))
    return final


# ══════════════════════════════════════════════════════════════════════════════
# PRIMARY PUBLIC API — QUESTION EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_questions_from_file(
    file_bytes:    bytes,
    filename:      str,
    subject:       str,
    grade:         str,
    exam_id:       str = "",
    school_folder: str = "shared",
) -> list[dict]:
    """
    PRIMARY entry point for question paper extraction.

    Pipeline:
      DOCX/ODT → LibreOffice → PDF → PyMuPDF renders pages at 200 DPI
      → pages uploaded to Firebase Storage (if exam_id provided)
      → vision AI extracts questions per page with image URLs, LaTeX, tables
      → merged and deduplicated question list returned

    Falls back to text extraction + LLM parse if LibreOffice unavailable.
    """
    lower = filename.lower()

    # ── Step 1: Get PDF ───────────────────────────────────────────────────
    if lower.endswith(".pdf"):
        pdf_bytes = file_bytes
        logger.info("[Extract] Direct PDF: %s", filename)
    else:
        logger.info("[Extract] Converting %s → PDF via LibreOffice", filename)
        pdf_bytes = _convert_to_pdf(file_bytes, filename)

    if pdf_bytes:
        # ── Step 2: Render pages ──────────────────────────────────────────
        raw_pages = _render_pages(pdf_bytes)

        if not raw_pages:
            logger.error("[Extract] No pages rendered from %s", filename)
        else:
            # ── Step 3: Upload page images + vision extract ───────────────
            page_results: list[list[dict]] = []

            for page_num, png_bytes in raw_pages:
                # Upload PNG to Firebase Storage
                page_url = (
                    _upload_page_image(school_folder, exam_id, page_num, png_bytes)
                    if exam_id else None
                )
                # Vision extraction
                page_b64  = base64.b64encode(png_bytes).decode()
                questions = _extract_page_questions(
                    page_b64, page_url, subject, grade, page_num
                )
                page_results.append(questions)

            questions = _merge_pages(page_results)

            if questions:
                logger.info("[Extract] ✓ %d questions via render-first pipeline",
                            len(questions))
                return questions

            logger.warning("[Extract] Vision returned 0 questions — trying text fallback")

    # ── Fallback: text extraction + LLM parse ────────────────────────────
    # Used when LibreOffice is not installed or vision returns nothing.
    logger.info("[Extract] Text fallback: %s", filename)
    text = extract_text_from_file(file_bytes, filename, subject)

    if text.strip():
        return parse_questions_universal(text, subject, grade)

    logger.error("[Extract] All methods exhausted for %s", filename)
    return []


# ══════════════════════════════════════════════════════════════════════════════
# TEXT EXTRACTION — fast path for memos
# ══════════════════════════════════════════════════════════════════════════════
# Memos only need question_number → answer text mapping.
# No vision, no images, no structure preservation needed.

def _docx_text(file_bytes: bytes) -> str:
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P
    try:
        doc, lines = Document(io.BytesIO(file_bytes)), []
        for child in doc.element.body.iterchildren():
            if isinstance(child, CT_P):
                t = Paragraph(child, doc).text.strip()
                if t:
                    lines.append(t)
            elif isinstance(child, CT_Tbl):
                for row in Table(child, doc).rows:
                    # Deduplicate merged cells by underlying _tc identity
                    seen_tc: set[int] = set()
                    cells = []
                    for c in row.cells:
                        if id(c._tc) not in seen_tc:
                            seen_tc.add(id(c._tc))
                            cells.append(c.text.strip())
                    row_txt = " | ".join(c for c in cells if c)
                    if row_txt:
                        lines.append(row_txt)
        return "\n".join(lines)
    except Exception as e:
        logger.error("[DOCX text] %s", e)
        return ""


def _odt_text(file_bytes: bytes) -> str:
    try:
        with tempfile.NamedTemporaryFile(suffix=".odt", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp.flush()
            odt_doc = load_odt(tmp.name)
            return "\n".join(
                teletype.extractText(p)
                for p in odt_doc.getElementsByType(odf_text.P)
                if teletype.extractText(p).strip()
            )
    except Exception as e:
        logger.error("[ODT text] %s", e)
        return ""


def _pdf_text(file_bytes: bytes, subject: str) -> str:
    """Extract text from PDF — tries native text first, falls back to vision OCR."""
    try:
        doc  = fitz.open(stream=file_bytes, filetype="pdf")
        text = "".join(page.get_text() + "\n" for page in doc)
        doc.close()
        if len(text.strip()) > 100:
            return text

        # Scanned PDF — run vision OCR page by page
        logger.info("[PDF text] No native text — running vision OCR")
        doc, all_text = fitz.open(stream=file_bytes, filetype="pdf"), ""
        for i, page in enumerate(doc):
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
            img = base64.b64encode(pix.tobytes("png")).decode()
            try:
                result   = ai_vision(img,
                    f"{subject} exam memo page. Extract ALL text. Plain text only.")
                all_text += result + "\n\n"
            except Exception as e:
                logger.error("[PDF OCR] Page %d: %s", i + 1, e)
        doc.close()
        return all_text
    except Exception as e:
        logger.error("[PDF text] %s", e)
        return ""


def extract_text_from_file(file_bytes: bytes,
                           filename:   str,
                           subject:    str = "General") -> str:
    """
    Text-only extraction — used for memo parsing.
    Fast path: no vision, no images, no structure preservation.
    """
    lower = filename.lower()
    if lower.endswith(".docx"):              return _docx_text(file_bytes)
    if lower.endswith(".odt"):               return _odt_text(file_bytes)
    if lower.endswith(".pdf"):               return _pdf_text(file_bytes, subject)
    if lower.endswith((".doc", ".docm", ".rtf")):
        try:
            return mammoth.extract_raw_text(io.BytesIO(file_bytes)).value
        except Exception as e:
            logger.error("[DOC/RTF] mammoth: %s", e)
    logger.warning("[extract_text] Unrecognised extension: %s", filename)
    return ""


# ══════════════════════════════════════════════════════════════════════════════
# TEXT FALLBACK PARSER — used when LibreOffice unavailable
# ══════════════════════════════════════════════════════════════════════════════

def parse_questions_universal(exam_text: str,
                               subject:   str,
                               grade:     str) -> list[dict]:
    """
    LLM-based question parser for plain text.
    Called when the render-first pipeline is unavailable (no LibreOffice) or
    when vision returns zero questions.
    Chunks the text to stay within model token limits and deduplicates results.
    """
    cat     = _subject_category(subject)
    hints   = _PARSING_HINTS.get(cat, "")
    all_qs: list[dict] = []
    seen:   set[str]   = set()

    # Build chunks with overlap so questions are never split across boundaries
    chunks: list[str] = []
    start = 0
    while start < len(exam_text):
        chunks.append(exam_text[start:start + _CHUNK_SIZE])
        start += _CHUNK_SIZE - _CHUNK_OVERLAP

    for idx, chunk in enumerate(chunks):
        logger.info("[Parser] Chunk %d/%d — %s", idx + 1, len(chunks), subject)

        prompt = f"""Parse this {subject} Grade {grade} exam into a JSON array.

MCQ → type="mcq" + options dict. True/False → "true_false". Calculation → "calculation".
Essay → "essay". Short answer → "short_answer". Default → "open".
Marks from (2) or [2]. Include question_number, parent_question, parent_context, section.
{hints}

Return ONLY a valid JSON array. Each item:
{{"question_number":"1.1","parent_question":"QUESTION 1","parent_context":null,
"section":"A","question":"...","type":"open","marks":1,"options":null,
"column_a":null,"column_b":null,"memo":null,"has_visual":false,
"question_latex":null,"question_table":null}}

EXAM TEXT:
{chunk}"""

        # Retry once with half-chunk on 413 token limit error
        for attempt, current_prompt in enumerate([prompt, prompt[:len(prompt)//2]]):
            try:
                raw = ai_text(current_prompt, max_tokens=8000, temperature=0)
                raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
                raw = re.sub(r"\s*```\s*$",        "", raw, flags=re.MULTILINE)
                m   = re.search(r"\[.*\]", raw, re.DOTALL)
                if m:
                    parsed = json.loads(m.group())
                    if isinstance(parsed, list):
                        for q in parsed:
                            key = (
                                _normalise_qnum(q.get("question_number", ""))
                                or q.get("question", "")[:60]
                            )
                            if key not in seen:
                                seen.add(key)
                                all_qs.append(q)
                break  # success
            except Exception as e:
                if "413" in str(e) and attempt == 0:
                    logger.warning("[Parser] Chunk %d too large, retrying at half", idx + 1)
                    continue
                logger.error("[Parser] Chunk %d: %s", idx + 1, e)
                break

    logger.info("[Parser] %d questions extracted (text fallback)", len(all_qs))
    return all_qs