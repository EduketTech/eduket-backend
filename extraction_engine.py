"""
extraction_engine.py — Eduket OS  v5.1  (multi-provider, render-first)
═══════════════════════════════════════════════════════════════════════════════
Architecture
──────────────
Every uploaded DOCX / ODT is converted to PDF by LibreOffice, rendered
page-by-page at 200 DPI, each page is uploaded to Firebase Storage, and
vision AI reads the page image to extract a structured question JSON.

This "render-first" approach handles everything in one pass:
  • Diagrams, graphs, maps, circuit diagrams  — visible in the page render
  • OMML equations (Word math)               — rendered faithfully by LibreOffice
  • Accounting tables / financial statements — exact layout preserved
  • Any font, any formatting                 — LibreOffice renders it all

Two-phase extraction (v5.1 improvement):
  Phase 1 — classify each page (cover / instructions / toc / questions / ...)
  Phase 2 — extract questions only from genuine question pages
  This prevents instruction numbers (1. Do not... 2. Write neatly...) from
  being falsely extracted as exam questions.

AI provider chain (automatic fallback)
──────────────────────────────────────
Text tasks: Groq → Gemini
Vision tasks: Groq vision (llama-4-scout) → Gemini vision (gemini-2.0-flash)
"""

from __future__ import annotations

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
import zipfile
from pathlib import Path
from typing import Optional

import fitz          # PyMuPDF
import mammoth
import magic
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

_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

_GROQ_MODEL_CANDIDATES = [
    "llama-3.3-70b-versatile",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
]

_PAGE_DPI            = 200
_MAX_TOKENS_PER_PAGE = 4096
_CHUNK_SIZE          = 6_000
_CHUNK_OVERLAP       = 400

_RESOLVED_GROQ_MODEL: str | None = None


# ══════════════════════════════════════════════════════════════════════════════
# SUBJECT CLASSIFICATION  (single definition — no duplicate below)
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
    """Map a subject name to a parsing hints category. Single definition."""
    s = subject.lower().strip()
    for cat, keywords in _SUBJECT_MAP.items():
        if any(k in s for k in keywords):
            return cat
    return "general"


# ══════════════════════════════════════════════════════════════════════════════
# PARSING HINTS  (single definition — comprehensive, not overwritten below)
# ══════════════════════════════════════════════════════════════════════════════

_PARSING_HINTS: dict[str, str] = {
    "accounting": """
ACCOUNTING: Financial statements (Income Statement, Balance Sheet, Cash Flow)
are ONE question. Reproduce the COMPLETE table in question_table as markdown:
| Account | Debit (R) | Credit (R) |
|---------|-----------|------------|
Preserve EVERY row, header, subtotal, and total line.
T-accounts: capture both debit AND credit columns.
type="accounting_statement" for statement preparation questions.
has_visual=true for any question containing a financial table or diagram.""",

    "mathematics": """
MATHEMATICS — NSC/SC EXAM RULES:

LATEX — every equation, expression, and formula goes in question_latex:
  Quadratic:      $(x+5)(x-2)=0$
  Exponential:    $2 \\cdot 2^{2x} - 9 \\cdot 2^x + 4 = 0$
  Surd/nested:    $\\sqrt{\\sqrt{\\frac{1}{x}} + 2} = \\frac{1}{\\sqrt{x}}$
  Logarithm:      $f(x) = \\log_{\\frac{1}{3}} x$
  Summation:      $\\sum_{p=k}^{117}(4p-1) = 26\\,675$
  Sequence:       $T_n = -n^2 + 38n - 1$
  First princip:  $f'(x) = \\lim_{h \\to 0} \\frac{f(x+h)-f(x)}{h}$
  2nd deriv:      $f''(x)$
  Rational dy/dx: $\\frac{dy}{dx}$ if $y = \\frac{2x^4+1}{x^2}$
  Inverse fn:     $f^{-1}$, $T_{25}$, $S_{\\infty}$

SECTION TOTAL vs QUESTION MARKS — critical:
  (2) immediately right of question text → marks=2 for THAT question
  [25] at END of question block          → section TOTAL, NOT a question's marks
  Example: "1.2 ... (6) [25]" → marks=6 (the [25] is QUESTION 1 total)

PARENT CONTEXT — capture EVERYTHING shared by sub-questions:
  Scenario text + ANY data table in the scenario must ALL go into parent_context.
  Example Q3: parent_context must include the torpedo scenario text AND the table:
    "The depth of a torpedo forms a quadratic pattern...
     | Time | Depth (m) |
     |------|-----------|
     | At the end of the first second | 36 |
     | At the end of the first 2 seconds | 71 |
     | At the end of the first 3 seconds | 104 |"

DATA TABLE inside a question body → question_table (markdown):
  | | JUICE | ENERGY DRINKS | TOTAL |
  |---|---|---|---|
  | Female | a | b | c |
  Sub-questions under that question inherit it via parent_context.

QUESTION TYPES:
  "Show that..."                   → type="proof"
  "Prove that..."                  → type="proof"
  "Determine f'(x) from first principles" → type="proof"
  "Calculate...", "Determine..."   → type="calculation"
  "Write down..."                  → type="short_answer"
  "Draw the graph...", "Sketch..." → type="open", has_visual=true
  "Describe the transformation"    → type="short_answer"
  Inequality solve (8x²>2x)        → type="calculation"

GRAPH PAGES — has_visual=true for ALL sub-questions when:
  - A graph/diagram appears on the same page
  - Describe the graph in parent_context:
      "[DIAGRAM: Graph of f(x)=log_{1/3}x. Decreasing curve.
       Point A on positive x-axis. Point (3;t) below x-axis.]"

BULLET POINT conditions before a single mark allocation = ONE question:
  "1.2 Calculate x and y if:
    • x is the sum of 2 and y
    • Five times the product..."  (6)
  → question_number="1.2", marks=6 — NOT two separate questions

SIGMA/SUMMATION:
  Always in LaTeX: $\\sum_{p=k}^{117}(4p-1) = 26\\,675$
  Do NOT write as plain text "sum from p=k to 117"

SECOND DERIVATIVE: f''(x) → $f''(x)$  (two primes, not f double prime)
""",

    "sciences": """
PHYSICAL SCIENCES: Preserve ALL SI units exactly (m·s⁻², N, J, Pa, mol·dm⁻³).
Circuit diagrams, force diagrams, velocity-time graphs: has_visual=true.
Describe inline: [DIAGRAM: resistor R1=10Ω connected in series...].
Equations in question_latex. type="practical" for investigation questions.""",

    "life_sciences": """
LIFE SCIENCES: Biological diagrams (cells, organs, food webs): has_visual=true.
Describe all labelled structures: [DIAGRAM: plant cell showing chloroplast...].
Data tables: reproduce as markdown in question_table.
type="practical" for investigation questions.""",

    "geography": """
GEOGRAPHY: Maps, climate graphs, cross-sections: has_visual=true.
Describe: [MAP: Gauteng region showing N1 highway...].
Stimulus/case study text → parent_context for sub-questions.
Data tables → question_table in markdown.""",

    "business": """
BUSINESS STUDIES / ECONOMICS:
Case study or scenario text → parent_context (never repeat per sub-question).
type="essay" for discuss / critically analyse / evaluate (20-40 marks).
type="short_answer" for define / identify / list (2-4 marks).
Financial data tables → question_table in markdown.""",

    "language": """
ENGLISH / LANGUAGE / LIFE ORIENTATION:
Reading passage / extract / poem → parent_context for ALL sub-questions.
type="mcq" for vocabulary / grammar / comprehension multiple-choice.
type="essay" for creative writing / formal essay / summary tasks.
COLUMN A / COLUMN B matching → type="matching", use column_a and column_b.
"(a) What tone... (b) Why would..." → two separate sub-questions.
"Discuss your view" → type="essay". Figure of speech ID → type="short_answer".
Marks shown as "(4 × 1) (4)" = 4 marks total for 4 matching items.""",

    "cat_it": """
CAT / IT: Code snippets preserved EXACTLY with indentation in triple backticks.
Spreadsheet references like B2:B10 or $A$1 preserved exactly.
type="practical" for spreadsheet / database / word-processing tasks.
type="calculation" for algorithm / pseudocode / trace table questions.""",

    "history": """
HISTORY: Source text (Document A, Cartoon B, photograph) → parent_context.
type="essay" for "to what extent" / "discuss" questions.
has_visual=true for cartoons, photographs, or maps.""",

    "general": "",
}


# ══════════════════════════════════════════════════════════════════════════════
# MULTI-PROVIDER AI LAYER
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_groq_model() -> str:
    global _RESOLVED_GROQ_MODEL
    if _RESOLVED_GROQ_MODEL:
        return _RESOLVED_GROQ_MODEL
    try:
        client    = Groq(api_key=os.getenv("GROQ_API_KEY"))
        available = {m.id for m in client.models.list().data}
        for candidate in _GROQ_MODEL_CANDIDATES:
            if candidate in available:
                logger.info("[Model] Groq model resolved: %s", candidate)
                _RESOLVED_GROQ_MODEL = candidate
                return candidate
    except Exception as e:
        logger.warning("[Model] Could not query Groq models: %s", e)
    fallback = _GROQ_MODEL_CANDIDATES[0]
    logger.warning("[Model] Falling back to: %s", fallback)
    _RESOLVED_GROQ_MODEL = fallback
    return fallback


def ai_text(prompt: str, max_tokens: int = 2000, temperature: float = 0.1) -> str:
    """Send a text prompt to Groq → Gemini fallback chain."""
    last_error: Exception | None = None

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
                    global _RESOLVED_GROQ_MODEL
                    _RESOLVED_GROQ_MODEL = None
                    logger.warning("[Groq] Model decommissioned — re-resolving")
                    continue
                if "413" in err_str and attempt == 0:
                    raise
                last_error = e
                logger.warning("[Groq] Attempt %d failed: %s", attempt + 1, err_str[:120])
                time.sleep(1)
                break

    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model    = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            last_error = e
            logger.warning("[Gemini] Failed: %s", str(e)[:120])

    raise RuntimeError(f"All AI providers failed. Last error: {last_error}")


def ai_vision(image_b64: str, prompt: str) -> str:
    """Send a page image + prompt to Groq vision → Gemini vision fallback."""
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        for attempt in range(3):           # ← retry up to 3 times
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
                err = str(e)
                if "429" in err or "rate_limit" in err.lower():
                    wait = 2 ** attempt      # 1s, 2s, 4s
                    logger.warning("[Vision] Groq rate limit — waiting %ds (attempt %d/3)",
                                   wait, attempt + 1)
                    time.sleep(wait)
                    continue
                logger.warning("[Vision] Groq failed: %s", err[:120])
                break                        # non-429 error → fall through to Gemini

    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model  = genai.GenerativeModel("gemini-2.0-flash")
            img    = {"inline_data": {"mime_type": "image/png", "data": image_b64}}
            result = model.generate_content([img, prompt])
            return result.text.strip()
        except Exception as e:
            logger.warning("[Vision] Gemini failed: %s", str(e)[:120])

    raise RuntimeError("All vision providers failed. Check API keys.")


# ══════════════════════════════════════════════════════════════════════════════
# FIREBASE STORAGE — page image upload
# ══════════════════════════════════════════════════════════════════════════════

def _upload_page_image(school_folder: str, exam_id: str,
                        page_num: int, png_bytes: bytes) -> Optional[str]:
    try:
        from firebase_admin import storage as fb_storage
        bucket  = fb_storage.bucket()
        token   = str(uuid.uuid4())
        path    = f"exam_pages/{school_folder}/{exam_id}/page_{page_num:03d}.png"
        blob    = bucket.blob(path)
        blob.metadata = {"firebaseStorageDownloadTokens": token}
        blob.upload_from_string(png_bytes, content_type="image/png")
        blob.patch()
        encoded = path.replace("/", "%2F")
        url = (f"https://firebasestorage.googleapis.com/v0/b/{bucket.name}"
               f"/o/{encoded}?alt=media&token={token}")
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


MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB


def _validate_document(file_bytes: bytes, filename: str) -> Optional[str]:
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        return f"File exceeds the 50 MB limit ({len(file_bytes) // 1024 // 1024} MB)"
    try:
        detected = magic.from_buffer(file_bytes[:2048], mime=True)
        allowed  = {
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword",
            "application/vnd.oasis.opendocument.text",
            "application/pdf",
        }
        if detected not in allowed:
            return f"Invalid file type detected: {detected}"
    except Exception:
        pass
    if filename.lower().endswith(".docx"):
        try:
            with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
                if sum(f.file_size for f in z.infolist()) > 500 * 1024 * 1024:
                    return "File rejected: ZIP bomb detected"
                if len(z.infolist()) > 10000:
                    return "File rejected: too many ZIP entries"
        except zipfile.BadZipFile:
            return "Invalid DOCX file format"
    return None


def _convert_to_pdf(file_bytes: bytes, filename: str) -> Optional[bytes]:
    error = _validate_document(file_bytes, filename)
    if error:
        logger.error("[Security] File rejected: %s", error)
        return None
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
                [cmd, "--headless", "--norestore", "--nofirststartwizard",
                 "--infilter=writer_pdf_Export",
                 "-env:UserInstallation=file:///tmp/libreoffice-sandbox",
                 "--convert-to", "pdf", "--outdir", tmp, inp],
                check=True, timeout=90, capture_output=True,
                env={**os.environ,
                     "http_proxy":  "http://127.0.0.1:0",
                     "https_proxy": "http://127.0.0.1:0",
                     "no_proxy": ""},
            )
            pdf_path = os.path.join(tmp, Path(filename).stem + ".pdf")
            if os.path.exists(pdf_path):
                with open(pdf_path, "rb") as f:
                    data = f.read()
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
# TWO-PHASE VISION EXTRACTION
# Phase 1: classify page type (skip covers, instructions, TOC, checklists)
# Phase 2: extract questions from genuine question pages only
# ══════════════════════════════════════════════════════════════════════════════

_PAGE_CLASSIFIER_PROMPT = """You are reading a page from a South African school exam paper (NSC/SC or lower grade).

Classify this page into EXACTLY ONE of these categories:

- "cover"        : Title page — subject name, grade, date, marks, time only
- "instructions" : "INSTRUCTIONS AND INFORMATION" or "Read this page carefully" — administrative rules (Do not... Write neatly...)
- "toc"          : Table of contents listing question numbers and page numbers
- "checklist"    : Tick-box checklist of sections answered
- "formula"      : Formula sheet, periodic table, data sheet, conversion table
- "extract"      : A passage, poem, short story, or extract WITHOUT questions (just source text for students to read)
- "questions"    : Page containing actual exam questions (numbered 1.1, 1.2.3, QUESTION 1 etc.)
- "mixed"        : Contains BOTH a passage/extract AND questions about it on the same page

KEY RULES:
- "INSTRUCTIONS AND INFORMATION" or "Read this page carefully before you begin" → ALWAYS "instructions"
- Numbered administrative rules (do not copy, write neatly, answer TWO sections) → "instructions"
- Numbered items asking students to DO something (explain, describe, state) → "questions"
- "TABLE OF CONTENTS" at top → "toc"
- "CHECKLIST" at top → "checklist"
- Only title + marks + time → "cover"

Return ONLY valid JSON:
{"page_type": "questions", "reason": "one sentence"}"""


def _build_extraction_prompt(subject: str, grade: str,
                              page_num: int, subject_hints: str) -> str:
    return f"""You are a professional South African NSC exam parser reading page {page_num} of a {subject} Grade {grade} exam paper.

WHAT IS NOT A QUESTION — skip these entirely:
  ✗ Numbered administrative instructions (1. Do not... 2. Answer TWO... 3. Write neatly...)
  ✗ TABLE OF CONTENTS rows
  ✗ CHECKLIST rows
  ✗ Section headings: "SECTION A: NOVEL", "SECTION B: DRAMA"
  ✗ Notes: "NOTE: Answer questions from ANY TWO sections"
  ✗ Directions: "Answer ALL the questions on the novel you have studied"
  ✗ Source references: "[Book 1, Chapter 8]", "[Act 3, Scene 1]"
  ✗ Footer text: "Copyright reserved", "Please turn over"
  ✗ Passage / poem / extract text that students READ (not answer)
  ✗ "The number of marks allocated serves as a guide to expected length"
  ✗ "Read the extract below and answer the questions set on each"

WHAT IS A QUESTION — extract these:
  ✓ Numbered items asking students to DO something: 1.1, 1.1.1, 1.1.2, 2.3.1
  ✓ "Explain", "Describe", "State", "Choose", "Refer to", "Discuss", "Identify"
  ✓ MCQ with options A B C D
  ✓ COLUMN A / COLUMN B matching tables
  ✓ "Discuss your view" essay questions
  ✓ Sub-questions (a) (b) (c) under a numbered question

SA NUMBERING RULES:
  "QUESTION 1" → section header only. parent_question = "QUESTION 1". Do NOT create a question entry.
  "1.1" → sub-question. parent_question = "QUESTION 1", question_number = "1.1"
  "1.1.1" → sub-sub-question. question_number = "1.1.1", parent_question = "QUESTION 1"
  "(a)" under "1.1.5" → question_number = "1.1.5(a)", parent_question = "QUESTION 1"
  "5.1", "5.2" → sub-questions of QUESTION 5
  "6.1.1" → sub-question of section 6.1 of QUESTION 6

PARENT CONTEXT:
  If a passage/extract appears on the same page ABOVE the questions, copy it
  into parent_context for ALL questions on the page that refer to it.
  If the extract was on a previous page, set parent_context = null.

MATCHING QUESTIONS (COLUMN A / COLUMN B):
  type = "matching"
  column_a = {{"(a)": "Mrs Kumalo", "(b)": "Johannes Pafuri", ...}}
  column_b = {{"A": "is forgiving...", "B": "is prepared to...", ...}}
  marks = number of items (e.g. 4 items × 1 = 4 marks)

MARKS: (2)=2, [2]=2, (4×1)=4, (4 x 1)=4. Default = 1 if not shown.

OUTPUT SCHEMA:
{{
  "question_number":  "1.1.1",
  "parent_question":  "QUESTION 1",
  "parent_context":   null,
  "section":          "A",
  "question":         "Full question text",
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

TYPES: mcq | true_false | matching | essay | short_answer | calculation | proof | open | accounting_statement

{subject_hints}

CRITICAL:
  1. Return [] for any non-question page.
  2. NEVER create an entry for a heading or instruction paragraph.
  3. Preserve the EXACT question number as printed ("1.1.1" not "Q1.1.1").
  4. For MCQ include FULL option text, not just the letter.
  5. memo is ALWAYS null.
  6. For diagrams: [DIAGRAM: x-axis=time(s), y-axis=velocity(m/s), peak at t=3s]

Return ONLY a valid JSON array. No markdown. No explanation."""


_SKIP_PAGE_TYPES = {"cover", "instructions", "toc", "checklist", "formula", "extract"}

_INSTRUCTION_PHRASES = [
    "do not attempt", "read this page carefully",
    "this question paper consists of", "answer two questions",
    "number the answers correctly", "start each section on a new page",
    "suggested time management", "write neatly and legibly",
    "copyright reserved", "please turn over", "table of contents",
    "the number of marks allocated", "serves as a guide to the expected",
    "answer the questions set on both",
]


def _extract_page_questions(page_b64: str, page_url: Optional[str],
                             subject: str, grade: str,
                             page_num: int) -> list[dict]:
    """
    Two-phase extraction:
      Phase 1 — classify page type (cheap call)
      Phase 2 — extract questions (only for question/mixed pages)
    """
    cat   = _subject_category(subject)
    hints = _PARSING_HINTS.get(cat, "")

    # ── Phase 1: classify ─────────────────────────────────────────────────
    page_type = "questions"  # safe default
    try:
        raw = ai_vision(page_b64, _PAGE_CLASSIFIER_PROMPT)
        raw = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
        m   = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            obj       = json.loads(m.group())
            page_type = obj.get("page_type", "questions")
            logger.info("[Vision] Page %d → '%s': %s",
                        page_num, page_type, obj.get("reason", ""))
    except Exception as e:
        logger.warning("[Vision] Page %d classify failed: %s", page_num, e)

    if page_type in _SKIP_PAGE_TYPES:
        logger.info("[Vision] Page %d skipped (%s)", page_num, page_type)
        return []

    # ── Phase 2: extract questions ────────────────────────────────────────
    prompt = _build_extraction_prompt(subject, grade, page_num, hints)
    try:
        raw = ai_vision(page_b64, prompt)
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\s*```\s*$",        "", raw, flags=re.MULTILINE)
        m   = re.search(r"\[.*\]", raw, re.DOTALL)
        if not m:
            return []

        parsed = json.loads(m.group())
        if not isinstance(parsed, list):
            return []

        valid = []
        for q in parsed:
            # Normalise question_number
            qn = re.sub(r"^Q\.?\s*", "", str(q.get("question_number", "")).strip(),
                        flags=re.IGNORECASE)
            if not qn:
                continue  # no question number = heading or instruction

            # Normalise section
            sec = str(q.get("section", "A")).strip().upper()
            q["section"]         = sec if re.match(r"^[A-Z]$", sec) else "A"
            q["question_number"] = qn

            # Normalise marks
            try:
                q["marks"] = int(q.get("marks", 1))
            except (TypeError, ValueError):
                q["marks"] = 1

            # Drop if question text is actually an instruction
            qt = str(q.get("question", "")).strip().lower()
            if any(p in qt for p in _INSTRUCTION_PHRASES):
                logger.debug("[Vision] Dropped instruction as question: %s", qn)
                continue

            # Attach page image URL for visual questions
            if page_url and q.get("has_visual"):
                q["questionImageUrl"] = page_url

            valid.append(q)

        logger.info("[Vision] Page %d → %d questions (%d dropped)",
                    page_num, len(valid), len(parsed) - len(valid))
        return valid

    except json.JSONDecodeError as e:
        logger.error("[Vision] Page %d JSON error: %s", page_num, e)
    except Exception as e:
        logger.error("[Vision] Page %d: %s", page_num, e)
    return []


# ══════════════════════════════════════════════════════════════════════════════
# MERGE + DEDUPLICATE
# ══════════════════════════════════════════════════════════════════════════════

def _normalise_qnum(qn: str) -> str:
    s = str(qn).lower().strip()
    s = re.sub(r"^(question|q|ques|no|nr)[\s.\-]*", "", s)
    s = re.sub(r"[^a-z0-9]", "", s)
    return s


def _merge_pages(page_results: list[list[dict]]) -> list[dict]:
    seen:  dict[str, int] = {}
    final: list[dict]     = []
    for page_qs in page_results:
        for q in page_qs:
            key = (_normalise_qnum(q.get("question_number", ""))
                   or q.get("question", "")[:60].strip())
            if key and key in seen:
                final[seen[key]] = q   # later version wins (more complete)
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
def _render_pages(pdf_bytes: bytes) -> list[tuple[int, bytes, str]]:
    """Returns (page_num, png_bytes, native_text) for each page."""
    pages = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        mat = fitz.Matrix(_PAGE_DPI / 72, _PAGE_DPI / 72)
        for i, page in enumerate(doc):
            try:
                native_text = page.get_text().strip()
                pix         = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
                png         = pix.tobytes("png")
                pages.append((i + 1, png, native_text))
            except Exception as e:
                logger.error("[Render] Page %d: %s", i + 1, e)
        doc.close()
    except Exception as e:
        logger.error("[Render] PDF open failed: %s", e)
    return pages


def extract_questions_from_file(
    file_bytes:    bytes,
    filename:      str,
    subject:       str,
    grade:         str,
    exam_id:       str = "",
    school_folder: str = "shared",
) -> list[dict]:
    """
    Primary entry point. DOCX/ODT → LibreOffice → PDF → vision AI → questions.
    Falls back to text extraction + LLM parse if LibreOffice unavailable.
    """
    lower = filename.lower()

    if lower.endswith(".pdf"):
        pdf_bytes = file_bytes
    else:
        logger.info("[Extract] Converting %s → PDF via LibreOffice", filename)
        pdf_bytes = _convert_to_pdf(file_bytes, filename)

    if pdf_bytes:
        raw_pages = _render_pages(pdf_bytes)
        if raw_pages:
            page_results: list[list[dict]] = []
            for page_num, png_bytes, native_text in raw_pages:
                # If page has plenty of native text, use text extraction (no API call)
                if len(native_text) > 300:
                    logger.info("[Extract] Page %d: using native text (%d chars)", page_num, len(native_text))
                    # Parse the native text directly for this page
                    # (falls through to parse_questions_universal at the end)
                    page_results.append([])  # vision skipped
                    continue

                # Otherwise use vision (for diagram-heavy pages)
                page_b64 = base64.b64encode(png_bytes).decode()
                page_url = (_upload_page_image(...) if exam_id else None)
                questions = _extract_page_questions(page_b64, page_url, subject, grade, page_num)
                page_results.append(questions)
                time.sleep(0.5)
            questions = _merge_pages(page_results)
            if questions:
                logger.info("[Extract] ✓ %d questions via render-first pipeline", len(questions))
                return questions
            logger.warning("[Extract] Vision returned 0 questions — trying text fallback")

    # Text fallback
    logger.info("[Extract] Text fallback: %s", filename)
    text = extract_text_from_file(file_bytes, filename, subject)
    if text.strip():
        return parse_questions_universal(text, subject, grade)
    logger.error("[Extract] All methods exhausted for %s", filename)
    return []


# ══════════════════════════════════════════════════════════════════════════════
# TEXT EXTRACTION — fast path for memos
# ══════════════════════════════════════════════════════════════════════════════

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
    try:
        doc  = fitz.open(stream=file_bytes, filetype="pdf")
        text = "".join(page.get_text() + "\n" for page in doc)
        doc.close()
        if len(text.strip()) > 100:
            return text
        logger.info("[PDF text] No native text — running vision OCR")
        doc, all_text = fitz.open(stream=file_bytes, filetype="pdf"), ""
        for i, page in enumerate(doc):
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
            img = base64.b64encode(pix.tobytes("png")).decode()
            try:
                result    = ai_vision(img, f"{subject} exam memo. Extract ALL text. Plain text only.")
                all_text += result + "\n\n"
            except Exception as e:
                logger.error("[PDF OCR] Page %d: %s", i + 1, e)
        doc.close()
        return all_text
    except Exception as e:
        logger.error("[PDF text] %s", e)
        return ""


def extract_text_from_file(file_bytes: bytes, filename: str,
                           subject: str = "General") -> str:
    """Text-only extraction — used for memo parsing."""
    lower = filename.lower()
    if lower.endswith(".docx"):
        return _docx_text(file_bytes)
    if lower.endswith(".odt"):
        return _odt_text(file_bytes)
    if lower.endswith(".pdf"):
        return _pdf_text(file_bytes, subject)
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
    """LLM-based question parser for plain text. Chunked to stay within token limits."""
    cat     = _subject_category(subject)
    hints   = _PARSING_HINTS.get(cat, "")
    all_qs: list[dict] = []
    seen:   set[str]   = set()

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
                            key = (_normalise_qnum(q.get("question_number", ""))
                                   or q.get("question", "")[:60])
                            if key not in seen:
                                seen.add(key)
                                all_qs.append(q)
                break
            except Exception as e:
                if "413" in str(e) and attempt == 0:
                    logger.warning("[Parser] Chunk %d too large, retrying at half", idx + 1)
                    continue
                logger.error("[Parser] Chunk %d: %s", idx + 1, e)
                break

    logger.info("[Parser] %d questions extracted (text fallback)", len(all_qs))
    return all_qs