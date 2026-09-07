"""
extract_exams_v2.py — Eduket batch exam-bank builder  v4.0  (Groq-primary hybrid)
═══════════════════════════════════════════════════════════════════════════════
Offline tool for seeding the exam bank from a folder of papers. Separate from
the production upload path in app.py, which handles teacher uploads.

WHAT CHANGED FROM v3
═════════════════════

REVERTED TO GROQ AS THE PRIMARY PROVIDER, WITH GEMINI AS A PAID RESCUE.
v3 moved everything to Gemini because Groq's old Llama 3.3 70B Versatile
model hit a 12,000 TPM ceiling. That model is now decommissioned (Aug 16,
2026) in favour of GPT OSS 120B / Qwen3.6 27B on Groq, which is why this is
worth reverting: cheaper/faster Groq handles the large majority of papers,
and Gemini is only paid for when a paper genuinely needs it.

Routing, per call (exam extraction AND memo extraction):
  1. Groq is tried first — see GROQ_MODEL_EXTRACT below.
  2. Falls back to Gemini when any of the following is true:
       - The rolling TPM budget (GROQ_TPM_BUDGET) would be exceeded by this
         call's estimated token count — skips Groq entirely rather than
         waiting for a 429.
       - Groq actually returns a rate-limit (429) or context-length error —
         triggers a cooldown (GROQ_COOLDOWN_SECONDS) during which further
         calls in this run skip straight to Gemini.
       - The input is a PDF that yields little or no extractable text (a
         scanned/image paper) — only Gemini reads a PDF's layout/images
         natively; Groq is text-only, so a text-extraction step
         (pdf_to_text, via pypdf) runs first for the Groq path, and a
         near-empty result means Gemini's native document input is the only
         option, independent of TPM budget.
  3. Whichever provider is used, output is normalised to the same schema
     shape before it reaches extract_exam()/extract_memo() — see
     _parse_structured_json() — since only Gemini's response_schema
     guarantees schema-conformant JSON; Groq's JSON mode only guarantees
     *valid* JSON, so the Groq path gets a light repair/validation pass that
     the Gemini path doesn't need.

MARKING / ANALYSIS: the docstring's DUPLICATION WARNING below still applies —
app.py (not this file) does per-submission marking and AI analysis. If it
migrates to this same Groq-first/Gemini-rescue pattern, keep the routing
logic (TPM budget + cooldown) here and there in sync, or better, factor it
into the shared module the warning below already asks for.

NATIVE DOCUMENT INPUT (Gemini only): PDFs and Word files still drop straight
into the input folder — no pre-chunking step. Legacy processed/*.json chunk
files still work, stitched and sent as text, and are the cheapest/fastest
path since no PDF-to-text step is needed for Groq.

STRUCTURED OUTPUT: guaranteed schema-conformant on the Gemini path via
response_schema. On the Groq path, JSON mode guarantees valid JSON only, so
_parse_structured_json() does a defensive parse (including a codefence-strip
fallback, since Groq's JSON mode is looser than Gemini's) before the rest of
the pipeline runs.

STRUCTURE PRESERVED: sections with titles and instructions, question
numbering as printed, shared source material once, MCQ options, matching
columns, tables as markdown, maths as LaTeX, figures described.

DUPLICATION WARNING
═══════════════════
EXAM_SCHEMA, MEMO_SCHEMA and the two prompts below are copies of the ones in
app.py. Two copies WILL drift — that is how this codebase ended up with two
ai_text() implementations, one of which silently swallowed the Gemini fallback.
Move them into a shared schemas.py that both files import. Kept inline here
only so this script runs standalone.

CONFIRM BEFORE RUNNING
═══════════════════════
GROQ_MODEL_EXTRACT below defaults to "openai/gpt-oss-120b" — Groq's hosted
slug for GPT OSS 120B, the model named in Groq's Llama 3.3 70B decommission
notice. Confirm the exact model string against
https://console.groq.com/docs/models before running at volume: model slugs
on Groq's catalog change, and this default may be stale by the time you read
it. Same caution applies to GROQ_TPM_BUDGET — set conservatively below;
raise it once you've confirmed your account's actual TPM limit for this
model in the Groq console, rather than trusting the default here.

Usage:
    python extract_exams_v2.py

Requires:  pip install google-genai groq pypdf
Env:       GEMINI_API_KEY, GROQ_API_KEY, optionally GEMINI_MODEL_EXTRACT,
           GROQ_MODEL_EXTRACT, GROQ_TPM_BUDGET, GROQ_COOLDOWN_SECONDS
"""
import os
import re
import time
import json
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum

from dotenv import load_dotenv
from google import genai
from google.genai import types
from groq import Groq, RateLimitError as GroqRateLimitError, \
    APIStatusError as GroqAPIStatusError, APIError as GroqAPIError
from typing import Any

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None  # PDF text extraction for the Groq path degrades to
                       # "always use Gemini for PDFs" if pypdf isn't installed

load_dotenv()

# ── Gemini: paid rescue provider ─────────────────────────────────────────
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY is not set.")

MODEL_NAME = os.getenv("GEMINI_MODEL_EXTRACT", "gemini-3.6-flash")

# ── Groq: primary provider ───────────────────────────────────────────────
# See the CONFIRM BEFORE RUNNING note in the module docstring — verify this
# slug and the TPM budget against your Groq console before a large run.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_EXTRACT = os.getenv("GROQ_MODEL_EXTRACT", "openai/gpt-oss-120b")
GROQ_TPM_BUDGET = int(os.getenv("GROQ_TPM_BUDGET", "50000"))   # conservative default
GROQ_COOLDOWN_SECONDS = int(os.getenv("GROQ_COOLDOWN_SECONDS", "90"))
GROQ_MIN_PDF_CHARS = 200   # below this, a "PDF" is treated as unreadable/scanned

INPUT_FOLDER = os.getenv("INPUT_FOLDER", "processed")
OUTPUT_FOLDER = "exams"
TRACK_FILE = "processed_exams.json"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

_client: genai.Client | None = None
_groq_client: Groq | None = None
gc = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=API_KEY)
    return _client


def groq_client() -> Groq | None:
    global _groq_client
    if not GROQ_API_KEY:
        return None
    if _groq_client is None:
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


# ═══════════════════════════════════════════════════════════════════════════════
# GROQ ROUTING — rolling TPM budget + post-429 cooldown
# ═══════════════════════════════════════════════════════════════════════════════
# In-process only: a fresh run starts with a clean budget. Good enough for a
# batch script invoked periodically; a long-lived server process (app.py)
# would want this backed by something shared across workers instead.

_groq_usage_log: list[tuple[float, int]] = []   # [(timestamp, tokens), ...]
_groq_cooldown_until: float = 0.0


def _estimate_tokens(text: str) -> int:
    # Rough ~4 chars/token estimate. Good enough for a pre-flight budget
    # check; the real count comes back in usage_metadata after the call.
    return max(1, len(text) // 4)


def _groq_budget_ok(estimated_tokens: int) -> bool:
    now = time.time()
    global _groq_usage_log
    _groq_usage_log = [(t, n) for t, n in _groq_usage_log if now - t < 60]
    used = sum(n for _, n in _groq_usage_log)
    return (used + estimated_tokens) <= GROQ_TPM_BUDGET


def _groq_record_usage(tokens: int) -> None:
    _groq_usage_log.append((time.time(), tokens))


def _groq_in_cooldown() -> bool:
    return time.time() < _groq_cooldown_until


def _groq_start_cooldown() -> None:
    global _groq_cooldown_until
    _groq_cooldown_until = time.time() + GROQ_COOLDOWN_SECONDS
    print(f"    Groq cooldown started — routing to Gemini for the next "
          f"{GROQ_COOLDOWN_SECONDS}s")


def pdf_to_text(pdf_bytes: bytes) -> str:
    """
    Text-only extraction for the Groq path. Deliberately dumb — no layout
    reconstruction, no OCR. A paper that comes back too short from this is
    assumed scanned/image-based and routed to Gemini instead, which reads
    the PDF's actual layout and any embedded images natively.
    """
    if not PdfReader:
        return ""
    try:
        from io import BytesIO
        reader = PdfReader(BytesIO(pdf_bytes))
        return "\n".join((page.extract_text() or "") for page in reader.pages).strip()
    except Exception as e:
        print(f"    pypdf text extraction failed: {type(e).__name__}: {e}")
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class QuestionType(Enum):
    MCQ = "mcq"
    MATCHING = "matching"
    TRUE_FALSE = "true_false"
    OPEN = "open"
    CALCULATION = "calculation"
    SHORT_ANSWER = "short_answer"
    ESSAY = "essay"
    COMPREHENSION = "comprehension"
    DIAGRAM_LABEL = "diagram_label"
    TABLE_COMPLETION = "table_completion"
    MULTI_PART = "multi_part"
    UNKNOWN = "unknown"


@dataclass
class Question:
    id: int
    question_number: str
    parent_question: str = ""
    parent_context: Optional[str] = None
    instructions: Optional[str] = None
    question: str = ""
    question_type: str = "open"
    marks: int = 1
    memo: Any = ""
    options: Optional[dict] = None
    column_a: Optional[list] = None
    column_b: Optional[list] = None
    # Replaces the old DiagramRef/TableRef dataclasses. Gemini describes the
    # figure and renders the table directly, so a reference object with an id
    # and a caption no longer earns its keep.
    table_markdown: Optional[str] = None
    formula: Optional[str] = None
    has_visual: bool = False
    visual_description: Optional[str] = None
    section: str = "A"
    sub_parts: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


@dataclass
class Section:
    section: str
    section_title: str = ""
    section_instructions: str = ""
    total_marks: Optional[int] = None
    questions: list = field(default_factory=list)

    def to_dict(self):
        return {
            "section": self.section,
            "section_title": self.section_title,
            "section_instructions": self.section_instructions,
            "total_marks": self.total_marks,
            "questions": [q.to_dict() for q in self.questions],
        }


@dataclass
class ExamMetadata:
    subject: str = ""
    subject_code: str = ""
    grade: str = ""
    year: str = ""
    paper_number: str = ""
    exam_type: str = ""
    language: str = "English"
    time_allocation: str = ""
    total_marks: Optional[int] = None
    instructions: str = ""

    def to_dict(self):
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMAS — keep in step with app.py, or move both to a shared module
# ═══════════════════════════════════════════════════════════════════════════════
# NOTE on `contexts`: a LIST of {group, kind, text}, not a map. response_schema
# needs concrete property names, so an object keyed by arbitrary question
# numbers is unreliable.

QUESTION_PROPERTIES = {
    "question_number": {"type": "string",
                        "description": "Exactly as printed: 1.1, 2.3.1, 4.7.1"},
    "parent_question": {"type": "string",
                        "description": "The group heading, e.g. 'QUESTION 1'"},
    "context_ref": {"type": "string", "nullable": True,
                    "description": "Group key of the shared source material, or null"},
    "instructions": {"type": "string", "nullable": True,
                     "description": "Directive lines like 'Refer to paragraph 2.'"},
    "question": {"type": "string",
                 "description": "Question text verbatim, without its number or mark allocation"},
    "question_type": {
        "type": "string",
        "enum": ["mcq", "true_false", "matching", "calculation", "essay",
                 "short_answer", "comprehension", "diagram_label",
                 "table_completion", "open"],
    },
    "marks": {"type": "integer"},
    "options": {
        "type": "array", "nullable": True,
        "description": "MCQ options in printed order",
        "items": {
            "type": "object",
            "properties": {"key": {"type": "string"}, "value": {"type": "string"}},
            "required": ["key", "value"],
        },
    },
    "column_a": {"type": "array", "nullable": True, "items": {"type": "string"}},
    "column_b": {"type": "array", "nullable": True, "items": {"type": "string"}},
    "table_markdown": {"type": "string", "nullable": True,
                       "description": "Any table the question depends on, as markdown"},
    "formula": {"type": "string", "nullable": True,
                "description": "Formulae in LaTeX when the question is mathematical"},
    "has_visual": {"type": "boolean",
                   "description": "True when the question depends on a diagram, map or graph"},
    "visual_description": {"type": "string", "nullable": True,
                           "description": "Description of the figure so the question stays answerable"},
}

EXAM_SCHEMA = {
    "type": "object",
    "properties": {
        "metadata": {
            "type": "object",
            "properties": {
                "subject":         {"type": "string"},
                "grade":           {"type": "string"},
                "year":            {"type": "string"},
                "paper_number":    {"type": "string"},
                "exam_type":       {"type": "string"},
                "language":        {"type": "string"},
                "total_marks":     {"type": "integer", "nullable": True},
                "time_allocation": {"type": "string", "nullable": True},
                "instructions":    {"type": "string", "nullable": True},
            },
        },
        "contexts": {
            "type": "array",
            "description": "Shared source material, each appearing exactly once",
            "items": {
                "type": "object",
                "properties": {
                    "group": {"type": "string", "description": "Question group served: '1', '2'"},
                    "kind": {"type": "string",
                             "enum": ["passage", "extract", "case_study", "source",
                                      "scenario", "data_set", "cartoon", "other"]},
                    "text": {"type": "string",
                             "description": "The material VERBATIM, every paragraph, no summary"},
                },
                "required": ["group", "text"],
            },
        },
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "section":              {"type": "string"},
                    "section_title":        {"type": "string"},
                    "section_instructions": {"type": "string", "nullable": True},
                    "total_marks":          {"type": "integer", "nullable": True},
                    "questions": {
                        "type": "array",
                        "items": {"type": "object",
                                  "properties": QUESTION_PROPERTIES,
                                  "required": ["question_number", "question",
                                               "question_type", "marks"]},
                    },
                },
                "required": ["section", "questions"],
            },
        },
    },
    "required": ["sections"],
}

MEMO_SCHEMA = {
    "type": "object",
    "properties": {
        "subject": {"type": "string"},
        "year":    {"type": "string"},
        "paper":   {"type": "string"},
        "answers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question_number": {"type": "string"},
                    "answer":          {"type": "string"},
                },
                "required": ["question_number", "answer"],
            },
        },
    },
    "required": ["answers"],
}


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPTS
# ═══════════════════════════════════════════════════════════════════════════════

EXTRACTION_PROMPT = """You are parsing a South African CAPS/NSC exam paper.

Reproduce the paper's STRUCTURE faithfully. Do not summarise anything.

SECTIONS
Keep the sections in printed order. Record each section's letter, its title as
printed (e.g. "SECTION A: COMPREHENSION"), its instruction line, and its total.

QUESTIONS
Extract EVERY question in printed order, numbered exactly as printed (1.1,
2.3.1, 4.7.1). Preserve wording verbatim — never rephrase or shorten. Take marks
from the brackets after each question. Do NOT put the question number or the
mark allocation inside the question text.

Directive lines such as "Refer to paragraph 2." or "Write down only the LETTER"
belong in "instructions", not in the question.

SHARED SOURCE MATERIAL
Papers print material once above a group of questions: a reading passage, a
literary extract, a case study, a newspaper source, a described cartoon, a
scenario, or a data set. Every question in that group is unanswerable without it.

List each piece ONCE in "contexts" with the group number it serves. Copy it
VERBATIM — every paragraph, including the source line. Never summarise, never
truncate, never write "see above". Set each question's "context_ref" to that
group. Do not repeat the material inside a question. Use null for context_ref
only when a question is genuinely self-contained.

QUESTION TYPES
- "Choose the correct answer" / A,B,C,D -> mcq, options in printed order
- "Match COLUMN A with COLUMN B" -> matching, both columns as arrays
- "Write TRUE or FALSE" -> true_false
- "Calculate" / "Determine" / "Show ALL calculations" -> calculation
- "Discuss"/"Evaluate"/"Analyse" over 10 marks -> essay
- "State"/"Name"/"List" at 5 marks or fewer -> short_answer
- Questions on a passage or source -> comprehension
- "Label the diagram" / "Study the diagram" -> diagram_label
- "Complete the table" -> table_completion
- Anything else -> open

RICH CONTENT
Tables a question depends on go in "table_markdown".
Mathematics goes in "formula" as LaTeX.
If a question depends on a diagram, map, graph or photograph, set has_visual
true and describe it in "visual_description" fully enough that the question can
still be answered.

Subject: {subject} | Grade: {grade}"""

MEMO_PROMPT = """You are parsing an NSC exam MARKING MEMORANDUM.

Extract EVERY answer, keyed by the question number exactly as printed.
- Multiple choice: the letter only, e.g. "C"
- Matching: the letter only, e.g. "R"
- True/False: "True", or "False - <the correction>"
- Calculations: full working and the final answer
- Open and essay: the marking points, one per line
- Where alternatives are accepted, separate them with " OR "

Do not invent answers for questions the memo does not cover.

Subject: {subject}"""


# ═══════════════════════════════════════════════════════════════════════════════
# INPUT HANDLING — PDFs and Word files natively, legacy chunk JSON as text
# ═══════════════════════════════════════════════════════════════════════════════

PDF_EXTS = {".pdf"}
WORD_EXTS = {".docx", ".doc", ".odt", ".rtf"}
CHUNK_EXTS = {".json"}
ALL_EXTS = PDF_EXTS | WORD_EXTS | CHUNK_EXTS


def _lo_binary() -> str | None:
    return shutil.which("libreoffice") or shutil.which("soffice")


def convert_to_pdf(path: str) -> bytes | None:
    """
    Word family -> PDF via LibreOffice.
    Unique profile per invocation (a shared one collides on concurrent runs),
    and no --infilter (that flag takes an OUTPUT filter and breaks the read).
    soffice exits 0 even on failure, so the output file is the only real signal.
    """
    cmd = _lo_binary()
    if not cmd:
        print("    LibreOffice not installed — cannot convert Word files")
        return None

    with tempfile.TemporaryDirectory() as tmp:
        profile = os.path.join(tmp, "loprofile")
        try:
            result = subprocess.run(
                [cmd, "--headless", "--norestore", "--nofirststartwizard",
                 f"-env:UserInstallation=file://{profile}",
                 "--convert-to", "pdf:writer_pdf_Export",
                 "--outdir", tmp, path],
                timeout=180, capture_output=True,
                env={**os.environ, "HOME": tmp},
            )
        except subprocess.TimeoutExpired:
            print(f"    LibreOffice timeout on {path}")
            return None

        out = os.path.join(tmp, Path(path).stem + ".pdf")
        if os.path.exists(out):
            with open(out, "rb") as f:
                return f.read()

        print(f"    LibreOffice produced no PDF | exit={result.returncode} | "
              f"{result.stderr.decode(errors='replace')[:200]}")
        return None


def load_source(filename: str):
    """
    Return ("pdf", bytes) or ("text", str) for any supported input.
    Legacy chunk JSON is stitched into plain text; everything else becomes PDF.
    """
    path = os.path.join(INPUT_FOLDER, filename)
    ext = Path(filename).suffix.lower()

    if ext in CHUNK_EXTS:
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as e:
            print(f"    Failed to load {filename}: {e}")
            return None, None
        if not isinstance(data, list):
            return None, None
        text = "\n".join(c.get("content", "").strip()
                         for c in data if c.get("content", "").strip())
        return ("text", text) if text else (None, None)

    if ext in PDF_EXTS:
        with open(path, "rb") as f:
            data = f.read()
        return ("pdf", data) if data.startswith(b"%PDF") else (None, None)

    if ext in WORD_EXTS:
        pdf = convert_to_pdf(path)
        return ("pdf", pdf) if pdf else (None, None)

    return None, None


# ═══════════════════════════════════════════════════════════════════════════════
# PROVIDER CALLS
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_structured_json(raw: str) -> dict:
    """
    Gemini's response_schema guarantees valid, schema-shaped JSON — json.loads
    is enough. Groq's JSON mode only guarantees *valid* JSON (not shape), and
    has occasionally been seen wrapping output in ```json fences despite
    response_format={"type": "json_object"}. This strips those defensively;
    it does NOT validate the shape against EXAM_SCHEMA/MEMO_SCHEMA — that's
    left to extract_exam()/extract_memo(), which already tolerate missing
    keys via .get(...) with defaults.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
    return json.loads(text)


def _generate_gemini(kind: str, payload, prompt: str, schema: dict, max_tokens: int, _retry: bool = True):
    if kind == "pdf":
        contents = [types.Part.from_bytes(data=payload, mime_type="application/pdf"), prompt]
    else:
        contents = f"{prompt}\n\nTEXT TO EXTRACT:\n{payload}"

    try:
        resp = client().models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=max_tokens,
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )
    except RuntimeError as e:
        if _retry and "client has been closed" in str(e):
            print("    Gemini client was closed — rebuilding and retrying once")
            global _client
            _client = None
            return _generate_gemini(kind, payload, prompt, schema, max_tokens, _retry=False)
        raise

    try:
        u = resp.usage_metadata
        print(f"    [gemini] tokens in={u.prompt_token_count} out={u.candidates_token_count}")
    except Exception:
        pass
    return json.loads(resp.text)

GROQ_MAX_OUTPUT_TOKENS = 65536  # openai/gpt-oss-120b's actual ceiling per Groq's docs

def _generate_groq(text, prompt, schema=None, max_tokens=None):
    # Ensure text/payload is converted to string content
    if isinstance(text, bytes):
        text_content = text.decode("utf-8", errors="ignore")
    elif not isinstance(text, str):
        text_content = str(text) if text is not None else ""
    else:
        text_content = text

    groq_max_tokens = min(max_tokens, GROQ_MAX_OUTPUT_TOKENS) if max_tokens else GROQ_MAX_OUTPUT_TOKENS

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": str(prompt)},
        {"role": "user", "content": text_content}
    ]

    resp = gc.chat.completions.create(
        model=GROQ_MODEL_EXTRACT,
        messages=messages,
        max_tokens=groq_max_tokens,
        response_format={"type": "json_object"},
    )

    raw = resp.choices[0].message.content
    return _parse_structured_json(raw)

def _generate(kind: str, payload, prompt: str, schema: dict, max_tokens: int = 65536):
    """
    One structured call, routed Groq-first with a Gemini rescue.
    kind is 'pdf' or 'text' — matches load_source()'s return.

    Order of decisions:
      1. No GROQ_API_KEY configured at all -> Gemini, unconditionally.
      2. kind == 'pdf' -> attempt pdf_to_text(); if that yields too little
         text (scanned/image paper), Groq can't read it at all -> Gemini,
         unconditionally, using the original PDF bytes (native document
         input).
      3. In an active post-429 cooldown, or this call's estimated tokens
         would exceed the rolling TPM budget -> skip Groq, use Gemini.
      4. Otherwise try Groq. On RateLimitError or a context-length/413-style
         APIStatusError, start a cooldown and fall back to Gemini for THIS
         call. Any other Groq error also falls back, but does not trigger
         the cooldown (it's not evidence Groq is out of budget).
    """
    if not GROQ_API_KEY:
        return _generate_gemini(kind, payload, prompt, schema, max_tokens)

    if kind == "pdf":
        text = pdf_to_text(payload)
        if len(text) < GROQ_MIN_PDF_CHARS:
            print("    PDF text extraction too thin (likely scanned) — Gemini required")
            return _generate_gemini(kind, payload, prompt, schema, max_tokens)
    else:
        text = payload

    estimated = _estimate_tokens(prompt) + _estimate_tokens(text)

    if _groq_in_cooldown():
        print("    Groq in cooldown — routing to Gemini")
        return _generate_gemini(kind, payload, prompt, schema, max_tokens)

    if not _groq_budget_ok(estimated):
        print(f"    Estimated {estimated} tokens would exceed the "
              f"{GROQ_TPM_BUDGET} TPM budget — routing to Gemini")
        return _generate_gemini(kind, payload, prompt, schema, max_tokens)

    try:
        return _generate_groq(text, prompt, schema, max_tokens)
    except GroqRateLimitError as e:
        print(f"    Groq rate limit hit: {e}")
        _groq_start_cooldown()
        return _generate_gemini(kind, payload, prompt, schema, max_tokens)
    except GroqAPIStatusError as e:
        if e.status_code in (413, 429):
            print(f"    Groq status {e.status_code} (likely too large for context): {e}")
            _groq_start_cooldown()
        else:
            print(f"    Groq error {e.status_code}, falling back this call only: {e}")
        return _generate_gemini(kind, payload, prompt, schema, max_tokens)
    except (GroqAPIError, json.JSONDecodeError) as e:
        print(f"    Groq call failed ({type(e).__name__}: {e}), falling back to Gemini")
        return _generate_gemini(kind, payload, prompt, schema, max_tokens)


# ═══════════════════════════════════════════════════════════════════════════════
# TYPE INFERENCE — fallback when the model returns a bare "open"
# ═══════════════════════════════════════════════════════════════════════════════

def infer_question_type(question_text, options=None, column_a=None, instructions=""):
    text_lower = (question_text + " " + (instructions or "")).lower()

    if options and isinstance(options, dict) and len(options) >= 2:
        return QuestionType.MCQ.value
    if column_a and isinstance(column_a, list) and len(column_a) >= 2:
        return QuestionType.MATCHING.value
    if "match column" in text_lower or ("column a" in text_lower and "column b" in text_lower):
        return QuestionType.MATCHING.value
    if "true or false" in text_lower:
        return QuestionType.TRUE_FALSE.value
    if any(w in text_lower for w in ("show all calculations", "calculate", "determine")):
        if any(c in text_lower for c in ("=", "+", "-", "times", "divide", "$")):
            return QuestionType.CALCULATION.value
    if any(w in text_lower for w in ("label the diagram", "study the diagram", "figure")):
        return QuestionType.DIAGRAM_LABEL.value
    if "complete the table" in text_lower or "use the table" in text_lower:
        return QuestionType.TABLE_COMPLETION.value
    if any(w in text_lower for w in ("discuss", "evaluate", "analyse", "critically")):
        m = re.search(r'\((\d+)\)', question_text)
        if m and int(m.group(1)) > 10:
            return QuestionType.ESSAY.value
    if any(w in text_lower for w in ("read the passage", "refer to", "according to")):
        return QuestionType.COMPREHENSION.value
    if any(w in text_lower for w in ("briefly", "state", "name", "list")):
        m = re.search(r'\((\d+)\)', question_text)
        if m and int(m.group(1)) <= 5:
            return QuestionType.SHORT_ANSWER.value
    return QuestionType.OPEN.value


# ═══════════════════════════════════════════════════════════════════════════════
# EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

MIN_PASSAGE_CHARS = 160


def _recover_passage(raw_text: str, group_number: str) -> str:
    """
    Safety net for the text path: pull the prose between 'QUESTION n' and its
    first sub-question. Verified against the Grade 9 Holes paper — recovers the
    full 5,146-character extract and rejects a short instruction line.
    Not needed on the PDF path, where the model sees the layout directly.
    """
    if not group_number or not raw_text:
        return ""
    pattern = (rf'QUESTION\s+{re.escape(str(group_number))}\b(.*?)'
               rf'(?=\n\s*{re.escape(str(group_number))}\.\d)')
    m = re.search(pattern, raw_text, re.DOTALL | re.IGNORECASE)
    if not m:
        return ""
    body = re.sub(r'\n{3,}', '\n\n', m.group(1).strip())
    body = re.sub(r'Refer to paragraph\s+\d+\.?\s*$', '', body).strip()
    return body if len(body) >= MIN_PASSAGE_CHARS else ""


def extract_exam(kind: str, payload, subject: str, grade: str):
    """
    One call for the whole paper.
    Returns (ExamMetadata, [Section], stats dict).
    """
    result = _generate(
        kind, payload,
        EXTRACTION_PROMPT.format(subject=subject, grade=grade),
        EXAM_SCHEMA,
    )

    metadata = ExamMetadata(**{
        k: v for k, v in (result.get("metadata") or {}).items()
        if k in ExamMetadata.__dataclass_fields__
    })

    # Index all source material by group without overwriting.
    # Group by question group, keeping ALL passages in a list for that group.
    contexts: dict[str, list[str]] = {}
    total_source_items = 0

    for c in (result.get("contexts") or []):
        g = str(c.get("group", "")).strip()
        t = (c.get("text") or "").strip()
        if g and t:
            if g not in contexts:
                contexts[g] = []
            contexts[g].append(t)
            total_source_items += 1

    raw_text = payload if kind == "text" else ""
    linked = recovered = 0
    qid = 0
    sections = []

    for sec_data in result.get("sections", []):
        questions = []
        for q_data in sec_data.get("questions", []):
            qid += 1
            qnum = str(q_data.get("question_number") or "").strip()

            # Resolve shared material: explicit ref, then the leading number
            ref = str(q_data.get("context_ref") or (qnum.split(".")[0] if qnum else "")).strip()

            ctx = ""
            if ref in contexts:
                # Combine all passages linked to this group into one string
                ctx = "\n\n---\n\n".join(contexts[ref])
                linked += 1
            elif raw_text:
                recovered_passage = _recover_passage(raw_text, ref)
                if recovered_passage:
                    contexts[ref] = [recovered_passage]
                    ctx = recovered_passage
                    recovered += 1

            # options arrive as [{key, value}] — the dataclass wants a dict
            opts = q_data.get("options")
            options = None
            if isinstance(opts, list) and opts:
                options = {o["key"]: o["value"] for o in opts if o.get("key")}

            q_type = q_data.get("question_type", "open")
            if q_type in ("open", "unknown", "", None):
                q_type = infer_question_type(
                    q_data.get("question", ""), options,
                    q_data.get("column_a"), q_data.get("instructions", ""),
                )

            try:
                marks = max(1, int(q_data.get("marks") or 1))
            except (TypeError, ValueError):
                marks = 1

            questions.append(Question(
                id=qid,
                question_number=qnum or str(qid),
                parent_question=q_data.get("parent_question", ""),
                parent_context=ctx or None,
                instructions=q_data.get("instructions"),
                question=(q_data.get("question") or "").strip(),
                question_type=q_type,
                marks=marks,
                options=options,
                column_a=q_data.get("column_a"),
                column_b=q_data.get("column_b"),
                table_markdown=q_data.get("table_markdown"),
                formula=q_data.get("formula"),
                has_visual=bool(q_data.get("has_visual")),
                visual_description=q_data.get("visual_description"),
                section=sec_data.get("section", "A"),
            ))

        sections.append(Section(
            section=sec_data.get("section", "A"),
            section_title=sec_data.get("section_title", ""),
            section_instructions=sec_data.get("section_instructions") or "",
            total_marks=sec_data.get("total_marks"),
            questions=questions,
        ))

    total_q = sum(len(s.questions) for s in sections)
    with_ctx = sum(1 for s in sections for q in s.questions
                   if (q.parent_context or "").strip())

    stats = {
        "source_items": total_source_items,
        "linked": linked,
        "recovered": recovered,
        "questions_with_context": with_ctx,
        "total_questions": total_q,
    }
    return metadata, sections, stats


def extract_memo(kind: str, payload, subject: str) -> dict:
    """Returns {question_number: answer} keyed exactly as printed."""
    result = _generate(kind, payload, MEMO_PROMPT.format(subject=subject),
                       MEMO_SCHEMA, max_tokens=16384)
    answers = {}
    for row in (result.get("answers") or []):
        qn = str(row.get("question_number", "")).strip()
        ans = (row.get("answer") or "").strip()
        if qn and ans and qn not in answers:
            answers[qn] = ans
    return answers


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def validate_exam_structure(sections):
    """
    Light sanity pass. The heavy regex recovery that used to live here existed
    to repair chunk-boundary damage; with one call per paper there is nothing
    to repair. Questions with no text are dropped, but reported — silently
    discarding them is how a short-answer question disappears unnoticed.
    """
    dropped = []
    for section in sections:
        keep = []
        for q in section.questions:
            if not q.question or len(q.question.strip()) < 5:
                dropped.append(q.question_number)
                continue

            if q.question_type == QuestionType.MCQ.value and (
                    not q.options or len(q.options) < 2):
                q.question_type = QuestionType.OPEN.value

            if q.question_type == QuestionType.MATCHING.value and (
                    not q.column_a or not q.column_b or len(q.column_a) < 2):
                q.question_type = QuestionType.OPEN.value

            if q.marks is None or q.marks < 1:
                q.marks = 1

            keep.append(q)
        section.questions = sort_questions(keep)

    if dropped:
        print(f"    Dropped {len(dropped)} empty questions: {dropped[:10]}")
    return sections


def sort_questions(questions):
    def sort_key(q):
        parts = (q.question_number or "").split(".")
        return tuple(int(p) if p.isdigit() else 0 for p in parts)
    return sorted(questions, key=sort_key)


def inject_memo(sections, memo_answers):
    """
    Attach memo answers. Only genuinely unmatched numbers land in `unmatched` —
    the previous version appended even after a sub-part matched.
    """
    matched = 0
    unmatched = []

    for section in sections:
        for q in section.questions:
            q_num = (q.question_number or "").strip()

            if q_num in memo_answers:
                q.memo = memo_answers[q_num]
                matched += 1
                continue

            clean_num = re.sub(r'\.0+$', '', q_num)
            if clean_num != q_num and clean_num in memo_answers:
                q.memo = memo_answers[clean_num]
                matched += 1
                continue

            sub_hit = False
            for sub in (q.sub_parts or []):
                sub_num = sub.get("sub_number", "")
                if sub_num in memo_answers:
                    sub["memo"] = memo_answers[sub_num]
                    matched += 1
                    sub_hit = True

            if not sub_hit:
                unmatched.append(q_num)

    return sections, matched, unmatched


def count_types(sections):
    counts = {t.value: 0 for t in QuestionType}
    for section in sections:
        for q in section.questions:
            counts[q.question_type] = counts.get(q.question_type, 0) + 1
    return counts


# ═══════════════════════════════════════════════════════════════════════════════
# FILE CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

EXAM_KEYWORDS = [
    "exam", "paper", "question", "theory", "p1", "p2", "p3",
    "nov", "november", "may", "june", "feb", "february", "march", "mar",
    "aug", "august", "sep", "september", "oct", "october", "term",
    "trial", "nsc", "dbe", "cat", "mathematics", "maths", "physical",
    "sciences", "life sciences", "geography", "history", "accounting",
    "economics", "business", "afrikaans", "english", "isizulu", "sesotho",
    "control", "test",
]

MEMO_KEYWORDS = ["memo", "memorandum", "answers", "answer_key", "marking",
                 "marking guidelines"]

SUBJECT_PATTERNS = {
    r'\bmathematics\b': 'Mathematics',
    r'\bmaths\b': 'Mathematics',
    r'\bmath\s+lit\b': 'Mathematical Literacy',
    r'\btechnical\s+math\b': 'Technical Mathematics',
    r'\bphysical\s+sciences?\b': 'Physical Sciences',
    r'\blife\s+sciences?\b': 'Life Sciences',
    r'\bgeography\b': 'Geography',
    r'\bhistory\b': 'History',
    r'\baccounting\b': 'Accounting',
    r'\beconomics\b': 'Economics',
    r'\bbusiness\s+studies\b': 'Business Studies',
    r'\bcat\b': 'Computer Applications Technology',
    r'\bit\b': 'Information Technology',
    r'\bengineering\s+graphics\b': 'Engineering Graphics & Design',
    r'\benglish\b': 'English',
    r'\bafrikaans\b': 'Afrikaans',
    r'\bisixhosa\b': 'isiXhosa',
    r'\bisizulu\b': 'isiZulu',
    r'\btshivenda\b': 'TshiVenda',
    r'\bsesotho\b': 'Sesotho',
}

NOISE_WORDS = {"memo", "memorandum", "answers", "answer", "marking", "key",
               "theory", "exam", "paper", "nsc", "dbe", "grade", "gr", "cat",
               "caps", "p1", "p2", "p3", "question", "chunks", "nov", "november",
               "oct", "october", "jun", "june", "feb", "february", "mar", "march",
               "aug", "august", "sep", "september", "jan", "january", "jul", "july",
               "apr", "april", "dec", "december", "trial", "term", "final"}

MONTH_CANONICAL = {
    "jan": "january", "january": "january", "feb": "february", "february": "february",
    "mar": "march", "march": "march", "apr": "april", "april": "april", "may": "may",
    "jun": "june", "june": "june", "jul": "july", "july": "july",
    "aug": "august", "august": "august", "sep": "september", "september": "september",
    "oct": "october", "october": "october", "nov": "november", "november": "november",
    "dec": "december", "december": "december",
}


def classify_file(filename):
    lower = filename.lower()
    if any(kw in lower for kw in MEMO_KEYWORDS):
        return "memo"
    if any(kw in lower for kw in EXAM_KEYWORDS):
        return "exam"
    return "skip"


def detect_subject(filename):
    lower = filename.lower()
    for pattern, subject in SUBJECT_PATTERNS.items():
        if re.search(pattern, lower):
            return subject
    return "Unknown"


def detect_grade(filename):
    m = re.search(r'\b(?:gr|grade)\s*[_\-]?\s*(\d{1,2})\b', filename, re.I)
    return m.group(1) if m else "12"


def extract_keywords(filename):
    name = filename.lower().strip()
    name = re.sub(r'\s+\.', '.', name)
    name = re.sub(r"\.(json|pdf|docx?|odt|rtf)$", "", name)
    name = re.sub(r"_(exam|chunks)$", "", name)
    tokens = re.split(r"[^a-z0-9]+", name)
    keywords = set()
    for token in tokens:
        if not token:
            continue
        if token in MONTH_CANONICAL:
            keywords.add(MONTH_CANONICAL[token])
            continue
        if re.match(r"^\d{4}$", token) or re.match(r"^(term|t)\d$", token) \
                or re.match(r"^p\d$", token):
            keywords.add(token)
            continue
        if token in NOISE_WORDS:
            continue
        if len(token) >= 2:
            keywords.add(token)
    return keywords


def find_matching_exam(memo_filename, exam_files):
    memo_kw = extract_keywords(memo_filename)
    if not memo_kw:
        return None, set(), 0
    best_file, best_shared, best_score = None, set(), 0
    for ef in exam_files:
        shared = memo_kw & extract_keywords(ef)
        if not shared:
            continue
        score = len(shared) / len(memo_kw | extract_keywords(ef))
        if score > best_score:
            best_score, best_shared, best_file = score, shared, ef
    return (best_file, best_shared, best_score) if best_file else (None, set(), 0)


# ═══════════════════════════════════════════════════════════════════════════════
# TRACKER
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_key(filename):
    return re.sub(r'\s+\.', '.', filename.strip().lower())


def load_tracker():
    if not os.path.exists(TRACK_FILE):
        return {}
    try:
        with open(TRACK_FILE) as f:
            data = json.load(f)
    except Exception:
        return {}
    if isinstance(data, list):
        data = {n: {"exam_done": False, "memo_merged": False} for n in data}
    normalised = {}
    for raw_key, value in data.items():
        nk = normalize_key(raw_key)
        if nk not in normalised:
            normalised[nk] = {"exam_done": False, "memo_merged": False, "memo_source": None}
        if value.get("exam_done"):
            normalised[nk]["exam_done"] = True
        if value.get("memo_merged"):
            normalised[nk]["memo_merged"] = True
        if value.get("memo_source"):
            normalised[nk]["memo_source"] = value["memo_source"]
    if normalised != data:
        with open(TRACK_FILE, "w") as f:
            json.dump(normalised, f, indent=2)
    return normalised


def save_tracker(t):
    with open(TRACK_FILE, "w") as f:
        json.dump(t, f, indent=2)


def tracker_get(t, f):
    return t.get(normalize_key(f), {})


def tracker_set(t, f, k, v):
    nk = normalize_key(f)
    t.setdefault(nk, {})[k] = v


def output_path_for(f):
    stem = re.sub(r"\.(json|pdf|docx?|odt|rtf)$", "", normalize_key(f))
    return os.path.join(OUTPUT_FOLDER, stem + "_exam.json")


def exam_output_exists(f):
    return os.path.exists(output_path_for(f))


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def process():
    if not os.path.exists(INPUT_FOLDER):
        print(f"Folder '{INPUT_FOLDER}' not found.")
        return

    tracker = load_tracker()
    SKIP = {"metadata.json", "chunk_ids.json", "processed_files.json",
            "processed_exams.json"}

    all_files = [f for f in sorted(os.listdir(INPUT_FOLDER))
                 if Path(f).suffix.lower() in ALL_EXTS and f not in SKIP]

    exam_files, memo_files, skipped = [], [], []
    for f in all_files:
        kind = classify_file(f)
        (exam_files if kind == "exam" else
         memo_files if kind == "memo" else skipped).append(f)

    groq_status = (f"{GROQ_MODEL_EXTRACT} (TPM budget {GROQ_TPM_BUDGET})"
                   if GROQ_API_KEY else "not configured — every call goes to Gemini")
    print(f"\n{'='*64}")
    print(f"Input: {INPUT_FOLDER}")
    print(f"Primary:  Groq — {groq_status}")
    print(f"Rescue:   Gemini — {MODEL_NAME}")
    print(f"Files: {len(all_files)} | Exams: {len(exam_files)} | "
          f"Memos: {len(memo_files)} | Skipped: {len(skipped)}")
    for f in exam_files:
        e = tracker_get(tracker, f)
        s = ("+memo" if e.get("exam_done") and e.get("memo_merged")
             else "done" if e.get("exam_done") else "pending")
        print(f"  [{s}] [{detect_subject(f)}] {f}")
    for f in memo_files:
        s = "merged" if tracker_get(tracker, f).get("memo_merged") else "pending"
        print(f"  [{s}] {f}")
    print(f"{'='*64}\n")

    # ── STEP 1: extract exams ────────────────────────────────────────────────
    pending = [f for f in exam_files
               if not (tracker_get(tracker, f).get("exam_done") and exam_output_exists(f))]
    print(f"STEP 1: {len(pending)} exam(s) to extract\n")

    for idx, exam_file in enumerate(pending, 1):
        print(f"  [{idx}/{len(pending)}] {exam_file}")
        subject = detect_subject(exam_file)
        grade = detect_grade(exam_file)
        print(f"    Subject: {subject} | Grade: {grade}")

        kind, payload = load_source(exam_file)
        if not kind:
            print("    Could not read — skipping\n")
            continue

        size = len(payload)
        print(f"    Input: {kind} ({size:,} {'bytes' if kind == 'pdf' else 'chars'})")

        try:
            metadata, sections, stats = extract_exam(kind, payload, subject, grade)
        except Exception as e:
            print(f"    Extraction failed: {type(e).__name__}: {e}\n")
            continue

        sections = validate_exam_structure(sections)
        sections.sort(key=lambda s: s.section)

        total_q = sum(len(s.questions) for s in sections)
        if not total_q:
            print("    Nothing extracted\n")
            continue

        type_counts = count_types(sections)
        metadata_dict = metadata.to_dict()
        metadata_dict["detected_from_filename"] = subject

        out_data = {
            "source": exam_file,
            "metadata": metadata_dict,
            "total_questions": total_q,
            "questions_with_context": stats["questions_with_context"],
            "source_items": stats["source_items"],
            "type_breakdown": type_counts,
            "memo_merged": False,
            "memo_source": None,
            "sections": [s.to_dict() for s in sections],
        }

        out_path = output_path_for(exam_file)
        with open(out_path, "w") as f:
            json.dump(out_data, f, indent=2)

        print(f"    Saved: {out_path}")
        print(f"    {total_q}q across {len(sections)} sections | "
              f"{stats['source_items']} source items | "
              f"{stats['questions_with_context']}/{total_q} carry source material")
        print(f"    MCQ:{type_counts['mcq']} Match:{type_counts['matching']} "
              f"T/F:{type_counts['true_false']} Calc:{type_counts['calculation']} "
              f"Comp:{type_counts['comprehension']} Essay:{type_counts['essay']} "
              f"Open:{type_counts['open']}\n")

        tracker_set(tracker, exam_file, "exam_done", True)
        tracker_set(tracker, exam_file, "memo_merged", False)
        save_tracker(tracker)

    # ── STEP 2: merge memos ──────────────────────────────────────────────────
    pending_memos = [f for f in memo_files
                     if not tracker_get(tracker, f).get("memo_merged")]
    print(f"\nSTEP 2: {len(pending_memos)} memo(s) to merge\n")

    for idx, memo_file in enumerate(pending_memos, 1):
        print(f"  [{idx}/{len(pending_memos)}] {memo_file}")

        matched_exam, shared_kw, score = find_matching_exam(memo_file, exam_files)
        if not matched_exam:
            print("    No matching exam\n")
            continue

        exam_output = output_path_for(matched_exam)
        if not os.path.exists(exam_output):
            print(f"    Missing: {exam_output}\n")
            continue
        if not tracker_get(tracker, matched_exam).get("exam_done"):
            print("    Exam not yet extracted\n")
            continue

        print(f"    -> {matched_exam} ({score:.0%} match on {sorted(shared_kw)})")

        kind, payload = load_source(memo_file)
        if not kind:
            print("    Could not read memo\n")
            continue

        try:
            answers = extract_memo(kind, payload, detect_subject(memo_file))
        except Exception as e:
            print(f"    Memo extraction failed: {type(e).__name__}: {e}\n")
            continue

        if not answers:
            print("    No answers found\n")
            continue
        print(f"    {len(answers)} answers extracted")

        with open(exam_output) as f:
            exam_data = json.load(f)

        sections = []
        for sec_data in exam_data.get("sections", []):
            questions = [
                Question(**{k: v for k, v in q.items()
                            if k in Question.__dataclass_fields__})
                for q in sec_data.get("questions", [])
            ]
            sections.append(Section(
                section=sec_data.get("section", "A"),
                section_title=sec_data.get("section_title", ""),
                section_instructions=sec_data.get("section_instructions", ""),
                total_marks=sec_data.get("total_marks"),
                questions=questions,
            ))

        sections, matched_count, unmatched = inject_memo(sections, answers)

        exam_data["sections"] = [s.to_dict() for s in sections]
        exam_data["memo_merged"] = True
        exam_data["memo_source"] = memo_file
        exam_data["memo_answers_total"] = len(answers)
        exam_data["memo_matched"] = matched_count
        exam_data["memo_unmatched"] = unmatched

        with open(exam_output, "w") as f:
            json.dump(exam_data, f, indent=2)

        print(f"    Saved: {exam_output} | merged {matched_count}/{len(answers)}")
        if unmatched:
            print(f"    Unmatched: {unmatched[:15]}"
                  f"{'...' if len(unmatched) > 15 else ''}")
        print()

        tracker_set(tracker, memo_file, "memo_merged", True)
        tracker_set(tracker, memo_file, "memo_source", matched_exam)
        tracker_set(tracker, matched_exam, "memo_merged", True)
        tracker_set(tracker, matched_exam, "memo_source", memo_file)
        save_tracker(tracker)

    print("All done.")


if __name__ == "__main__":
    process()