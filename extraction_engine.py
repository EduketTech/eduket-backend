"""
extraction_engine.py
────────────────────────────────────────────────────────────────────────────────
Multi-modal exam extraction engine for Eduket OS.

Drop-in replacement for the extraction helpers in app.py. Import this file
and remove the equivalent functions from app.py (see INTEGRATION NOTE at
the bottom of this file).

WHAT THIS FIXES vs app.py
──────────────────────────
1. DOCX IMAGES  — DOCX files are ZIP containers. Images live at
   word/media/. This module extracts every embedded image, sends each one
   to Groq vision (llama-4-scout), and inserts a [DIAGRAM: <description>]
   placeholder at the correct paragraph position so the question parser
   "sees" diagram context.

2. OMML MATH    — Word stores equations as <m:oMath> XML elements. The
   para.text shortcut returns "" for equation blocks, silently dropping all
   formulas. This module walks the XML directly and converts OMML to ASCII
   math notation ([EQ: ...]) before handing off to the parser.

3. PDF IMAGES   — Rather than only falling back to full-page OCR when native
   text fails, this module does a hybrid pass: native PyMuPDF text + per-image
   Groq vision description on every page regardless of text density.

4. SUBJECT-AWARE PARSING — The Groq question-parsing prompt is dynamically
   extended with subject-specific instructions:
   • Accounting: ledger/T-account/financial statement structure
   • Mathematics / Sciences: equation and formula preservation
   • Languages: comprehension passage as parent_context
   • Business/Economics: case study as parent_context
   • CAT/IT: code snippet preservation
   All other subjects fall through to sensible CAPS/NSC defaults.

INTEGRATION
───────────
In app.py, replace:
    from [inline functions] import extract_text_from_docx, ...
with:
    from extraction_engine import (
        extract_text_from_file,
        parse_questions_universal,
    )

Also update the two calls in run_extraction_pipeline that omit subject:
    extract_text_from_file(exam_bytes, exam_fn)          # old
    extract_text_from_file(exam_bytes, exam_fn, subject) # new
    extract_text_from_file(memo_bytes, memo_fn)          # old
    extract_text_from_file(memo_bytes, memo_fn, subject) # new
"""

import io
import os
import re
import json
import base64
import zipfile
import tempfile
import logging
from pathlib import Path
from xml.etree import ElementTree as ET

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

# ─── Groq models ──────────────────────────────────────────────────────────────
_VISION_MODEL  = "meta-llama/llama-4-scout-17b-16e-instruct"
_PARSER_MODEL  = "llama-3.3-70b-versatile"
_IMAGE_MIN_BYTES = 2_000          # skip images smaller than this (bullets/borders)
_CHUNK_SIZE      = 10_000
_CHUNK_OVERLAP   = 800


# Clark-notation URI for r:embed — never varies regardless of prefix used
_R_EMBED = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"
# ODT namespaces (clark form)
_ODT_TEXT_P     = "{urn:oasis:names:tc:opendocument:xmlns:text:1.0}p"
_ODT_TEXT_H     = "{urn:oasis:names:tc:opendocument:xmlns:text:1.0}h"
_ODT_DRAW_IMAGE = "{urn:oasis:names:tc:opendocument:xmlns:drawing:1.0}image"
_ODT_XLINK_HREF = "{http://www.w3.org/1999/xlink}href"



# ══════════════════════════════════════════════════════════════════════════════
# 1.  SUBJECT CLASSIFICATION
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
    "arts":          {"visual arts", "dramatic arts", "music", "dance"},
}



def _find_rids_in_element(elem) -> list:
    """
    Reliably find all r:embed relationship IDs in an lxml element subtree.
    Uses clark notation so namespace prefix variations never matter.
    Replaces the fragile r'r:embed="(rId\\d+)"' regex on serialized XML.
    """
    return [
        node.get(_R_EMBED)
        for node in elem.iter()
        if node.get(_R_EMBED)
    ]


def _odt_walk_content(content_xml_bytes: bytes, image_descs: dict) -> list:
    """
    Walk ODT content.xml and return text lines with [DIAGRAM: ...]
    placeholders inserted at the paragraph they actually belong to,
    not appended at the end of the document.
    """
    lines = []
    try:
        root = ET.fromstring(content_xml_bytes)

        def _para_parts(elem):
            text = "".join(elem.itertext()).strip()
            diagrams = []
            for node in elem.iter():
                if node.tag == _ODT_DRAW_IMAGE:
                    href = node.get(_ODT_XLINK_HREF, "")
                    # Normalize: "./Pictures/img.png" → "Pictures/img.png"
                    for prefix in ("./", "../", "/"):
                        if href.startswith(prefix):
                            href = href[len(prefix):]
                    if href and not href.startswith("Pictures/"):
                        href = f"Pictures/{href}"
                    if href in image_descs:
                        diagrams.append(image_descs[href])
            return text, diagrams

        for elem in root.iter():
            if elem.tag in (_ODT_TEXT_P, _ODT_TEXT_H):
                text, diagrams = _para_parts(elem)
                # Diagrams come first — they usually precede the question text
                for desc in diagrams:
                    lines.append(f"[DIAGRAM: {desc}]")
                if text:
                    lines.append(text)
    except Exception as exc:
        logger.error("[ODT walk] %s", exc)
    return lines

def _subject_category(subject: str) -> str:
    s = subject.lower().strip()
    for cat, keywords in _SUBJECT_MAP.items():
        if any(k in s for k in keywords):
            return cat
    return "general"


# ══════════════════════════════════════════════════════════════════════════════
# 2.  GROQ VISION HELPER
# ══════════════════════════════════════════════════════════════════════════════

def _vision_describe(image_bytes: bytes, ext: str, subject: str,
                     client: Groq) -> str:
    """
    Sends a single image to Groq vision and returns a concise
    educational description suitable for insertion into exam text.
    """
    try:
        media = {
            "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "gif": "image/gif", "webp": "image/webp", "bmp": "image/bmp",
            "tiff": "image/tiff", "tif": "image/tiff",
        }.get(ext.lower(), "image/png")

        b64 = base64.b64encode(image_bytes).decode()
        resp = client.chat.completions.create(
            model=_VISION_MODEL,
            messages=[{"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:{media};base64,{b64}"}},
                {"type": "text", "text": (
                    f"This is a figure from a South African {subject} exam paper. "
                    "Describe it precisely for a student who cannot see it. Include: "
                    "all labels, axes with units, scale, measurements, and component names. "
                    "If a graph: state x-axis, y-axis, curve shape, key points. "
                    "If a circuit diagram: list all components and connections. "
                    "If a geometric figure: all side lengths, angles, labels. "
                    "If a biological diagram: all labelled structures. "
                    "If a map: region names, legend, key features. "
                    "If a table: describe columns, rows, key values. "
                    "If pure equations or formulas: transcribe them in ASCII math exactly. "
                    "Max 120 words. No preamble."
                )},
            ]}],
            max_tokens=350,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        logger.warning("[Vision] Failed: %s", exc)
        return "figure or diagram (vision description unavailable)"


# ══════════════════════════════════════════════════════════════════════════════
# 3.  OMML → ASCII MATH  (DOCX equations)
# ══════════════════════════════════════════════════════════════════════════════

_OMML = "http://schemas.openxmlformats.org/officeDocument/2006/math"
_W    = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _omml_node_to_ascii(elem) -> str:
    """
    Lightweight recursive OMML → ASCII conversion.
    Goal: produce something an LLM can reason about,
    not perfect LaTeX.
    """
    local = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag

    def children_text():
        return "".join(_omml_node_to_ascii(c) for c in elem)

    def child_text(local_name):
        for c in elem:
            if c.tag.split("}")[-1] == local_name:
                return _omml_node_to_ascii(c)
        return ""

    if local == "t":               # text run
        return elem.text or ""
    if local == "f":               # fraction
        return f"({child_text('num')}/{child_text('den')})"
    if local == "sSup":            # superscript  x^n
        return f"{child_text('e')}^({child_text('sup')})"
    if local == "sSub":            # subscript    x_n
        return f"{child_text('e')}_{{{child_text('sub')}}}"
    if local == "sSubSup":         # sub+superscript
        return f"{child_text('e')}_{{{child_text('sub')}}}^({child_text('sup')})"
    if local == "rad":             # radical / nth-root
        deg = child_text("deg").strip()
        return f"sqrt({child_text('e')})" if not deg else f"root_{deg}({child_text('e')})"
    if local == "nary":            # summation / integral / product
        # Try to extract the operator character
        chr_elem = elem.find(f".//{{{_OMML}}}chr")
        op = (chr_elem.get(f"{{{_OMML}}}val", "∫")
              if chr_elem is not None else "∫")
        sub = child_text("sub")
        sup = child_text("sup")
        body = child_text("e")
        return f"{op}_{{{sub}}}^({sup}) {body}"
    if local == "d":               # delimiter ()[]{}
        return f"({children_text()})"
    if local == "m":               # matrix
        rows = [child_text("mr") for c in elem
                if c.tag.split("}")[-1] == "mr"]
        return "[[" + "], [".join(rows) + "]]"
    if local == "func":            # function  sin, cos, lim …
        name = child_text("fName")
        arg  = child_text("e")
        return f"{name}({arg})"
    if local in ("r", "num", "den", "e", "sup", "sub", "deg",
                 "fName", "base", "lim", "mr", "oMathPara"):
        return children_text()
    if local == "oMath":
        return children_text()
    # Default: recurse
    return children_text()


def _para_text_with_math(para) -> str:
    """
    Extract text from a python-docx Paragraph including OMML equations.
    Returns a string with equations rendered as [EQ: ...].
    """
    parts = []
    for child in para._element:
        tag_local = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        ns_uri    = child.tag.split("}")[0].lstrip("{") if "}" in child.tag else ""

        if tag_local == "oMath" and ns_uri == _OMML:
            math_text = _omml_node_to_ascii(child).strip()
            if math_text:
                parts.append(f"[EQ: {math_text}]")

        elif tag_local == "r":      # regular run
            for node in child:
                if node.tag.split("}")[-1] == "t" and node.text:
                    parts.append(node.text)

        elif tag_local == "hyperlink":
            for run in child:
                if run.tag.split("}")[-1] == "r":
                    for node in run:
                        if node.tag.split("}")[-1] == "t" and node.text:
                            parts.append(node.text)

    return "".join(parts).strip()


# ══════════════════════════════════════════════════════════════════════════════
# 4.  DOCX EXTRACTION  (text + math + images)
# ══════════════════════════════════════════════════════════════════════════════

def _docx_extract_images(docx_bytes: bytes) -> dict:
    """
    Returns {rId: (image_bytes, extension)} for all images in the DOCX body.
    DOCX is a ZIP — images live at word/media/<name>.
    """
    images = {}
    try:
        with zipfile.ZipFile(io.BytesIO(docx_bytes)) as z:
            # Parse relationship file to map rId → image path
            try:
                rels_xml  = z.read("word/_rels/document.xml.rels")
                rels_root = ET.fromstring(rels_xml)
                ns = {"r": "http://schemas.openxmlformats.org/package/2006/relationships"}
                for rel in rels_root.findall("r:Relationship", ns):
                    if "image" not in rel.get("Type", "").lower():
                        continue
                    rid    = rel.get("Id", "")
                    target = rel.get("Target", "")
                    if not target.startswith("word/"):
                        target = f"word/{target}"
                    try:
                        img_bytes = z.read(target)
                        if len(img_bytes) < _IMAGE_MIN_BYTES:
                            continue          # skip decorative bullets

                        ext = Path(target).suffix.lstrip(".").lower() or "png"
                        if ext in ("wmf", "emf"):
                            continue  # vector formats, skip vision
                        images[rid] = (img_bytes, ext)
                    except KeyError:
                        pass
            except Exception as exc:
                logger.warning("[DOCX images] Rels parse failed: %s", exc)
    except Exception as exc:
        logger.warning("[DOCX images] ZIP failed: %s", exc)
    return images


def extract_text_from_docx(file_bytes: bytes, subject: str = "General") -> str:
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P

    try:
        doc    = Document(io.BytesIO(file_bytes))
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        cat    = _subject_category(subject)

        raw_images = _docx_extract_images(file_bytes)
        image_descs = {}
        if raw_images:
            logger.info("[DOCX] Describing %d image(s) …", len(raw_images))
        for rid, (img_bytes, ext) in raw_images.items():
            image_descs[rid] = _vision_describe(img_bytes, ext, subject, client)

        lines = []

        def _add_para(para: Paragraph):
            text = _para_text_with_math(para)
            if text:
                lines.append(text)
            # ── FIX 1: use clark-notation attribute lookup, not XML string regex ──
            for rid in _find_rids_in_element(para._element):
                if rid in image_descs:
                    lines.append(f"[DIAGRAM: {image_descs[rid]}]")

        def _add_table(table: Table):
            for i, row in enumerate(table.rows):
                # ── FIX 3: deduplicate merged cells by underlying _tc identity ──
                seen_tc: set = set()
                unique_cells = []
                for c in row.cells:
                    tc_id = id(c._tc)
                    if tc_id not in seen_tc:
                        seen_tc.add(tc_id)
                        unique_cells.append(c)

                cell_texts = [c.text.strip() for c in unique_cells]

                if cat == "accounting":
                    if any(cell_texts):
                        prefix = "HEADER:" if i == 0 else "ROW:"
                        lines.append(f"{prefix} " + " | ".join(cell_texts))
                else:
                    row_txt = " | ".join(t for t in cell_texts if t)
                    if row_txt:
                        lines.append(row_txt)

                # ── FIX 2: extract images from table cells ──────────────────────
                for c in unique_cells:
                    for rid in _find_rids_in_element(c._tc):
                        if rid in image_descs:
                            lines.append(f"[DIAGRAM: {image_descs[rid]}]")

        for child in doc.element.body.iterchildren():
            if isinstance(child, CT_P):
                _add_para(Paragraph(child, doc))
            elif isinstance(child, CT_Tbl):
                _add_table(Table(child, doc))

        result = "\n".join(lines)
        logger.info("[DOCX] %d chars, %d images", len(result), len(image_descs))
        return result

    except Exception as exc:
        logger.error("[DOCX] Enhanced extraction failed: %s", exc)
        return _docx_basic(file_bytes)


def _docx_basic(file_bytes: bytes) -> str:
    """Fallback plain-text DOCX extractor (mirrors original app.py)."""
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P
    try:
        doc   = Document(io.BytesIO(file_bytes))
        lines = []
        for child in doc.element.body.iterchildren():
            if isinstance(child, CT_P):
                t = Paragraph(child, doc).text.strip()
                if t:
                    lines.append(t)
            elif isinstance(child, CT_Tbl):
                for row in Table(child, doc).rows:
                    cells = [c.text.strip() for c in row.cells]
                    r = " | ".join(c for c in cells if c)
                    if r:
                        lines.append(r)
        return "\n".join(lines)
    except Exception as exc:
        logger.error("[DOCX basic] %s", exc)
        return ""


# ══════════════════════════════════════════════════════════════════════════════
# 5.  ODT EXTRACTION  (text + images)
# ══════════════════════════════════════════════════════════════════════════════

def extract_text_from_odt(file_bytes: bytes, subject: str = "General") -> str:
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        # Extract and describe all images from the ZIP
        image_descs: dict = {}
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
            for name in z.namelist():
                if not name.startswith("Pictures/") or name.endswith("/"):
                    continue
                try:
                    img_bytes = z.read(name)
                    if len(img_bytes) < _IMAGE_MIN_BYTES:
                        continue
                    ext  = Path(name).suffix.lstrip(".") or "png"
                    desc = _vision_describe(img_bytes, ext, subject, client)
                    image_descs[name] = desc
                    logger.info("[ODT] %s → %s …", name, desc[:60])
                except Exception as exc:
                    logger.warning("[ODT] image %s failed: %s", name, exc)

            # ── FIX 4: parse content.xml for inline image positions ──────────
            try:
                content_xml = z.read("content.xml")
                lines = _odt_walk_content(content_xml, image_descs)
                if lines:
                    result = "\n".join(lines)
                    logger.info("[ODT] %d chars, %d images inline",
                                len(result), len(image_descs))
                    return result
            except Exception as exc:
                logger.warning("[ODT] content.xml walk failed: %s — "
                               "falling back to odfpy", exc)

        # Fallback: odfpy plain text (images already described, append at end)
        with tempfile.NamedTemporaryFile(suffix=".odt", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp.flush()
            odt_doc    = load_odt(tmp.name)
            paragraphs = odt_doc.getElementsByType(odf_text.P)
            lines      = [
                teletype.extractText(p) for p in paragraphs
                if teletype.extractText(p).strip()
            ]

        if image_descs:
            lines.append("\n─── FIGURES / DIAGRAMS ───")
            for i, desc in enumerate(image_descs.values(), 1):
                lines.append(f"[DIAGRAM {i}: {desc}]")

        result = "\n".join(lines)
        logger.info("[ODT] fallback: %d chars", len(result))
        return result

    except Exception as exc:
        logger.error("[ODT] All methods failed: %s", exc)
        return ""


# ══════════════════════════════════════════════════════════════════════════════
# 6.  PDF EXTRACTION  (hybrid: native text + per-image vision)
# ══════════════════════════════════════════════════════════════════════════════

def extract_text_from_pdf(file_bytes: bytes,
                          subject: str = "General") -> str:
    """
    Hybrid PDF extractor:
    Stage 1 — PyMuPDF native text per page.
    Stage 2 — Groq vision for every embedded image on that page,
               regardless of whether the page also has native text.
    Stage 3 — Pure Groq vision OCR for scanned / image-only PDFs.
    """
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    try:
        doc             = fitz.open(stream=file_bytes, filetype="pdf")
        total_text      = ""
        page_blocks: list[str] = []

        for i, page in enumerate(doc):
            page_text  = page.get_text().strip()
            total_text += page_text
            page_lines = []
            if page_text:
                page_lines.append(page_text)

            # Describe every embedded image on this page
            for img_idx, img_info in enumerate(page.get_images(full=True)):
                try:
                    xref = img_info[0]
                    pix  = fitz.Pixmap(doc, xref)
                    if pix.n > 4:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    img_bytes = pix.tobytes("png")
                    if len(img_bytes) < _IMAGE_MIN_BYTES:
                        continue
                    desc = _vision_describe(img_bytes, "png", subject, client)
                    page_lines.append(f"[DIAGRAM: {desc}]")
                    logger.info("[PDF] p%d img%d → %s …",
                                i + 1, img_idx + 1, desc[:60])
                except Exception as exc:
                    logger.warning("[PDF] p%d img%d failed: %s",
                                   i + 1, img_idx + 1, exc)

            page_blocks.append("\n".join(page_lines))

        doc.close()

        if total_text.strip():
            result = "\n\n".join(page_blocks)
            logger.info("[PDF] Hybrid: %d chars", len(result))
            return result

        # No native text — full OCR
        logger.info("[PDF] No native text — full vision OCR")
        return _pdf_vision_ocr(file_bytes, subject, client)

    except Exception as exc:
        logger.error("[PDF] Enhanced failed: %s", exc)
        return _pdf_vision_ocr(
            file_bytes, subject, Groq(api_key=os.getenv("GROQ_API_KEY"))
        )


def _pdf_vision_ocr(pdf_bytes: bytes, subject: str, client: Groq) -> str:
    """Full-page Groq vision OCR for scanned PDFs."""
    all_text = ""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for i, page in enumerate(doc):
            try:
                pix  = page.get_pixmap(dpi=200)
                img  = base64.b64encode(pix.tobytes("png")).decode()
                resp = client.chat.completions.create(
                    model=_VISION_MODEL,
                    messages=[{"role": "user", "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{img}"}},
                        {"type": "text", "text": (
                            f"South African CAPS/NSC {subject} exam page. "
                            "Extract ALL text exactly. Preserve question numbers, "
                            "marks in brackets like (2), MCQ options A B C D. "
                            "For any diagram/graph/figure write [DIAGRAM: <description>]. "
                            "For any equation write [EQ: <ASCII math>]. "
                            "Plain text only, no markdown."
                        )},
                    ]}],
                    max_tokens=2500,
                )
                page_text = resp.choices[0].message.content.strip()
                all_text += page_text + "\n\n"
                logger.info("[OCR] Page %d done", i + 1)
            except Exception as exc:
                logger.error("[OCR] Page %d failed: %s", i + 1, exc)
        doc.close()
    except Exception as exc:
        logger.error("[OCR] Fatal: %s", exc)
    return all_text


# ══════════════════════════════════════════════════════════════════════════════
# 7.  UNIFIED FILE ROUTER
# ══════════════════════════════════════════════════════════════════════════════

def extract_text_from_file(file_bytes: bytes,
                           filename: str,
                           subject: str = "General") -> str:
    """
    Route to the correct extractor by file extension.
    Subject is passed through to enable subject-aware extraction
    (image descriptions, table formatting, math extraction).
    """
    lower = filename.lower()
    if lower.endswith(".docx"):
        return extract_text_from_docx(file_bytes, subject)
    if lower.endswith(".odt"):
        return extract_text_from_odt(file_bytes, subject)
    if lower.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes, subject)
    if lower.endswith(".doc"):
        try:
            return mammoth.extract_raw_text(io.BytesIO(file_bytes)).value
        except Exception as exc:
            logger.error("[DOC] mammoth failed: %s", exc)
    logger.warning("[extract_text_from_file] Unrecognised extension: %s",
                   filename)
    return ""


# ══════════════════════════════════════════════════════════════════════════════
# 8.  SUBJECT-AWARE PARSING HINTS
# ══════════════════════════════════════════════════════════════════════════════

_PARSING_HINTS: dict[str, str] = {
    "accounting": """
ACCOUNTING-SPECIFIC RULES:
- Financial statements (Income Statement, Balance Sheet, Cash Flow, Notes)
  are ONE question even if they span many lines. Capture the entire table
  structure inside the "question" field using HEADER:/ROW: lines.
- T-accounts / Ledger accounts: capture both debit AND credit sides.
- Trial balance: capture every account name and its debit/credit amount.
- Journal entries: include date, account name, debit, credit columns.
- Reconciliation statements: preserve every line item.
- Mark allocation per section often appears as (x) at end of statement.
- Sub-questions like "1.1 Prepare the Income Statement (20)" should each be
  a separate item with parent_question = "QUESTION 1".
- type="accounting_statement" for financial statement preparation questions.
""",
    "mathematics": """
MATHEMATICS-SPECIFIC RULES:
- Preserve ALL equations and formulas using [EQ: ...] notation already in
  the text — do NOT strip or simplify them.
- Geometric diagrams described as [DIAGRAM: ...] must appear in the question
  field of the question that references them.
- Multi-part questions (1.1.1, 1.1.2) must maintain full hierarchy with
  parent_question and parent_context fields populated.
- type="proof" for "Show that…" / "Prove that…" questions.
- type="calculation" for numerical answer questions.
- Data tables in data handling questions must be preserved in full.
- Marks usually appear at end of line as (3) — parse carefully.
""",
    "sciences": """
PHYSICAL SCIENCES-SPECIFIC RULES:
- Preserve all SI units exactly (m·s⁻², N, J, Pa, mol·dm⁻³, etc.).
- [DIAGRAM: ...] placeholders for circuit diagrams, force diagrams, graphs,
  wave diagrams must be included in the question field they belong to.
- Equations such as [EQ: v² = u² + 2as] must be preserved in full.
- Scenario/context paragraphs (the "given information" block before sub-
  questions) are the parent_context for all sub-questions under them.
- type="practical" for investigation / experiment design questions.
- type="calculation" for numerical questions requiring formula application.
""",
    "life_sciences": """
LIFE SCIENCES-SPECIFIC RULES:
- [DIAGRAM: ...] placeholders (cells, organs, food webs) must be included in
  the question field that asks about them.
- "Label the parts" / "Identify structure X" questions: the diagram
  description is the parent_context.
- Data table / graph interpretation: preserve all column headers and values.
- type="practical" for investigation / experiment questions.
""",
    "geography": """
GEOGRAPHY-SPECIFIC RULES:
- [DIAGRAM: ...] placeholders for maps, climate graphs, cross-sections must
  appear in the question field of the question referencing them.
- Case study / stimulus text (newspaper extract, data set) is the
  parent_context for all sub-questions beneath it.
- Preserve all statistical values, percentages, and place names exactly.
""",
    "business": """
BUSINESS STUDIES / ECONOMICS-SPECIFIC RULES:
- Case study or scenario text is the parent_context for every sub-question
  that references it — do NOT repeat it in each question field.
- type="essay" for discussion / critically analyse / evaluate questions
  (usually 20–40 marks).
- type="short_answer" for definition / identify / list questions (2–4 marks).
- Quotations from the scenario that sub-questions reference must appear in
  parent_context.
""",
    "language": """
LANGUAGE (ENGLISH / AFRIKAANS / IsiZulu / etc.) RULES:
- Reading comprehension passage is the parent_context for ALL questions
  that follow it — set parent_context to the full passage text (or a clear
  reference like "See Reading Passage A above").
- type="mcq" for vocabulary / grammar multiple-choice questions.
- type="essay" for creative writing / formal essay / summary tasks.
- type="short_answer" for contextual comprehension questions.
- Quoted lines from poetry / prose that a question asks about must appear
  in the question field itself.
""",
    "cat_it": """
CAT / INFORMATION TECHNOLOGY RULES:
- Code snippets MUST be preserved exactly, including indentation and spacing.
  Wrap in triple backticks inside the question field: ```pascal ... ```.
- Scenario text (the problem description before sub-questions) is the
  parent_context.
- Spreadsheet cell references like B2:B10 or $A$1 must be preserved exactly.
- type="practical" for spreadsheet / word processing / database tasks.
- type="calculation" for algorithm / pseudocode / trace table questions.
""",
    "history": """
HISTORY-SPECIFIC RULES:
- Source / extract text (Document A, B, C) is the parent_context for any
  question that refers to it.
- type="essay" for extended writing / "to what extent" questions.
- type="short_answer" for source analysis questions (2–4 marks).
- Preserve all dates, names, and quoted text exactly.
""",
    "general": "",
}


# ══════════════════════════════════════════════════════════════════════════════
# 9.  QUESTION PARSER
# ══════════════════════════════════════════════════════════════════════════════

def _normalise_qnum(qn: str) -> str:
    s = str(qn).lower().strip()
    s = re.sub(r"^(question|q|ques|no|nr)[\s.\-]*", "", s)
    s = re.sub(r"[^a-z0-9]", "", s)
    return s


def parse_questions_universal(exam_text: str,
                               subject: str,
                               grade: str) -> list:
    """
    Subject-aware question parser. Chunks the exam text and calls Groq
    for each chunk, then deduplicates by question number.

    Compared to the original app.py version, this adds:
    - Subject-specific parsing hints in the prompt
    - Instructions to preserve [DIAGRAM: ...] and [EQ: ...] in question fields
    - Instructions to populate parent_context for scenario-based subjects
    """
    client   = Groq(api_key=os.getenv("GROQ_API_KEY"))
    cat      = _subject_category(subject)
    hints    = _PARSING_HINTS.get(cat, "")
    all_qs:  list[dict] = []
    seen:    set[str]   = set()

    chunks: list[str] = []
    start = 0
    while start < len(exam_text):
        chunks.append(exam_text[start:start + _CHUNK_SIZE])
        start += _CHUNK_SIZE - _CHUNK_OVERLAP

    for idx, chunk in enumerate(chunks):
        logger.info("[Parser] Chunk %d/%d — %s", idx + 1, len(chunks), subject)

        prompt = f"""You are an expert at parsing South African CAPS/NSC/IEB exam papers for {subject} Grade {grade}.

Extract EVERY question from the text below into a JSON array.

GENERAL RULES:
- MCQ: split options into A/B/C/D dict, type="mcq"
- True/False: type="true_false"
- Matching: type="matching", column_a=[], column_b=[]
- Calculation: type="calculation"
- Essay / discuss / analyse: type="essay"
- Short answer / define / identify: type="short_answer"
- Default: type="open"
- Marks: integer from brackets like (2) or [2], default 1
- question_number: exactly as printed, e.g. "1.1", "2.3.1", "QUESTION 3"
- parent_question: the top-level question heading, e.g. "QUESTION 1"
- parent_context: scenario / passage / source text shared by multiple sub-questions (null if none)
- If a [DIAGRAM: ...] placeholder appears near a question, include it INSIDE the "question" field
- If an [EQ: ...] placeholder appears near a question, include it INSIDE the "question" field
- Preserve table structures by including them verbatim in the "question" field

SUBJECT-SPECIFIC RULES:
{hints}

Return ONLY a valid JSON array. No markdown, no explanation.

Each item:
{{
  "question_number": "1.1",
  "parent_question": "QUESTION 1",
  "parent_context": null,
  "section": "A",
  "question": "Full question text including any [DIAGRAM: ...] or [EQ: ...] content",
  "type": "short_answer",
  "marks": 2,
  "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
  "column_a": null,
  "column_b": null,
  "memo": null
}}

Subject: {subject} | Grade: {grade}

EXAM TEXT:
{chunk}"""

        try:
            resp = client.chat.completions.create(
                model=_PARSER_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=8000,
            )
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
            raw = re.sub(r"\s*```\s*$",        "", raw, flags=re.MULTILINE)
            m   = re.search(r"\[.*\]", raw, re.DOTALL)
            if m:
                parsed = json.loads(m.group())
                if isinstance(parsed, list):
                    for q in parsed:
                        qn  = _normalise_qnum(q.get("question_number", ""))
                        key = qn or q.get("question", "")[:60]
                        if key not in seen:
                            seen.add(key)
                            all_qs.append(q)
        except Exception as exc:
            logger.error("[Parser] Chunk %d failed: %s", idx + 1, exc)

    logger.info("[Parser] %d questions extracted for %s", len(all_qs), subject)
    return all_qs