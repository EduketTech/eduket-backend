"""
chunk_pdf.py  —  EduCAT PDF chunking pipeline  (LangChain migration)

WHAT CHANGED
────────────
No LangChain migration needed for this file.
chunk_pdf.py uses only stdlib (os, json, hashlib) and pdfplumber.
It produces the JSON chunk files that feed into:

  process_exams.py  →  LangChain-migrated exam extraction pipeline
  rag.py            →  LangChain FAISS RAG index builder

The output format (list of dicts with source / chunk_index /
total_chunks / content keys) is consumed unchanged by both downstream
files after the LangChain migration.

Minor housekeeping applied:
1. hashlib.md5(usedforsecurity=False) — suppresses Python 3.9+
   security warning when md5 is used for non-cryptographic hashing.
2. Type hints added to all function signatures for readability.
3. Module-level docstring added for consistency with the migrated files.

WHAT DID NOT CHANGE
───────────────────
- BLOCKLIST logic
- DATA_FOLDER / PROCESSED_FOLDER / TRACKER_FILE paths
- Tracker load/save/migration
- extract_text_from_pdf()
- chunk_text() — chunk size, overlap, metadata fields
- merge_chunks()
- process_files() — all logic, output format, summary printing

PIPELINE POSITION
─────────────────
  1. chunk_pdf.py         ← YOU ARE HERE
     Reads PDFs from data/
     Writes JSON chunk files to processed/

  2. process_exams.py
     Reads chunk JSON from processed/
     Writes structured exam JSON to exams/
     (uses langchain_groq ChatGroq for LLM extraction)

  3. rag.py
     Reads theory-book chunk JSON from processed/
     Builds / updates FAISS vector index
     (uses langchain_huggingface + langchain_community FAISS)

  4. agent.py / app.py
     Serves the Flask API using the agent loop and exam JSON
"""

import pdfplumber
import os
import json
import hashlib

print("🔥 SCRIPT STARTED")
print("📁 Current directory:", os.getcwd())


# =========================
# 🚫 BLOCKLIST
# PDFs here will never be processed — no chunks created.
# Add filenames that should always be skipped (e.g. source books
# that have already been manually processed, or files not yet ready).
# =========================
BLOCKLIST: set[str] = {
    "Gr12_CAT_Theory Book.pdf",
    "May-memo_ 2025.pdf",
    # Add more filenames here as needed:
    # "some_other_file.pdf",
}


# =========================
# 📂 PATHS
# =========================
DATA_FOLDER      = "data/"
PROCESSED_FOLDER = "processed/"
TRACKER_FILE     = os.path.join(PROCESSED_FOLDER, "processed_files.json")

os.makedirs(PROCESSED_FOLDER, exist_ok=True)


# =========================
# 🔐 FILE HASH  (change detection)
# Uses MD5 purely for file-change detection — not for security.
# usedforsecurity=False suppresses the Python 3.9+ DeprecationWarning.
# =========================
def get_file_hash(filepath: str) -> str:
    """Return MD5 hex digest of a file for change detection."""
    h = hashlib.md5(usedforsecurity=False)
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# =========================
# 🧠 LOAD / SAVE TRACKER
# Tracks which PDFs have been processed and their content hash
# so the pipeline can detect new or modified files on re-runs.
#
# Tracker format:
# {
#   "Nov_Theory_2024.pdf": {
#     "hash":        "abc123...",
#     "chunks_file": "processed/Nov_Theory_2024.json",
#     "chunk_count": 10
#   },
#   ...
# }
# =========================
def load_tracker() -> dict:
    """
    Load the processing tracker from disk.
    Migrates the old list format to the current dict format automatically.
    """
    if not os.path.exists(TRACKER_FILE):
        return {}
    try:
        with open(TRACKER_FILE) as f:
            data = json.load(f)
            # Migrate old list format → new dict format
            if isinstance(data, list):
                print("⚙️  Migrating old tracker format...")
                return {
                    name: {"hash": None, "chunks_file": None, "chunk_count": None}
                    for name in data
                }
            return data
    except (json.JSONDecodeError, Exception):
        print("⚠️  Could not read tracker — starting fresh.")
        return {}


def save_tracker(tracker: dict) -> None:
    with open(TRACKER_FILE, "w") as f:
        json.dump(tracker, f, indent=2)


# =========================
# 🧾 EXTRACT TEXT FROM PDF
# Uses pdfplumber to pull plain text from every page.
# Images, tables, and embedded objects are not extracted —
# only selectable text.
# =========================
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file using pdfplumber."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


# =========================
# ✂️ CHUNK TEXT
# Splits extracted text into fixed-size word chunks.
#
# Each chunk carries full metadata so that downstream consumers
# (process_exams.py, rag.py) have everything they need:
#
#   source       : original PDF filename — used by process_exams.py
#                  to classify the file (exam / memo / theory) and by
#                  rag.py to record provenance in vector metadata.
#   chunk_index  : position in the document — used by process_exams.py
#                  to stitch chunks back into order before sliding-window
#                  extraction.
#   total_chunks : total chunk count — displayed in progress logs.
#   content      : the text of this chunk — embedded by rag.py and sent
#                  to Groq by process_exams.py.
#
# chunk_size=300 words produces chunks of ~1500-2000 characters,
# well within Groq's context window and the 6000-char sliding window
# used in process_exams.py.
# =========================
def chunk_text(text: str, source_file: str, chunk_size: int = 300) -> list[dict]:
    """
    Split text into word-based chunks with source metadata attached.

    Args:
        text        : full extracted text from the PDF
        source_file : original PDF filename (no path)
        chunk_size  : words per chunk (default 300 ≈ 1500 chars)

    Returns:
        list of chunk dicts with source / chunk_index / total_chunks / content
    """
    words      = text.split()
    raw_chunks = [
        " ".join(words[i : i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]
    total = len(raw_chunks)

    return [
        {
            "source":       source_file,
            "chunk_index":  idx,
            "total_chunks": total,
            "content":      chunk,
        }
        for idx, chunk in enumerate(raw_chunks)
    ]


# =========================
# 🔀 MERGE CHUNKS
# When a PDF is re-processed after changes, we merge the new chunks
# into the existing output file rather than overwriting it blindly.
# This preserves chunks from other sources that may share the same
# output file (future: multi-source merging).
# =========================
def merge_chunks(existing_chunks: list, new_chunks: list) -> list:
    """
    Replace chunks from the same source with the new version.
    Chunks from OTHER sources in the same file are kept untouched.

    Args:
        existing_chunks : chunks already in the output JSON file
        new_chunks      : freshly extracted chunks from the updated PDF

    Returns:
        merged list with new_chunks replacing any old chunks for the same source
    """
    if not existing_chunks:
        return new_chunks

    source = new_chunks[0]["source"] if new_chunks else None
    kept   = [c for c in existing_chunks if c.get("source") != source]
    return kept + new_chunks


# =========================
# 🚀 PROCESS FILES
# Main pipeline: scan data/, detect new/changed PDFs,
# extract text, chunk, and save to processed/.
# =========================
def process_files() -> None:
    tracker = load_tracker()

    print("\n📌 Already tracked files:", list(tracker.keys()) or "none")

    if not os.path.exists(DATA_FOLDER):
        print("❌ data/ folder NOT found!")
        return

    pdf_files = [f for f in sorted(os.listdir(DATA_FOLDER)) if f.endswith(".pdf")]

    if not pdf_files:
        print("⚠️  No PDF files found in data/")
        return

    # ── Pre-flight classification ─────────────────────────
    blocked  = [f for f in pdf_files if f in BLOCKLIST]
    eligible = [f for f in pdf_files if f not in BLOCKLIST]

    # A file is "changed" if its hash differs from the tracked value
    new_files = [f for f in eligible if f not in tracker]
    changed_files = [
        f for f in eligible
        if f in tracker
        and tracker[f].get("hash") != get_file_hash(os.path.join(DATA_FOLDER, f))
    ]
    unchanged_files = [
        f for f in eligible
        if f in tracker
        and tracker[f].get("hash") == get_file_hash(os.path.join(DATA_FOLDER, f))
    ]

    print(f"\n📂 Total PDFs     : {len(pdf_files)}")
    print(f"🚫 Blocklisted    : {len(blocked)}")
    for b in blocked:
        print(f"     → {b}")
    print(f"⏭️  Unchanged      : {len(unchanged_files)}")
    print(f"🆕 New            : {len(new_files)}")
    print(f"✏️  Changed        : {len(changed_files)}")

    to_process = new_files + changed_files

    if not to_process:
        print("\n✅ All files are up to date. Nothing to do.")
        return

    print(f"\n🔄 Processing {len(to_process)} file(s)...\n")

    new_count     = 0
    updated_count = 0

    for file in to_process:
        is_update = file in tracker
        label     = "✏️  UPDATE" if is_update else "🆕 NEW"

        print(f"{label}: {file}")

        pdf_path = os.path.join(DATA_FOLDER, file)

        # 📥 Extract text from PDF
        text = extract_text_from_pdf(pdf_path)
        print(f"  📝 Extracted {len(text)} characters")

        if not text.strip():
            print(f"  ⚠️  No text extracted — skipping.\n")
            continue

        # ✂️ Chunk the extracted text
        new_chunks  = chunk_text(text, source_file=file)
        output_path = os.path.join(PROCESSED_FOLDER, file.replace(".pdf", ".json"))
        print(f"  🔹 {len(new_chunks)} chunk(s) created")

        # 🔀 Merge with existing chunks if this is an update
        if is_update and os.path.exists(output_path):
            try:
                with open(output_path) as f:
                    existing_chunks = json.load(f)
                merged = merge_chunks(existing_chunks, new_chunks)
                print(
                    f"  🔀 Merged: {len(existing_chunks)} old + "
                    f"{len(new_chunks)} new → {len(merged)} total chunks"
                )
            except Exception:
                merged = new_chunks
                print("  ⚠️  Could not read existing chunks — replacing.")
        else:
            merged = new_chunks

        # 💾 Save chunks to processed/<filename>.json
        with open(output_path, "w") as f:
            json.dump(merged, f, indent=2)
        print(f"  💾 Saved: {output_path}")

        # ✅ Record new hash in tracker
        tracker[file] = {
            "hash":        get_file_hash(pdf_path),
            "chunks_file": output_path,
            "chunk_count": len(merged),
        }
        save_tracker(tracker)

        if is_update:
            updated_count += 1
        else:
            new_count += 1

        print()

    # ── Final summary ─────────────────────────────────────
    print("=" * 45)
    print(f"🆕 New files processed    : {new_count}")
    print(f"✏️  Updated files          : {updated_count}")
    print(f"⏭️  Skipped (unchanged)    : {len(unchanged_files)}")
    print(f"🚫 Skipped (blocklisted)  : {len(blocked)}")
    print(f"📦 Total tracked          : {len(tracker)}")
    print("✅ DONE")


# =========================
# ▶️ RUN
# =========================
if __name__ == "__main__":
    process_files()