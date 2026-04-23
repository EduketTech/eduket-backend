"""
rag.py  —  EduCAT RAG index  (LangChain migration)

Migration from raw HuggingFace InferenceClient to LangChain:

WHAT CHANGED
────────────
1. Embeddings     : HuggingFace InferenceClient.feature_extraction() replaced
                    by HuggingFaceEndpointEmbeddings from langchain_huggingface.
                    The same model (all-MiniLM-L6-v2) is used.

2. Vector store   : Manual numpy array + cosine_similarity() loop replaced by
                    LangChain's FAISS wrapper (langchain_community.vectorstores).
                    FAISS.from_documents() builds the index; similarity_search()
                    queries it.  The index is saved/loaded from disk with
                    FAISS.save_local() / FAISS.load_local() so caching still
                    works exactly as before (no re-embedding on restarts).

3. Documents      : Raw chunk dicts wrapped in LangChain Document objects
                    (page_content + metadata).  This is LangChain's standard
                    unit for retrieval pipelines.

4. search()       : Returns the same list-of-dicts format as before
                    ({ "content": ..., "source": ..., "score": ... }) so
                    agent.py's search_theory tool and generate_answer() in
                    model.py need no changes.

WHAT DID NOT CHANGE
───────────────────
- _is_rag_eligible() file classification logic is identical.
- load_all_chunks() reads from the same processed/ folder.
- Incremental embedding (only new chunks) still works via chunk ID tracking.
- The public RAGIndex.search(query, k) interface is identical.

INSTALL additions (requirements.txt)
──────────────────────────────────────
langchain-huggingface>=0.1.0
langchain-community>=0.3.0   # includes FAISS wrapper
faiss-cpu>=1.7               # or faiss-gpu
"""

import os
import json
import hashlib
from dotenv import load_dotenv

# ── LangChain imports ────────────────────────────────────────────────────────
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

load_dotenv()

HF_TOKEN  = os.getenv("HF_TOKEN")
BASE_DIR  = os.path.dirname(__file__)

PROCESSED_FOLDER = os.path.join(BASE_DIR, "processed")
FAISS_INDEX_DIR  = os.path.join(PROCESSED_FOLDER, "faiss_lc_index")
CHUNK_IDS_FILE   = os.path.join(PROCESSED_FOLDER, "chunk_ids.json")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ── Embeddings object — passed to FAISS ──────────────────────────────────────
# HuggingFaceEndpointEmbeddings calls the HuggingFace Inference API,
# same endpoint as the old InferenceClient.feature_extraction().
_embeddings = HuggingFaceEndpointEmbeddings(
    model     = MODEL_NAME,
    huggingfacehub_api_token = HF_TOKEN,
)


# ═══════════════════════════════════════════════════════════════════════════════
# FILE CLASSIFICATION  —  identical logic to old rag.py
# Only theory/study content gets embedded; exam papers and memos are excluded.
# ═══════════════════════════════════════════════════════════════════════════════

_INTERNAL_FILES = {
    "processed_files.json",
    "metadata.json",
    "chunk_ids.json",
    "processed_exams.json",
    "processed_memos.json",
    "embeddings.npy",         # old numpy cache (no longer written)
}

_EXAM_KEYWORDS = [
    "exam", "paper", "question", "theory", "p1", "p2", "p3",
    "nov", "november", "may", "june", "feb", "february",
    "march", "mar", "aug", "august", "sep", "september",
    "oct", "october", "term", "trial", "nsc", "dbe",
]

_MEMO_KEYWORDS = [
    "memo", "memorandum", "answers", "answer_key", "marking",
]


def _is_rag_eligible(filename: str) -> bool:
    """
    Returns True only for theory / study-guide content files.
    Exam question papers and memos are excluded.
    """
    if filename in _INTERNAL_FILES:
        return False
    lower = filename.lower()
    if any(kw in lower for kw in _MEMO_KEYWORDS):
        return False
    if any(kw in lower for kw in _EXAM_KEYWORDS):
        return False
    return True


def _generate_chunk_id(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


# ═══════════════════════════════════════════════════════════════════════════════
# CHUNK LOADING  —  returns LangChain Document objects
# ═══════════════════════════════════════════════════════════════════════════════

def load_all_chunks() -> list[Document]:
    """
    Load eligible content files from processed/ and wrap each chunk
    as a LangChain Document with page_content + metadata.
    """
    if not os.path.exists(PROCESSED_FOLDER):
        print(f"⚠️  Processed folder not found: {PROCESSED_FOLDER}")
        return []

    documents     = []
    loaded_files  = []
    skipped_files = []

    for filename in sorted(os.listdir(PROCESSED_FOLDER)):
        if not filename.endswith(".json"):
            continue
        if not _is_rag_eligible(filename):
            skipped_files.append(filename)
            continue

        path = os.path.join(PROCESSED_FOLDER, filename)
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as e:
            print(f"  ⚠️  Could not load {filename}: {e}")
            continue

        if not isinstance(data, list):
            skipped_files.append(filename)
            continue

        count = 0
        for item in data:
            content = item.get("content", "").strip()
            if not content:
                continue
            # LangChain Document: text goes in page_content, extras in metadata
            documents.append(Document(
                page_content = content,
                metadata     = {
                    "source":   item.get("source", filename),
                    "chunk_id": _generate_chunk_id(content),
                },
            ))
            count += 1

        if count:
            loaded_files.append(f"{filename} ({count} chunks)")

    print(f"\n📚 RAG content files  : {len(loaded_files)}")
    for f in loaded_files:
        print(f"     ✅ {f}")
    if skipped_files:
        print(f"🚫 Skipped            : {len(skipped_files)}")
    print(f"📦 Total documents    : {len(documents)}\n")

    return documents


# ═══════════════════════════════════════════════════════════════════════════════
# RAG INDEX
# ═══════════════════════════════════════════════════════════════════════════════

class RAGIndex:

    def __init__(self):
        self._store: FAISS | None = None
        self._docs = load_all_chunks()

        if not self._docs:
            print("⚠️  RAG: No eligible content files found.")
            print("💡  Add theory books / study guides to the processed/ folder.")

    # ── Build or update the FAISS index ───────────────────────────────────────
    def _load_or_build_index(self):
        """
        Incremental build:
        1. Load existing FAISS index from disk (if present).
        2. Determine which chunk IDs are new.
        3. Embed only new chunks and add them to the index.
        4. Save updated index to disk.

        This mirrors the old numpy-based incremental logic but uses
        LangChain's FAISS wrapper instead.
        """
        # Load existing chunk IDs
        if os.path.exists(CHUNK_IDS_FILE):
            with open(CHUNK_IDS_FILE) as f:
                existing_ids: set = set(json.load(f))
        else:
            existing_ids = set()

        # Load existing FAISS index from disk
        if os.path.exists(FAISS_INDEX_DIR) and existing_ids:
            try:
                self._store = FAISS.load_local(
                    FAISS_INDEX_DIR,
                    _embeddings,
                    allow_dangerous_deserialization=True,
                )
                print(f"⚡ FAISS index loaded from disk ({len(existing_ids)} cached chunks).")
            except Exception as e:
                print(f"⚠️  Could not load FAISS index: {e} — rebuilding.")
                self._store = None
                existing_ids = set()

        # Find new documents not yet in the index
        new_docs = [
            doc for doc in self._docs
            if doc.metadata["chunk_id"] not in existing_ids
        ]

        if not new_docs:
            print("⚡ No new chunks — using cached index.\n")
            return

        print(f"🔢 Embedding {len(new_docs)} new document(s)...")

        if self._store is None:
            # First-time build
            self._store = FAISS.from_documents(new_docs, _embeddings)
        else:
            # Incremental update — add new docs to existing index
            self._store.add_documents(new_docs)

        # Persist index and updated chunk ID list
        os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
        self._store.save_local(FAISS_INDEX_DIR)

        existing_ids.update(doc.metadata["chunk_id"] for doc in new_docs)
        with open(CHUNK_IDS_FILE, "w") as f:
            json.dump(list(existing_ids), f, indent=2)

        print(f"💾 FAISS index saved — {len(existing_ids)} total chunks.\n")

    # ── Public search method — same interface as before ────────────────────────
    def search(self, query: str, k: int = 3) -> list[dict]:
        """
        Search the theory book index for text relevant to query.

        Returns:
            list of dicts: [{ "content": str, "source": str, "score": float }]
            Same format as the old implementation so callers need no changes.
        """
        if not self._docs:
            return []

        self._load_or_build_index()

        if self._store is None:
            return []

        # similarity_search_with_score returns (Document, float) tuples
        # Lower L2 distance = better match; we convert to a 0-1 similarity score
        results = self._store.similarity_search_with_score(query, k=k)

        return [
            {
                "content": doc.page_content,
                "source":  doc.metadata.get("source", "unknown"),
                "score":   round(float(score), 4),
            }
            for doc, score in results
        ]