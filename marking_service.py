"""
marking_service.py — Eduket OS  Batch Marking Service  v2.0
═══════════════════════════════════════════════════════════════════════════════
Powers app.py's /api/marking/init-cache and /api/marking/evaluate routes —
a SEPARATE batch-marking flow from the student self-marking exam pipeline
(/submit, which uses extraction_engine.mark_answer() and is unrelated to
this file).

WHAT CHANGED FROM v1
═════════════════════
1. SIGNATURE MISMATCH FIXED. app.py's routes called
   create_rubric_cache(rubric_text=..., subject_name=..., ttl_minutes=...)
   and mark_student_submission(cache_name=..., student_id=..., student_answers=...)
   — but this file's actual functions took create_rubric_cache(ttl_minutes)
   and mark_student_submission(student_id, student_answers), with no
   rubric_text, subject_name or cache_name parameters at all. Every call to
   either route raised TypeError: unexpected keyword argument, so both
   routes have been failing on every invocation. Both functions now accept
   exactly what app.py already sends.

2. THE REAL UPLOADED RUBRIC IS NOW USED. v1 cached a hardcoded sample CAT
   Grade 12 rubric (EXAM_MEMO_CONTEXT) regardless of what a teacher
   actually uploaded — rubric_text had nowhere to go even before the
   signature fix, since the parameter didn't exist. Every submission, for
   every subject, was being graded against the same fixed sample. rubric_text
   is now what actually gets cached and sent to the model.

3. PER-MEMO CACHES, NOT ONE GLOBAL CACHE. v1 kept a single module-level
   _gemini_cache singleton, correct only if the whole process ever marks
   one memo at a time. Caches are now stored per memo in _RUBRIC_STORE,
   keyed by the stable handle app.py hands back to the caller as
   `cache.name` (and stores in its own ACTIVE_RUBRIC_CACHES[memo_id]).

4. EXPIRED-CACHE RECOVERY. A Gemini cached-content resource expires after
   its TTL. If a student submits after that point, the old code would just
   fail. mark_student_submission() now detects an expired/missing
   underlying cache and transparently recreates it from the stored
   rubric_text — under the SAME stable handle, so app.py's
   ACTIVE_RUBRIC_CACHES[memo_id] entry (created once, at init-cache time)
   keeps working without app.py needing to know the cache was ever expired
   or recreated.

5. GROQ_MODEL_MARK DEFAULT FIXED. Was "groq/compound-mini" — a smaller
   variant of Groq's agentic Compound system, which can autonomously invoke
   tools (web search, code execution) rather than just completing the
   prompt it's given. That's the same root cause already fixed in the exam
   extraction pipeline (see extract_exams_v2.py / extraction_engine.py) and
   in the /agent-chat route in app.py — an agentic model is the wrong
   choice for marking a student's answer strictly against a memo, since
   there's no guarantee it stays grounded in the memo text it was given
   rather than reasoning outside it. Now defaults to "openai/gpt-oss-120b",
   matching extraction_engine.GROQ_MODEL_MARK.

Requires:  pip install google-genai groq
Env:       GEMINI_API_KEY, GEMINI_MODEL_MARK, GEMINI_CACHE_MODEL,
           GROQ_API_KEY, GROQ_MODEL_MARK, GROQ_TPM_BUDGET,
           GROQ_COOLDOWN_SECONDS
"""
import os
import re
import time
import json
from google import genai
from google.genai import types
from groq import Groq, RateLimitError as GroqRateLimitError, \
    APIStatusError as GroqAPIStatusError, APIError as GroqAPIError

# ══════════════════════════════════════════════════════════════════════════════
# PROVIDER SETUP — Groq primary, Gemini paid rescue
# ══════════════════════════════════════════════════════════════════════════════
# Same routing shape as extraction_engine.py: Groq first, Gemini only when
# Groq is unconfigured, in a post-429 cooldown, over its rolling TPM budget,
# or actually errors out. This still duplicates that module's TPM/cooldown
# logic rather than importing it — the same DUPLICATION WARNING that applies
# throughout this codebase applies here too; move this into a shared
# ai_routing.py that extraction_engine.py, app.py and this file all import,
# before the three drift against each other the way extraction's routing
# already did once.

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_MARK = os.getenv("GEMINI_MODEL_MARK", "gemini-3.6-flash")
GEMINI_CACHE_MODEL = os.getenv("GEMINI_CACHE_MODEL", "gemini-3.5-flash-lite")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# FIXED: was "groq/compound-mini" (an agentic, tool-using system — see the
# module docstring for why that's the wrong choice for marking). Now
# matches extraction_engine.GROQ_MODEL_MARK's default.
GROQ_MODEL_MARK = os.getenv("GROQ_MODEL_MARK", "openai/gpt-oss-120b")
GROQ_TPM_BUDGET = int(os.getenv("GROQ_TPM_BUDGET", "50000"))
GROQ_COOLDOWN_SECONDS = int(os.getenv("GROQ_COOLDOWN_SECONDS", "90"))

client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
_groq_client: Groq | None = None


def groq_client() -> Groq | None:
    global _groq_client
    if not GROQ_API_KEY:
        return None
    if _groq_client is None:
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


_groq_usage_log: list[tuple[float, int]] = []
_groq_cooldown_until: float = 0.0


def _estimate_tokens(text: str) -> int:
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


def _parse_json_response(raw: str) -> dict:
    """Groq's json_object mode is looser than Gemini's response_schema — same
    codefence-strip defensive parse used elsewhere in this codebase."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
    return json.loads(text)


# ══════════════════════════════════════════════════════════════════════════════
# RUBRIC STORE
# ══════════════════════════════════════════════════════════════════════════════
# Keyed by a STABLE handle — the Gemini cache.name at the moment
# create_rubric_cache() first creates it. app.py stores this same handle in
# its own ACTIVE_RUBRIC_CACHES[memo_id] dict and passes it back as
# cache_name on every /evaluate call for that memo.
#
# The stable handle deliberately never changes, even if the underlying
# Gemini cache resource expires and gets recreated — only the
# "gemini_cache_name" field inside the stored entry is updated when that
# happens (see _get_live_gemini_cache_name below). This is what lets
# app.py's ACTIVE_RUBRIC_CACHES[memo_id] keep working across a cache
# expiry without app.py needing to know it happened.

_RUBRIC_STORE: dict[str, dict] = {}
# stable_handle -> {
#     "rubric_text":       str,
#     "subject_name":      str,
#     "ttl_minutes":        int,
#     "gemini_cache_name":  str,   # current live Gemini resource name
# }


def create_rubric_cache(rubric_text: str, subject_name: str = "General",
                        ttl_minutes: int = 120):
    """
    Creates a Gemini context cache for THIS rubric — the actual text a
    teacher uploaded, not a hardcoded sample. Returns an object with
    `.name` (the stable handle to pass into mark_student_submission's
    cache_name= later) and `.expire_time`.

    Called once per memo, from app.py's /api/marking/init-cache. The
    returned handle should be stored by the caller (app.py keeps it in
    ACTIVE_RUBRIC_CACHES[memo_id]) and reused for every subsequent
    /evaluate call against that memo — do not call this again per student.
    """
    if client is None:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. A rubric cache requires Gemini even "
            "though marking itself is Groq-primary, because Groq has no "
            "equivalent to Gemini's cached-content feature — Groq instead "
            "receives rubric_text directly on every marking call, read back "
            "out of this same store."
        )

    print(f"Creating rubric cache for subject={subject_name!r} (TTL: {ttl_minutes}m)...")

    cache = client.caches.create(
        model=GEMINI_CACHE_MODEL,
        config=types.CreateCachedContentConfig(
            contents=[rubric_text],
            system_instruction=(
                f"You are an expert automated {subject_name} exam evaluator. "
                "Mark student answers strictly against this memorandum."
            ),
            ttl=f"{ttl_minutes * 60}s",
        )
    )
    print(f"✓ Cache created: {cache.name} (expires {cache.expire_time})")

    stable_handle = cache.name
    _RUBRIC_STORE[stable_handle] = {
        "rubric_text": rubric_text,
        "subject_name": subject_name,
        "ttl_minutes": ttl_minutes,
        "gemini_cache_name": cache.name,
    }
    return cache


def _get_live_gemini_cache_name(stable_handle: str, entry: dict) -> str:
    """
    Returns a Gemini cache resource name that's actually still valid,
    recreating it from the stored rubric_text if the original has expired.
    The stable_handle's entry in _RUBRIC_STORE is updated in place, so
    future calls with the same stable_handle reuse the fresh cache.
    """
    live_name = entry["gemini_cache_name"]
    try:
        # A cheap existence/liveness check — raises if the resource has
        # expired or been deleted server-side.
        client.caches.get(name=live_name)
        return live_name
    except Exception as e:
        print(f"    Rubric cache {live_name} unavailable ({type(e).__name__}) — recreating from stored rubric_text")
        cache = client.caches.create(
            model=GEMINI_CACHE_MODEL,
            config=types.CreateCachedContentConfig(
                contents=[entry["rubric_text"]],
                system_instruction=(
                    f"You are an expert automated {entry['subject_name']} exam evaluator. "
                    "Mark student answers strictly against this memorandum."
                ),
                ttl=f"{entry['ttl_minutes'] * 60}s",
            )
        )
        entry["gemini_cache_name"] = cache.name
        print(f"✓ Recreated cache: {cache.name} (expires {cache.expire_time})")
        return cache.name


def _mark_gemini_cached(stable_handle: str, entry: dict, student_id: str,
                        student_answers: str) -> str:
    """
    Evaluates one student's script using the (possibly just-recreated)
    cached rubric. Rescue path — only reached when Groq is unconfigured, in
    cooldown, over its TPM budget, or actually errors.
    """
    live_cache_name = _get_live_gemini_cache_name(stable_handle, entry)

    prompt = f"""
    Evaluate the following student submission for Student ID: {student_id}.

    STUDENT ANSWERS:
    {student_answers}

    Provide output in structured JSON format with:
    - total_score
    - breakdown: list of objects (question, mark_awarded, max_mark, feedback)
    """

    response = client.models.generate_content(
        model=GEMINI_MODEL_MARK,
        contents=prompt,
        config=types.GenerateContentConfig(
            cached_content=live_cache_name,
            response_mime_type="application/json",
            temperature=0.1,  # Low variance for consistent grading
        )
    )

    usage = response.usage_metadata
    print(f"    [gemini] cached={usage.cached_content_token_count} "
          f"new_in={usage.prompt_token_count - (usage.cached_content_token_count or 0)} "
          f"out={usage.candidates_token_count}")

    return response.text


def _mark_groq(entry: dict, student_id: str, student_answers: str) -> str:
    """
    Primary path. Groq has no equivalent to Gemini's context caching, so the
    rubric text is sent in full on every call — read back out of the
    rubric store rather than a module-level constant, so this reflects
    whatever memo was actually cached for this handle.

    If a real multi-page marking guideline makes re-sending the full rubric
    on every submission too expensive/slow, that's exactly the case
    Gemini's caching exists for — consider routing large rubrics to the
    Gemini path deliberately rather than forcing Groq to eat the full text
    every time. Not implemented as a size threshold here; flagging so it
    isn't a silent trap once rubrics grow.
    """
    gc = groq_client()
    if gc is None:
        raise RuntimeError("GROQ_API_KEY not set")

    rubric_text = entry["rubric_text"]
    subject_name = entry["subject_name"]

    prompt = f"""{rubric_text}

You are an expert automated {subject_name} exam evaluator. Mark the student
submission below strictly against the memorandum above.

Evaluate the following student submission for Student ID: {student_id}.

STUDENT ANSWERS:
{student_answers}

Respond with ONLY a single JSON object (no markdown fences, no commentary)
with:
- total_score
- breakdown: list of objects (question, mark_awarded, max_mark, feedback)
"""

    resp = gc.chat.completions.create(
        model=GROQ_MODEL_MARK,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    usage = getattr(resp, "usage", None)
    total_tokens = getattr(usage, "total_tokens", None) or _estimate_tokens(prompt)
    _groq_record_usage(total_tokens)
    if usage:
        print(f"    [groq] in={usage.prompt_tokens} out={usage.completion_tokens}")

    parsed = _parse_json_response(resp.choices[0].message.content)
    return json.dumps(parsed)  # keep the same str-return shape as the Gemini path


def mark_student_submission(cache_name: str, student_id: str, student_answers: str) -> str:
    """
    Groq-primary, Gemini-rescue dispatcher. cache_name is the stable handle
    returned by create_rubric_cache() (app.py stores this per memo_id in
    ACTIVE_RUBRIC_CACHES and passes it back here on every /evaluate call).

    Routing:
      1. No entry for cache_name in the rubric store -> the caller passed a
         stale/unknown handle (e.g. init-cache was never called for this
         memo_id, or this process restarted and lost its in-memory store)
         -> raises ValueError with a clear message rather than failing
         deeper in a confusing way.
      2. No GROQ_API_KEY -> Gemini, unconditionally.
      3. Active cooldown or estimated tokens would exceed the rolling TPM
         budget -> Gemini, without attempting Groq.
      4. Otherwise try Groq. RateLimitError or a 413/429-style
         APIStatusError starts a cooldown and falls back to Gemini for this
         call; any other Groq error falls back without starting a cooldown
         (not evidence Groq is out of budget).
    """
    entry = _RUBRIC_STORE.get(cache_name)
    if entry is None:
        raise ValueError(
            f"No rubric cache found for cache_name={cache_name!r}. "
            "Call /api/marking/init-cache for this memo before /evaluate — "
            "or, if this service process restarted since init-cache was "
            "called, the in-memory rubric store was lost and init-cache "
            "needs to be called again."
        )

    if not GROQ_API_KEY:
        return _mark_gemini_cached(cache_name, entry, student_id, student_answers)

    estimated = _estimate_tokens(entry["rubric_text"]) + _estimate_tokens(student_answers)

    if _groq_in_cooldown():
        print(f"    [{student_id}] Groq in cooldown — routing to Gemini")
        return _mark_gemini_cached(cache_name, entry, student_id, student_answers)

    if not _groq_budget_ok(estimated):
        print(f"    [{student_id}] would exceed {GROQ_TPM_BUDGET} TPM budget — routing to Gemini")
        return _mark_gemini_cached(cache_name, entry, student_id, student_answers)

    try:
        return _mark_groq(entry, student_id, student_answers)
    except GroqRateLimitError as e:
        print(f"    [{student_id}] Groq rate limit hit: {e}")
        _groq_start_cooldown()
        return _mark_gemini_cached(cache_name, entry, student_id, student_answers)
    except GroqAPIStatusError as e:
        if e.status_code in (413, 429):
            print(f"    [{student_id}] Groq status {e.status_code}: {e}")
            _groq_start_cooldown()
        else:
            print(f"    [{student_id}] Groq error {e.status_code}, falling back this call only: {e}")
        return _mark_gemini_cached(cache_name, entry, student_id, student_answers)
    except (GroqAPIError, json.JSONDecodeError) as e:
        print(f"    [{student_id}] Groq call failed ({type(e).__name__}: {e}), falling back to Gemini")
        return _mark_gemini_cached(cache_name, entry, student_id, student_answers)


# ══════════════════════════════════════════════════════════════════════════════
# USAGE EXAMPLE (Batch marking a class, using a real rubric end-to-end)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    groq_status = (f"{GROQ_MODEL_MARK} (TPM budget {GROQ_TPM_BUDGET})"
                   if GROQ_API_KEY else "not configured — every call goes to Gemini")
    print("="*64)
    print(f"Primary:  Groq — {groq_status}")
    print(f"Rescue:   Gemini — {GEMINI_MODEL_MARK} (cache created only if/when needed)")
    print("="*64)

    sample_rubric = """
QUESTION 1: DATABASE VALIDATION (10 Marks)
1.1 Explain the difference between Field Size and Validation Rule.
   - Field Size: Sets maximum characters stored (e.g., Text size 20). [1 mark]
   - Validation Rule: Expression that limits input values (e.g., >0 AND <100). [1 mark]
1.2 Write an Access Validation Rule for dates after 01 January 2026.
   - Answer: >#2026/01/01# or >#2026-01-01# [1 mark]
1.3 Identify TWO properties to prevent null entries.
   - Required = Yes [1 mark]
   - Allow Zero Length = No [1 mark]

QUESTION 2: NETWORKS & SECURITY (10 Marks)
2.1 Define Firewall and explain its primary function.
   - Hardware/software filtering network traffic based on security rules. [2 marks]
2.2 Explain Two-Factor Authentication (2FA).
   - Security process requiring two distinct forms of identification before access. [2 marks]
"""

    cache = create_rubric_cache(
        rubric_text=sample_rubric,
        subject_name="Computer Applications Technology",
        ttl_minutes=120,
    )

    submissions = [
        {
            "student_id": "STU_101",
            "answers": "1.1 Field size sets length. Validation rule limits value. 1.2 >#2026/01/01# 2.1 Firewall blocks hackers."
        },
        {
            "student_id": "STU_102",
            "answers": "1.1 Both do the same thing. 1.2 >=2026 2.1 Hardware filtering network traffic based on rules."
        }
    ]

    for sub in submissions:
        print(f"\n--- Marking {sub['student_id']} ---")
        result = mark_student_submission(
            cache_name=cache.name,
            student_id=sub["student_id"],
            student_answers=sub["answers"],
        )
        print(result)