"""
model.py  —  EduCAT answer marking + AI feedback  (LangChain + Groq)

Migration from raw Groq API to LangChain:

WHAT CHANGED
────────────
1. LLM client    : Groq() replaced by ChatGroq from langchain_groq.
2. LLM calls     : client.chat.completions.create() replaced by
                   llm.invoke() which returns an AIMessage object.
                   .content gives the text, identical to .choices[0].message.content.
3. Output parsing: LangChain's JsonOutputParser is used for open-question
                   marking so malformed JSON is handled cleanly without a
                   manual try/except json.loads() block.
4. Prompt objects : PromptTemplate / ChatPromptTemplate replace raw f-strings
                   where it reduces repetition.  Simple single-call prompts
                   remain as plain strings passed to llm.invoke() — that is
                   idiomatic LangChain for short prompts.

WHAT DID NOT CHANGE
───────────────────
- mark_answer() function signature is identical.
- MCQ, True/False, and Matching question types still use pure Python logic
  with zero LLM calls — no change in behaviour or performance.
- Open-question marking logic (meaning comparison, score clamping) is
  identical; only the API call mechanics changed.
- generate_answer() and generate_exam_feedback() return the same strings.
- All callers (agent.py tools, app.py /submit) work without modification.
"""

import os
import re
import json
import time
from dotenv import load_dotenv

# ── LangChain imports ────────────────────────────────────────────────────────
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ── Shared LLM instance ───────────────────────────────────────────────────────
# Re-uses the same model as agent.py.  Having one ChatGroq instance per module
# is fine — each call is stateless.
llm = ChatGroq(
    model        = "llama-3.3-70b-versatile",
    groq_api_key = os.getenv("GROQ_API_KEY"),
)

# JSON output parser — used for open-question marking
# Raises OutputParserException on invalid JSON (caught below)
_json_parser = JsonOutputParser()


# ═══════════════════════════════════════════════════════════════════════════════
# AI TUTOR  —  RAG-grounded answer generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_answer(context: str, question: str) -> str:
    """
    Generate a tutor answer using RAG context from theory books.
    Called by app.py /chat endpoint (legacy) and agent search_theory tool.
    """
    prompt = (
        f"You are a friendly CAT Grade 12 tutor.\n"
        f"Context: {context or 'Use general CAT knowledge.'}\n"
        f"Question: {question}\n"
        f"Answer:"
    )
    try:
        # llm.invoke() returns an AIMessage; .content is the text string
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        return f"⚠️ Error generating answer: {e}"


# ═══════════════════════════════════════════════════════════════════════════════
# ANSWER MARKING
# MCQ / True-False / Matching use pure Python (zero LLM calls).
# Open questions use the LLM with a JsonOutputParser chain.
# ═══════════════════════════════════════════════════════════════════════════════

def mark_answer(
    question: str,
    question_number: str,
    q_type: str,
    student_answer: str,
    memo,
    marks: int,
    options=None,
) -> dict:
    """
    Grade a single student answer.

    memo is always sourced from q['memo'] — never from an external lookup.
    This is the architectural fix that eliminated the Kickstarter/Q1.1 bug.

    Returns:
        dict with keys: score (int), feedback (str), status (str)
    """
    student = str(student_answer).strip() if student_answer else ""

    # ── MCQ — exact letter match, no LLM needed ───────────────────────────────
    if q_type == "mcq":
        correct = str(memo).strip().upper()
        ans     = student.upper()
        if not correct:
            return {"score": 0, "feedback": "No memo available.", "status": "no_memo"}
        if not ans:
            return {"score": 0, "feedback": f"No answer selected. Correct: {correct}.", "status": "missing"}
        # Enrich feedback with option text when available
        opt_text = ""
        for opt in (options or []):
            if isinstance(opt, dict) and opt.get("key", "").upper() == correct:
                opt_text = f" ({opt['value']})"
                break
        if ans == correct:
            return {"score": marks, "feedback": f"Correct! {correct}{opt_text}.", "status": "correct"}
        return {
            "score":    0,
            "feedback": f"Incorrect. You selected {ans}; correct is {correct}{opt_text}.",
            "status":   "incorrect",
        }

    # ── True/False — regex split for correction word ──────────────────────────
    if q_type == "true_false":
        if not memo:
            return {"score": 0, "feedback": "No memo available.", "status": "no_memo"}
        cl = str(memo).strip().lower()
        al = student.lower()
        if not al:
            return {"score": 0, "feedback": f"No answer provided. Correct: {memo}", "status": "missing"}

        correct_is_true = cl.startswith("true")
        student_is_true = al.startswith("true")

        if correct_is_true and student_is_true:
            return {"score": marks, "feedback": "Correct — True.", "status": "correct"}

        if not correct_is_true and not student_is_true:
            # Extract correction words from both sides of the dash/em-dash
            def extract_correction(s):
                parts = re.split(r"[-—]", s, maxsplit=1)
                return parts[1].strip().lower() if len(parts) > 1 else ""

            memo_word    = extract_correction(cl)
            student_word = extract_correction(al)

            if not memo_word or (student_word and (memo_word in student_word or student_word in memo_word)):
                return {
                    "score":    marks,
                    "feedback": f"Correct — False, correction: {student_word or memo_word}.",
                    "status":   "correct",
                }
            return {
                "score":    marks // 2,
                "feedback": (
                    f"Correct that it is FALSE, but wrong correction. "
                    f"Expected '{memo_word}', got '{student_word or '(none)'}'."
                ),
                "status":   "partial",
            }

        return {"score": 0, "feedback": f"Incorrect. Correct answer: {memo}.", "status": "incorrect"}

    # ── Matching — per-pair letter scoring ────────────────────────────────────
    if q_type == "matching":
        if not isinstance(memo, dict) or not memo:
            return {"score": 0, "feedback": "No memo for matching.", "status": "no_memo"}
        try:
            student_map = json.loads(student) if student else {}
        except Exception:
            student_map = {}

        correct_count = 0
        details       = []
        for col_a_item, correct_letter in memo.items():
            student_val    = student_map.get(col_a_item, "")
            # Student value looks like "R. convergence" — extract letter only
            student_letter = student_val.strip().split(".")[0].strip().upper() if student_val else ""
            correct_clean  = str(correct_letter).strip().upper()
            if student_letter == correct_clean:
                correct_count += 1
                details.append(f"✅ {col_a_item.split()[0]}: {student_letter}")
            else:
                details.append(
                    f"❌ {col_a_item.split()[0]}: got '{student_letter or '—'}' need '{correct_clean}'"
                )

        total_pairs = len(memo)
        earned      = round((correct_count / total_pairs) * marks) if total_pairs else 0
        status      = "correct" if earned == marks else "partial" if earned > 0 else "incorrect"
        return {
            "score":    earned,
            "feedback": f"{correct_count}/{total_pairs} correct. " + " | ".join(details),
            "status":   status,
        }

    # ── Open — LangChain chain: prompt | llm | JsonOutputParser ───────────────
    if not student:
        return {"score": 0, "feedback": "No answer provided.", "status": "missing"}

    memo_text = memo if isinstance(memo, str) else json.dumps(memo) if memo else ""
    has_memo  = bool(memo_text.strip())

    # ChatPromptTemplate with a system + human turn for structured marking
    marking_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a strict South African NSC CAT examiner. "
            "Award 1 mark per distinct correct fact. "
            "Return ONLY a valid JSON object — no markdown, no explanation.",
        ),
        (
            "human",
            "Question {question_number} ({marks} mark{plural}):\n"
            "{question}\n\n"
            "{memo_section}"
            "Student's answer:\n{student_answer}\n\n"
            "Return JSON: "
            '{{ "score": <int 0-{marks}>, '
            '"feedback": "<what was correct, what was missing>", '
            '"status": "<correct|partial|incorrect>" }}',
        ),
    ])

    # Build the chain: prompt → LLM → JSON parser
    marking_chain = marking_prompt | llm | _json_parser

    for attempt in range(2):
        try:
            result = marking_chain.invoke({
                "question_number": question_number,
                "marks":           marks,
                "plural":          "s" if marks != 1 else "",
                "question":        question,
                "memo_section":    (
                    f"Marking guideline:\n{memo_text}\n\n" if has_memo
                    else "No guideline — use CAT Grade 12 knowledge.\n\n"
                ),
                "student_answer":  student,
            })
            # Clamp score to valid range
            result["score"] = max(0, min(int(result.get("score", 0)), marks))
            return result

        except Exception as e:
            if attempt == 0:
                time.sleep(5)   # back off once before retrying
            else:
                return {"score": 0, "feedback": f"Could not mark: {e}", "status": "incorrect"}


# ═══════════════════════════════════════════════════════════════════════════════
# EXAM FEEDBACK SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def generate_exam_feedback(results: list, score: int, total: int, percentage: float) -> str:
    """
    Generate a short personalised performance summary after exam submission.
    Called by app.py /submit after all questions are marked.
    """
    wrong = [r["question_number"] for r in results if r["status"] != "correct"]
    prompt = (
        f"You are a motivating CAT Grade 12 teacher.\n"
        f"Score: {score}/{total} ({percentage}%)\n"
        f"Wrong/partial questions: {', '.join(wrong) if wrong else 'none — perfect score!'}\n\n"
        f"Write 3-4 sentences of encouraging, specific feedback. "
        f"Mention topics to revise if applicable."
    )
    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        return f"Score: {score}/{total} ({percentage}%). Keep practising! 🚀"