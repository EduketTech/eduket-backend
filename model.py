"""
model.py — EduCAT Universal Answer Marking + AI Feedback

MARKING STRATEGY:
  Every question type is marked using the best available signal:

  MCQ / True-False / Matching
    → Pure Python — zero LLM, instant, no tokens used

  Short answer / Open / Comprehension / Diagram / Table
    → memo  +  subject AI knowledge combined
    → If memo present: AI checks student answer against memo AND subject context
    → If no memo: AI marks purely from NSC subject knowledge
    → Partial credit awarded where student shows partial understanding

  Calculation
    → Step-by-step: formula check, substitution, answer, units
    → subject config drives weighting

  Essay
    → Rubric-based via LLM: content, structure, language, evidence

  All types fall back gracefully if LLM call fails.
"""

import os
import re
import json
import time
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY"),
)
_json_parser = JsonOutputParser()


# ═══════════════════════════════════════════════════════════════════════════════
# SUBJECT CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

SUBJECT_MARKING_CONFIG = {
    "mathematics": {
        "step_marks": True, "formula_weight": 0.3,
        "answer_weight": 0.4, "working_weight": 0.3,
        "unit_required": False,
        "rubric_criteria": ["correct_formula","correct_substitution","correct_calculation","correct_answer"],
    },
    "mathematical literacy": {
        "step_marks": True, "formula_weight": 0.2,
        "answer_weight": 0.5, "working_weight": 0.3,
        "unit_required": True,
        "rubric_criteria": ["correct_method","correct_calculation","correct_units","correct_answer"],
    },
    "physical sciences": {
        "step_marks": True, "formula_weight": 0.25,
        "answer_weight": 0.35, "working_weight": 0.25, "concept_weight": 0.15,
        "unit_required": True,
        "rubric_criteria": ["correct_concept","correct_formula","correct_substitution","correct_calculation","correct_answer","correct_units"],
    },
    "life sciences": {
        "step_marks": False, "terminology_weight": 0.4,
        "explanation_weight": 0.4, "accuracy_weight": 0.2,
        "rubric_criteria": ["correct_terminology","clear_explanation","scientific_accuracy"],
    },
    "geography": {
        "step_marks": False, "fact_weight": 0.5,
        "explanation_weight": 0.3, "example_weight": 0.2,
        "rubric_criteria": ["correct_facts","clear_explanation","relevant_examples"],
    },
    "history": {
        "step_marks": False, "argument_weight": 0.3,
        "evidence_weight": 0.3, "perspective_weight": 0.2, "structure_weight": 0.2,
        "rubric_criteria": ["clear_argument","relevant_evidence","multiple_perspectives","logical_structure"],
    },
    "accounting": {
        "step_marks": True, "calculation_weight": 0.4,
        "concept_weight": 0.3, "presentation_weight": 0.3,
        "rubric_criteria": ["correct_calculation","correct_concept","proper_format"],
    },
    "economics": {
        "step_marks": False, "definition_weight": 0.3,
        "application_weight": 0.4, "evaluation_weight": 0.3,
        "rubric_criteria": ["correct_definitions","real_world_application","critical_evaluation"],
    },
    "business studies": {
        "step_marks": False, "knowledge_weight": 0.3,
        "application_weight": 0.4, "analysis_weight": 0.3,
        "rubric_criteria": ["factual_knowledge","case_application","critical_analysis"],
    },
    "computer applications technology": {
        "step_marks": False, "fact_weight": 0.5,
        "explanation_weight": 0.3, "example_weight": 0.2,
        "rubric_criteria": ["correct_facts","clear_explanation","relevant_examples"],
    },
    "cat": {
        "step_marks": False, "fact_weight": 0.5,
        "explanation_weight": 0.3, "example_weight": 0.2,
        "rubric_criteria": ["correct_facts","clear_explanation","relevant_examples"],
    },
    "information technology": {
        "step_marks": True, "code_weight": 0.4,
        "logic_weight": 0.3, "output_weight": 0.3,
        "rubric_criteria": ["correct_syntax","correct_logic","expected_output"],
    },
    "english": {
        "step_marks": False, "content_weight": 0.4,
        "language_weight": 0.3, "structure_weight": 0.3,
        "rubric_criteria": ["relevant_content","language_quality","textual_structure"],
    },
    "afrikaans": {
        "step_marks": False, "content_weight": 0.4,
        "language_weight": 0.3, "structure_weight": 0.3,
        "rubric_criteria": ["relevant_content","language_quality","textual_structure"],
    },
}


def get_subject_config(subject: str) -> dict:
    key = subject.lower().strip()
    return SUBJECT_MARKING_CONFIG.get(key, {
        "step_marks": False, "fact_weight": 0.5,
        "explanation_weight": 0.3, "example_weight": 0.2,
        "rubric_criteria": ["correct_facts","clear_explanation","relevant_examples"],
    })


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def extract_keywords(text: str) -> set:
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    stops = {
        'the','and','for','are','but','not','you','all','can','had','her','was',
        'one','our','out','get','has','him','his','how','its','may','new','now',
        'see','two','who','did','she','use','way','many','sit','set','run','eat',
        'far','sea','eye','ago','off','too','any','say','man','try','ask','end',
        'why','let','put','tell','very','when','much','would','there','their',
        'what','said','each','which','will','about','could','other','after',
        'first','never','these','think','where','being','every','great','might',
        'shall','still','those','while','this','that','with','have','from',
        'they','know','want','been','good','some','time','come','here','just',
        'like','long','make','over','such','take','than','them','well','were',
    }
    return {w for w in words if w not in stops}


def keyword_overlap(student: str, memo: str) -> float:
    sk = extract_keywords(student)
    mk = extract_keywords(memo)
    if not mk:
        return 0.0
    return len(sk & mk) / len(sk | mk) if (sk | mk) else 0.0


def compare_numerical(student: str, correct: str, tolerance: float = 0.02) -> bool:
    try:
        sv = float(re.findall(r'-?\d+\.?\d*', student)[-1])
        cv = float(re.findall(r'-?\d+\.?\d*', correct)[-1])
        return abs(sv - cv) <= tolerance * abs(cv) if cv else sv == cv
    except (ValueError, IndexError):
        return False


def _llm_call(prompt_messages: list, invoke_vars: dict, marks: int, fallback_memo: str = "") -> dict:
    """
    Shared LLM call with retry + keyword fallback.
    prompt_messages: list of (role, template) tuples for ChatPromptTemplate
    """
    chain = ChatPromptTemplate.from_messages(prompt_messages) | llm | _json_parser
    for attempt in range(2):
        try:
            result = chain.invoke(invoke_vars)
            result["score"] = max(0, min(int(result.get("score", 0)), marks))
            if result["score"] == marks:
                result["status"] = "correct"
            elif result["score"] > 0:
                result["status"] = "partial"
            else:
                result["status"] = "incorrect"
            return result
        except Exception as e:
            if attempt == 0:
                time.sleep(4)
            else:
                # Keyword fallback
                student_text = invoke_vars.get("student_answer", "")
                overlap = keyword_overlap(student_text, fallback_memo)
                score = int(overlap * marks)
                return {
                    "score": score,
                    "feedback": f"Auto-marked (keyword match {overlap:.0%}). Manual review recommended.",
                    "status": "correct" if score == marks else "partial" if score > marks * 0.3 else "incorrect",
                }


# ═══════════════════════════════════════════════════════════════════════════════
# GENERATE ANSWER (AI Tutor)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_answer(context: str, question: str) -> str:
    prompt = (
        f"You are a friendly NSC Grade 12 tutor.\n"
        f"Context: {context or 'Use general NSC knowledge.'}\n"
        f"Question: {question}\nAnswer:"
    )
    try:
        return llm.invoke(prompt).content.strip()
    except Exception as e:
        return f"Error generating answer: {e}"


# ═══════════════════════════════════════════════════════════════════════════════
# MARKING — PURE PYTHON (no LLM)
# ═══════════════════════════════════════════════════════════════════════════════

def _mark_mcq(student: str, memo: str, marks: int, options) -> dict:
    correct = str(memo).strip().upper()
    ans = student.strip().upper()
    if not correct:
        return {"score": 0, "feedback": "No memo available.", "status": "no_memo"}
    if not ans:
        return {"score": 0, "feedback": f"No answer selected. Correct: {correct}.", "status": "missing"}

    # Find option text for feedback
    opt_text = ""
    if isinstance(options, dict) and correct in options:
        opt_text = f" — {options[correct]}"
    elif isinstance(options, list):
        for o in options:
            if isinstance(o, dict) and o.get("key","").upper() == correct:
                opt_text = f" — {o['value']}"
                break

    if ans == correct:
        return {"score": marks, "feedback": f"Correct! Answer: {correct}{opt_text}.", "status": "correct"}
    return {
        "score": 0,
        "feedback": f"Incorrect. You chose {ans}; correct is {correct}{opt_text}.",
        "status": "incorrect",
    }


def _mark_true_false(student: str, memo: str, marks: int) -> dict:
    if not memo:
        return {"score": 0, "feedback": "No memo available.", "status": "no_memo"}
    if not student:
        return {"score": 0, "feedback": f"No answer provided. Correct: {memo}", "status": "missing"}

    cl = str(memo).strip().lower()
    al = student.lower()
    correct_true = cl.startswith("true")
    student_true = al.startswith("true")

    def get_correction(s):
        parts = re.split(r"[-—]", s, maxsplit=1)
        return parts[1].strip().lower() if len(parts) > 1 else ""

    if correct_true and student_true:
        return {"score": marks, "feedback": "Correct — True.", "status": "correct"}

    if not correct_true and not student_true:
        memo_word = get_correction(cl)
        student_word = get_correction(al)
        if not memo_word or (student_word and (memo_word in student_word or student_word in memo_word)):
            return {"score": marks, "feedback": f"Correct — False, correction: {student_word or memo_word}.", "status": "correct"}
        return {
            "score": marks // 2 if marks > 1 else 0,
            "feedback": f"Correct it is FALSE, but wrong correction. Expected '{memo_word}', got '{student_word or '(none)'}'.",
            "status": "partial",
        }

    return {"score": 0, "feedback": f"Incorrect. Correct answer: {memo}.", "status": "incorrect"}


def _mark_matching(student: str, memo, marks: int) -> dict:
    if not isinstance(memo, dict) or not memo:
        return {"score": 0, "feedback": "No memo for matching.", "status": "no_memo"}
    try:
        student_map = json.loads(student) if student else {}
    except Exception:
        student_map = {}

    correct_count = 0
    details = []
    for col_a, correct_val in memo.items():
        student_val = student_map.get(col_a, "")
        student_letter = student_val.strip().split(".")[0].strip().upper() if student_val else ""
        correct_clean = str(correct_val).strip().upper()
        if student_letter == correct_clean:
            correct_count += 1
            details.append(f"✅ {str(col_a)[:20]}: {student_letter}")
        else:
            details.append(f"❌ {str(col_a)[:20]}: got '{student_letter or '—'}' need '{correct_clean}'")

    total = len(memo)
    earned = round((correct_count / total) * marks) if total else 0
    status = "correct" if earned == marks else "partial" if earned > 0 else "incorrect"
    return {
        "score": earned,
        "feedback": f"{correct_count}/{total} correct. " + " | ".join(details),
        "status": status,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MARKING — AI-POWERED (memo + subject knowledge combined)
# ═══════════════════════════════════════════════════════════════════════════════

def _mark_open_with_ai(
    question: str, student: str, memo: str, marks: int, subject: str,
    q_type: str = "open", extra_instruction: str = ""
) -> dict:
    """
    Core AI marking function used by open, short_answer, comprehension etc.

    Combines:
    1. Memo answer (if available) — primary marking guide
    2. NSC subject knowledge — catches correct answers not in memo
    3. Subject-specific criteria — terminology, examples, structure etc.

    This means a student can get credit for a correct answer even if worded
    differently from the memo, as long as the subject AI confirms it's valid.
    """
    if not student:
        return {"score": 0, "feedback": "No answer provided.", "status": "missing"}

    has_memo = bool(memo and str(memo).strip())
    config = get_subject_config(subject)
    criteria = config.get("rubric_criteria", ["correct_facts", "clear_explanation"])

    memo_section = (
        f"MARKING MEMORANDUM:\n{memo}\n\n"
        f"Important: Award marks for answers that convey the same meaning as the memo, "
        f"even if worded differently. Also award marks for additional correct points "
        f"that are not in the memo but are factually correct for {subject or 'this subject'}.\n\n"
        if has_memo else
        f"No memo provided. Mark based on your NSC Grade 12 {subject or 'subject'} knowledge. "
        f"Award full marks only for complete, accurate answers.\n\n"
    )

    messages = [
        ("system",
         f"You are a strict but fair South African NSC examiner marking {subject or 'a'} paper. "
         f"Award marks for correct content regardless of phrasing. "
         f"Give partial credit where student shows partial understanding. "
         f"Return ONLY valid JSON — no markdown, no explanation outside JSON."),
        ("human",
         f"QUESTION ({marks} mark{'s' if marks != 1 else ''}):\n{{question}}\n\n"
         f"{memo_section}"
         f"MARKING CRITERIA: {{criteria}}\n\n"
         f"STUDENT ANSWER:\n{{student_answer}}\n\n"
         f"{extra_instruction}"
         f"Return: {{{{\"score\": <int 0-{marks}>, "
         f"\"feedback\": \"<specific: what was correct, what was missing>\", "
         f"\"status\": \"<correct|partial|incorrect>\"}}}}"),
    ]

    return _llm_call(
        messages,
        {"question": question, "student_answer": student, "criteria": ", ".join(criteria)},
        marks,
        fallback_memo=memo or question,
    )


def _mark_calculation(question: str, student: str, memo: str, marks: int, subject: str) -> dict:
    """Mark calculations with step-by-step partial credit."""
    if not student:
        return {"score": 0, "feedback": "No answer provided.", "status": "missing"}
    if not memo:
        return _mark_open_with_ai(question, student, "", marks, subject, "calculation",
                                   "Check method, working shown, and final answer. ")

    config = get_subject_config(subject)
    score = 0
    parts = []

    # Check working shown
    has_working = len(student.split('\n')) > 1 or student.count('=') > 1
    if has_working:
        parts.append("Working shown ✓")

    # Check numerical answer
    if compare_numerical(student, memo):
        score += max(1, int(marks * config.get("answer_weight", 0.5)))
        parts.append("Correct answer ✓")
    else:
        parts.append("Answer incorrect")

    # Check units if required
    if config.get("unit_required"):
        unit_pattern = r'\b(m|km|kg|g|cm|mm|ml|l|s|min|h|N|J|W|Pa|V|A|Hz|°C|mol)\b'
        s_units = re.findall(unit_pattern, student.lower())
        m_units = re.findall(unit_pattern, str(memo).lower())
        if s_units and m_units and any(u in m_units for u in s_units):
            score += 1
            parts.append("Correct units ✓")
        else:
            parts.append("Check units")

    score = max(0, min(score, marks))

    # If pure Python check inconclusive, use AI to verify working
    if score == 0 and has_working:
        ai_result = _mark_open_with_ai(
            question, student, memo, marks, subject, "calculation",
            "This is a calculation. Check each step: formula, substitution, arithmetic, answer. "
        )
        return ai_result

    return {
        "score": score,
        "feedback": " | ".join(parts),
        "status": "correct" if score == marks else "partial" if score > 0 else "incorrect",
    }


def _mark_essay(question: str, student: str, memo: str, marks: int, subject: str) -> dict:
    """Mark essays using rubric via LLM."""
    if not student:
        return {"score": 0, "feedback": "No essay submitted.", "status": "missing"}

    config = get_subject_config(subject)
    criteria = config.get("rubric_criteria", ["content", "structure", "language"])
    word_count = len(student.split())

    messages = [
        ("system",
         f"You are a strict NSC examiner marking a {subject or 'Grade 12'} essay. "
         f"Mark according to CAPS standards. Return ONLY JSON."),
        ("human",
         f"QUESTION ({marks} marks):\n{{question}}\n\n"
         f"{'MARKING GUIDELINE:\\n' + memo + chr(10) + chr(10) if memo else 'Mark from subject knowledge.\\n\\n'}"
         f"RUBRIC CRITERIA: {{criteria}}\n\n"
         f"STUDENT ESSAY ({word_count} words):\n{{student_answer}}\n\n"
         f"Return: {{{{\"score\": <int 0-{marks}>, "
         f"\"feedback\": \"<criterion-by-criterion feedback>\", "
         f"\"status\": \"<correct|partial|incorrect>\"}}}}"),
    ]

    result = _llm_call(
        messages,
        {"question": question, "student_answer": student, "criteria": ", ".join(criteria)},
        marks,
        fallback_memo=memo or question,
    )
    result["feedback"] = f"Word count: {word_count}. " + result.get("feedback", "")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN DISPATCHER
# ═══════════════════════════════════════════════════════════════════════════════

def mark_answer(
    question: str,
    question_number: str,
    q_type: str,
    student_answer: str,
    memo,
    marks: int,
    options=None,
    instructions: str = "",
    subject: str = "",
    sub_parts: list = None,
) -> dict:
    """
    Universal answer marking dispatcher.

    MARKING LOGIC BY TYPE:
      mcq, true_false, matching  → pure Python (fast, accurate, no tokens)
      calculation                → Python step check + AI fallback
      essay                      → AI rubric marking
      short_answer, open,
      comprehension, diagram,
      table, multi_part          → AI with memo + subject knowledge combined
    """
    q_type = (q_type or "open").lower().strip()
    student = str(student_answer).strip() if student_answer else ""
    memo_str = str(memo).strip() if memo and not isinstance(memo, dict) else (memo or "")
    marks = max(1, int(marks or 1))

    # ── Pure Python types ─────────────────────────────────────────────────────
    if q_type == "mcq":
        # Normalise options for dispatcher
        opts = options
        if isinstance(opts, list) and opts and isinstance(opts[0], dict):
            opts = {o["key"]: o["value"] for o in opts}
        return _mark_mcq(student, str(memo_str), marks, opts)

    if q_type == "true_false":
        return _mark_true_false(student, str(memo_str), marks)

    if q_type == "matching":
        memo_dict = memo if isinstance(memo, dict) else {}
        if not memo_dict and isinstance(memo_str, str):
            try:
                memo_dict = json.loads(memo_str)
            except Exception:
                pass
        return _mark_matching(student, memo_dict, marks)

    # ── Calculation ───────────────────────────────────────────────────────────
    if q_type == "calculation":
        return _mark_calculation(question, student, str(memo_str), marks, subject)

    # ── Essay ─────────────────────────────────────────────────────────────────
    if q_type == "essay":
        return _mark_essay(question, student, str(memo_str), marks, subject)

    # ── All AI-marked types ───────────────────────────────────────────────────
    # short_answer, open, comprehension, diagram_label,
    # table_completion, multi_part, unknown — all use the combined
    # memo + subject AI marker

    extra = ""
    if q_type == "short_answer":
        extra = "Award 1 mark per correct distinct point. Do not penalise for minor phrasing differences. "
    elif q_type == "comprehension":
        extra = "Check that the answer refers to the passage/context. Credit inference backed by evidence. "
    elif q_type == "diagram_label":
        extra = "Accept synonyms for diagram labels. Mark part-by-part. "
    elif q_type == "table_completion":
        extra = "Check each cell. Accept equivalent values (e.g. numerical equality within 2%). "
    elif q_type == "multi_part":
        extra = "This is a multi-part answer. Mark each sub-part separately and sum. "
    elif instructions:
        extra = f"Special instruction: {instructions}. "

    return _mark_open_with_ai(question, student, str(memo_str), marks, subject, q_type, extra)


# ═══════════════════════════════════════════════════════════════════════════════
# EXAM FEEDBACK SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def generate_exam_feedback(
    results: list, score: int, total: int, percentage: float, subject: str = ""
) -> str:
    """
    Generate personalised performance summary after submission.
    Subject-aware with specific study recommendations.
    """
    wrong = [r for r in results if r.get("status") not in ("correct",)]
    wrong_by_type: dict = {}
    for r in wrong:
        qt = r.get("type", "open")
        wrong_by_type.setdefault(qt, []).append(r.get("question_number", "?"))

    type_labels = {
        "mcq": "Multiple Choice", "matching": "Matching",
        "true_false": "True/False", "calculation": "Calculations",
        "essay": "Essay Writing", "short_answer": "Short Answers",
        "comprehension": "Comprehension", "diagram_label": "Diagram Labelling",
        "table_completion": "Table Completion", "multi_part": "Multi-Part",
        "open": "Open-Ended",
    }

    weak_lines = "\n".join(
        f"- {type_labels.get(qt, qt.title())}: Q{', Q'.join(nums[:5])}{'...' if len(nums) > 5 else ''}"
        for qt, nums in wrong_by_type.items()
    )

    prompt = (
        f"You are a motivating NSC Grade 12 {'` + subject + `' if subject else 'teacher'}.\n"
        f"Student scored: {score}/{total} ({percentage}%)\n"
        f"Incorrect/partial: {len(wrong)} questions\n"
        f"{('Weak areas:\n' + weak_lines) if weak_lines else 'All correct!'}\n\n"
        f"Write 4-5 sentences of encouraging, specific feedback. "
        f"Mention exactly which topics and question types to revise. "
        f"Give ONE concrete study tip for their weakest area. "
        f"Keep it motivating and practical."
    )

    try:
        return llm.invoke(prompt).content.strip()
    except Exception:
        if percentage >= 70:
            return f"Excellent work! {score}/{total} ({percentage}%). Keep it up! 🎉"
        elif percentage >= 50:
            return f"Good effort! {score}/{total} ({percentage}%). Focus on: {', '.join(wrong_by_type.keys())}. 📈"
        else:
            return f"Keep going! {score}/{total} ({percentage}%). Review: {', '.join(wrong_by_type.keys())}. 💪"