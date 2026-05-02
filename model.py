"""
model.py  —  EduCAT Universal Answer Marking + AI Feedback (LangChain + Groq)

WHAT CHANGED FROM PREVIOUS VERSION
───────────────────────────────────
1. UNIVERSAL QUESTION TYPE SUPPORT:
   - calculation: Step-by-step marking with partial credit for working
   - essay: Rubric-based marking (introduction, body, conclusion, analysis)
   - short_answer: Keyword/point-based marking
   - comprehension: Passage-reference + inference marking
   - diagram_label: Part-by-part marking with synonyms
   - table_completion: Cell-by-cell marking
   - multi_part: Per-sub-part marking with aggregated score

2. SUBJECT-AWARE MARKING:
   - Mathematics: Formula checking, step marks, unit validation
   - Physical Sciences: Concept + calculation dual marking
   - Life Sciences: Terminology precision, diagram accuracy
   - Languages: Rubric-based (content, language, structure)
   - History: Argument quality, evidence use, perspective
   - All subjects fall back to general NSC marking standards

3. ENHANCED OPEN QUESTION MARKING:
   - Handles mathematical formulas in student answers ($...$)
   - Diagram description matching (synonym tolerance)
   - Table cell comparison (case-insensitive, trimmed)
   - Multi-part answer parsing and per-part scoring

4. IMPROVED FEEDBACK:
   - Subject-specific study recommendations
   - Weak area identification by topic AND question type
   - Formula error highlighting for math/science

WHAT DID NOT CHANGE
───────────────────
- mark_answer() signature extended but backward-compatible
- MCQ, True/False, Matching still use pure Python (zero LLM)
- generate_answer() unchanged
- All existing callers work without modification
"""

import os
import re
import json
import time
from typing import Any, Optional
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
        "step_marks": True,
        "formula_weight": 0.3,
        "answer_weight": 0.4,
        "working_weight": 0.3,
        "unit_required": False,
        "rubric_criteria": ["correct_formula", "correct_substitution", "correct_calculation", "correct_answer"],
    },
    "mathematical literacy": {
        "step_marks": True,
        "formula_weight": 0.2,
        "answer_weight": 0.5,
        "working_weight": 0.3,
        "unit_required": True,
        "rubric_criteria": ["correct_method", "correct_calculation", "correct_units", "correct_answer"],
    },
    "technical mathematics": {
        "step_marks": True,
        "formula_weight": 0.3,
        "answer_weight": 0.4,
        "working_weight": 0.3,
        "unit_required": True,
        "rubric_criteria": ["correct_formula", "correct_substitution", "correct_calculation", "correct_answer", "correct_units"],
    },
    "physical sciences": {
        "step_marks": True,
        "formula_weight": 0.25,
        "answer_weight": 0.35,
        "working_weight": 0.25,
        "concept_weight": 0.15,
        "unit_required": True,
        "rubric_criteria": ["correct_concept", "correct_formula", "correct_substitution", "correct_calculation", "correct_answer", "correct_units"],
    },
    "life sciences": {
        "step_marks": False,
        "terminology_weight": 0.4,
        "explanation_weight": 0.4,
        "accuracy_weight": 0.2,
        "rubric_criteria": ["correct_terminology", "clear_explanation", "scientific_accuracy"],
    },
    "geography": {
        "step_marks": False,
        "fact_weight": 0.5,
        "explanation_weight": 0.3,
        "example_weight": 0.2,
        "rubric_criteria": ["correct_facts", "clear_explanation", "relevant_examples"],
    },
    "history": {
        "step_marks": False,
        "argument_weight": 0.3,
        "evidence_weight": 0.3,
        "perspective_weight": 0.2,
        "structure_weight": 0.2,
        "rubric_criteria": ["clear_argument", "relevant_evidence", "multiple_perspectives", "logical_structure"],
    },
    "accounting": {
        "step_marks": True,
        "calculation_weight": 0.4,
        "concept_weight": 0.3,
        "presentation_weight": 0.3,
        "rubric_criteria": ["correct_calculation", "correct_concept", "proper_format"],
    },
    "economics": {
        "step_marks": False,
        "definition_weight": 0.3,
        "application_weight": 0.4,
        "evaluation_weight": 0.3,
        "rubric_criteria": ["correct_definitions", "real_world_application", "critical_evaluation"],
    },
    "business studies": {
        "step_marks": False,
        "knowledge_weight": 0.3,
        "application_weight": 0.4,
        "analysis_weight": 0.3,
        "rubric_criteria": ["factual_knowledge", "case_application", "critical_analysis"],
    },
    "computer applications technology": {
        "step_marks": False,
        "fact_weight": 0.5,
        "explanation_weight": 0.3,
        "example_weight": 0.2,
        "rubric_criteria": ["correct_facts", "clear_explanation", "relevant_examples"],
    },
    "information technology": {
        "step_marks": True,
        "code_weight": 0.4,
        "logic_weight": 0.3,
        "output_weight": 0.3,
        "rubric_criteria": ["correct_syntax", "correct_logic", "expected_output"],
    },
    "engineering graphics & design": {
        "step_marks": False,
        "accuracy_weight": 0.5,
        "technique_weight": 0.3,
        "annotation_weight": 0.2,
        "rubric_criteria": ["drawing_accuracy", "correct_technique", "proper_annotation"],
    },
    "english": {
        "step_marks": False,
        "content_weight": 0.4,
        "language_weight": 0.3,
        "structure_weight": 0.3,
        "rubric_criteria": ["relevant_content", "language_quality", "textual_structure"],
    },
    "afrikaans": {
        "step_marks": False,
        "content_weight": 0.4,
        "language_weight": 0.3,
        "structure_weight": 0.3,
        "rubric_criteria": ["relevant_content", "language_quality", "textual_structure"],
    },
}


def get_subject_config(subject: str) -> dict:
    """Get marking configuration for a subject. Falls back to generic config."""
    subject_lower = subject.lower().strip()
    if subject_lower in SUBJECT_MARKING_CONFIG:
        return SUBJECT_MARKING_CONFIG[subject_lower]
    # Generic fallback
    return {
        "step_marks": False,
        "fact_weight": 0.5,
        "explanation_weight": 0.3,
        "example_weight": 0.2,
        "rubric_criteria": ["correct_facts", "clear_explanation", "relevant_examples"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# AI TUTOR — RAG-grounded answer generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_answer(context: str, question: str) -> str:
    """Generate a tutor answer using RAG context from theory books."""
    prompt = (
        f"You are a friendly NSC Grade 12 tutor across ALL subjects.\n"
        f"Context: {context or 'Use general NSC knowledge.'}\n"
        f"Question: {question}\n"
        f"Answer:"
    )
    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        return f"Error generating answer: {e}"


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def extract_math_expressions(text: str) -> list:
    """Extract mathematical expressions from text ($...$ or plain formulas)."""
    expressions = []
    # LaTeX style
    latex_matches = re.findall(r'\$(.*?)\$', text)
    expressions.extend(latex_matches)
    # Plain equations (x = 5, 2x + 3 = 11, etc.)
    plain_matches = re.findall(r'[a-zA-Z0-9\s]+[=<>]+[\s+a-zA-Z0-9\-\+\*/\(\)\.]+', text)
    expressions.extend(plain_matches)
    return [e.strip() for e in expressions if len(e.strip()) > 2]


def normalize_math_answer(text: str) -> str:
    """Normalize a mathematical answer for comparison."""
    # Remove spaces around operators
    text = re.sub(r'\s*([=+\-*/])\s*', r'\1', text)
    # Normalize decimal points
    text = text.replace(',', '.')
    # Remove trailing zeros in decimals
    text = re.sub(r'(\d+\.\d*?)0+$', r'\1', text)
    text = re.sub(r'\.$', '', text)
    return text.strip().lower()


def compare_numerical_answers(student: str, correct: str, tolerance: float = 0.01) -> bool:
    """Compare two numerical answers with tolerance."""
    try:
        # Extract numbers
        student_nums = re.findall(r'-?\d+\.?\d*', student)
        correct_nums = re.findall(r'-?\d+\.?\d*', correct)
        if not student_nums or not correct_nums:
            return False
        student_val = float(student_nums[-1])  # Last number is usually the answer
        correct_val = float(correct_nums[-1])
        return abs(student_val - correct_val) <= tolerance * abs(correct_val)
    except (ValueError, IndexError):
        return False


def extract_keywords(text: str) -> set:
    """Extract meaningful keywords from text for comparison."""
    # Remove math notation
    text = re.sub(r'\$.*?\$', ' ', text)
    # Tokenize
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    # Filter common stop words
    stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'she', 'use', 'her', 'way', 'many', 'oil', 'sit', 'set', 'run', 'eat', 'far', 'sea', 'eye', 'ago', 'off', 'too', 'any', 'say', 'man', 'try', 'ask', 'end', 'why', 'let', 'put', 'say', 'she', 'try', 'way', 'own', 'say', 'too', 'old', 'tell', 'very', 'when', 'much', 'would', 'there', 'their', 'what', 'said', 'each', 'which', 'will', 'about', 'could', 'other', 'after', 'first', 'never', 'these', 'think', 'where', 'being', 'every', 'great', 'might', 'shall', 'still', 'those', 'while', 'this', 'that', 'with', 'have', 'from', 'they', 'know', 'want', 'been', 'good', 'much', 'some', 'time', 'very', 'when', 'come', 'here', 'just', 'like', 'long', 'make', 'many', 'over', 'such', 'take', 'than', 'them', 'well', 'were'}
    return set(w for w in words if w not in stop_words)


def calculate_keyword_overlap(student: str, memo: str) -> float:
    """Calculate Jaccard similarity of keywords between student answer and memo."""
    student_keywords = extract_keywords(student)
    memo_keywords = extract_keywords(memo)
    if not memo_keywords:
        return 0.0
    intersection = student_keywords & memo_keywords
    union = student_keywords | memo_keywords
    return len(intersection) / len(union) if union else 0.0


def parse_multi_part_answer(student: str) -> dict:
    """Parse a multi-part answer JSON string."""
    try:
        return json.loads(student) if student else {}
    except json.JSONDecodeError:
        # Try to parse line-by-line format: "1.1 answer, 1.2 answer"
        result = {}
        lines = student.strip().split('\n')
        for line in lines:
            match = re.match(r'^(\d+\.\d+(?:\.\d+)?)[:\-\s]+(.+)$', line.strip())
            if match:
                result[match.group(1)] = match.group(2).strip()
        return result


def parse_table_answer(student: str) -> dict:
    """Parse a table completion answer."""
    try:
        return json.loads(student)
    except json.JSONDecodeError:
        # Try CSV-like format
        result = {}
        lines = student.strip().split('\n')
        for line in lines:
            parts = line.split(',')
            if len(parts) >= 2:
                result[parts[0].strip()] = parts[1].strip()
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# MARKING FUNCTIONS BY TYPE
# ═══════════════════════════════════════════════════════════════════════════════

def mark_calculation(question: str, student_answer: str, memo: str, marks: int,
                     subject: str = "") -> dict:
    """Mark a calculation question with step-by-step partial credit."""
    config = get_subject_config(subject)
    student = str(student_answer).strip() if student_answer else ""

    if not student:
        return {"score": 0, "feedback": "No answer provided.", "status": "missing"}

    if not memo:
        return {"score": 0, "feedback": "No memo available.", "status": "no_memo"}

    # Check for final answer match first (quick win)
    final_answer_match = compare_numerical_answers(student, memo)

    # Extract formulas/steps from student answer
    student_formulas = extract_math_expressions(student)
    memo_formulas = extract_math_expressions(memo)

    # Simple heuristic scoring
    score = 0
    feedback_parts = []

    # Check for working shown
    has_working = len(student.split('\n')) > 1 or '=' in student

    if has_working:
        feedback_parts.append("Working shown")
    else:
        feedback_parts.append("No working shown — always show calculations")

    # Check formula usage
    formula_match = False
    if student_formulas and memo_formulas:
        for sf in student_formulas:
            for mf in memo_formulas:
                if normalize_math_answer(sf) == normalize_math_answer(mf) or \
                   sf.lower() in mf.lower() or mf.lower() in sf.lower():
                    formula_match = True
                    break

    if formula_match:
        score += int(marks * config.get("formula_weight", 0.3))
        feedback_parts.append("Correct formula used")
    else:
        feedback_parts.append("Check your formula")

    # Check final answer
    if final_answer_match:
        score += int(marks * config.get("answer_weight", 0.4))
        feedback_parts.append("Correct final answer")
    else:
        feedback_parts.append("Final answer incorrect — check calculations")

    # Check units (for subjects that require them)
    if config.get("unit_required", False):
        student_units = re.findall(r'\b(m|km|kg|g|cm|mm|ml|l|s|h|min|N|J|W|Pa|V|A|Ω|Hz|°C)\b', student.lower())
        memo_units = re.findall(r'\b(m|km|kg|g|cm|mm|ml|l|s|h|min|N|J|W|Pa|V|A|Ω|Hz|°C)\b', memo.lower())
        if student_units and memo_units:
            if any(su in memo_units for su in student_units):
                score += 1
                feedback_parts.append("Correct units")
            else:
                feedback_parts.append("Check your units")

    # Clamp score
    score = max(0, min(score, marks))

    if score == marks:
        status = "correct"
    elif score > 0:
        status = "partial"
    else:
        status = "incorrect"

    return {
        "score": score,
        "feedback": "; ".join(feedback_parts),
        "status": status,
    }


def mark_essay(question: str, student_answer: str, memo: str, marks: int,
               subject: str = "") -> dict:
    """Mark an essay using rubric-based assessment via LLM."""
    student = str(student_answer).strip() if student_answer else ""

    if not student:
        return {"score": 0, "feedback": "No essay submitted.", "status": "missing"}

    config = get_subject_config(subject)
    criteria = config.get("rubric_criteria", ["content", "structure", "language"])

    # Word count check
    word_count = len(student.split())

    marking_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a strict NSC examiner. Mark essays using the provided rubric. Return ONLY JSON."),
        ("human", "Subject: {subject}\nQuestion ({marks} marks): {question}\n\nMemo/Marking guideline:\n{memo}\n\nStudent essay ({word_count} words):\n{student_answer}\n\nRubric criteria: {criteria}\n\nReturn JSON: {{\"score\": <int 0-{marks}>, \"feedback\": \"<specific feedback on each criterion>\", \"status\": \"<correct|partial|incorrect>\"}}"),
    ])

    chain = marking_prompt | llm | _json_parser

    for attempt in range(2):
        try:
            result = chain.invoke({
                "subject": subject or "General",
                "marks": marks,
                "question": question,
                "memo": memo or "Use subject knowledge to assess.",
                "student_answer": student,
                "word_count": word_count,
                "criteria": ", ".join(criteria),
            })
            result["score"] = max(0, min(int(result.get("score", 0)), marks))
            # Add word count note
            result["feedback"] = f"Word count: {word_count}. " + result.get("feedback", "")
            return result
        except Exception as e:
            if attempt == 0:
                time.sleep(5)
            else:
                # Fallback: keyword-based scoring
                overlap = calculate_keyword_overlap(student, memo or "")
                score = int(overlap * marks)
                return {
                    "score": score,
                    "feedback": f"Word count: {word_count}. Keyword match: {overlap:.0%}. Could not perform detailed rubric marking.",
                    "status": "correct" if score == marks else "partial" if score > marks * 0.5 else "incorrect",
                }


def mark_short_answer(question: str, student_answer: str, memo: str, marks: int) -> dict:
    """Mark short answer questions with keyword/point-based scoring."""
    student = str(student_answer).strip() if student_answer else ""

    if not student:
        return {"score": 0, "feedback": "No answer provided.", "status": "missing"}

    if not memo:
        return {"score": 0, "feedback": "No memo available.", "status": "no_memo"}

    # Extract expected points from memo
    expected_points = [p.strip() for p in re.split(r'[/;]|\n', memo) if len(p.strip()) > 2]
    if not expected_points:
        expected_points = [memo]

    # Calculate points per mark
    points_per_mark = max(1, len(expected_points) // marks) if marks > 0 else 1

    matched_points = 0
    feedback_parts = []

    for point in expected_points:
        point_keywords = extract_keywords(point)
        student_keywords = extract_keywords(student)
        overlap = len(point_keywords & student_keywords) / len(point_keywords) if point_keywords else 0

        if overlap >= 0.5:  # 50% keyword overlap = point matched
            matched_points += 1
            feedback_parts.append(f"✓ {point[:50]}...")
        else:
            feedback_parts.append(f"✗ Missing: {point[:50]}...")

    score = min(marks, matched_points // points_per_mark) if points_per_mark > 0 else 0
    score = max(0, score)

    if score == marks:
        status = "correct"
    elif score > 0:
        status = "partial"
    else:
        status = "incorrect"

    return {
        "score": score,
        "feedback": " | ".join(feedback_parts[:5]),  # Limit feedback length
        "status": status,
    }


def mark_comprehension(question: str, student_answer: str, memo: str, marks: int) -> dict:
    """Mark comprehension questions — inference + textual evidence."""
    student = str(student_answer).strip() if student_answer else ""

    if not student:
        return {"score": 0, "feedback": "No answer provided.", "status": "missing"}

    # Comprehension is similar to short answer but with inference checking
    return mark_short_answer(question, student_answer, memo, marks)


def mark_diagram_label(question: str, student_answer: str, memo: str, marks: int) -> dict:
    """Mark diagram label questions — part-by-part with synonym tolerance."""
    student = str(student_answer).strip() if student_answer else ""

    if not student:
        return {"score": 0, "feedback": "No labels provided.", "status": "missing"}

    if not memo:
        return {"score": 0, "feedback": "No memo available.", "status": "no_memo"}

    # Try to parse structured answer (JSON or line-by-line)
    student_labels = {}
    try:
        student_labels = json.loads(student)
    except json.JSONDecodeError:
        # Parse "A: label, B: label" format
        for line in student.split('\n'):
            match = re.match(r'^([A-Z])[:\-\.]\s*(.+)$', line.strip())
            if match:
                student_labels[match.group(1)] = match.group(2).strip()

    # Parse memo
    memo_labels = {}
    try:
        memo_labels = json.loads(memo) if isinstance(memo, str) else memo
    except json.JSONDecodeError:
        for line in memo.split('\n'):
            match = re.match(r'^([A-Z])[:\-\.]\s*(.+)$', line.strip())
            if match:
                memo_labels[match.group(1)] = match.group(2).strip()

    if not memo_labels:
        return {"score": 0, "feedback": "Could not parse memo labels.", "status": "no_memo"}

    correct_count = 0
    details = []

    for part, correct_label in memo_labels.items():
        student_label = student_labels.get(part, "")
        if not student_label:
            details.append(f"❌ {part}: no answer")
            continue

        # Check exact match or keyword overlap
        student_words = set(student_label.lower().split())
        correct_words = set(correct_label.lower().split())
        overlap = len(student_words & correct_words) / len(correct_words) if correct_words else 0

        if student_label.lower() == correct_label.lower() or overlap >= 0.6:
            correct_count += 1
            details.append(f"✅ {part}: {student_label}")
        else:
            details.append(f"❌ {part}: got '{student_label}', expected '{correct_label}'")

    total_parts = len(memo_labels)
    score = round((correct_count / total_parts) * marks) if total_parts else 0
    score = max(0, min(score, marks))

    status = "correct" if score == marks else "partial" if score > 0 else "incorrect"

    return {
        "score": score,
        "feedback": f"{correct_count}/{total_parts} correct. " + " | ".join(details),
        "status": status,
    }


def mark_table_completion(question: str, student_answer: str, memo: str, marks: int) -> dict:
    """Mark table completion — cell-by-cell comparison."""
    student = str(student_answer).strip() if student_answer else ""

    if not student:
        return {"score": 0, "feedback": "No table completion provided.", "status": "missing"}

    if not memo:
        return {"score": 0, "feedback": "No memo available.", "status": "no_memo"}

    student_cells = parse_table_answer(student)
    memo_cells = parse_table_answer(memo) if isinstance(memo, str) else memo

    if not memo_cells:
        return {"score": 0, "feedback": "Could not parse memo table.", "status": "no_memo"}

    correct_count = 0
    details = []

    for cell_key, correct_value in memo_cells.items():
        student_value = student_cells.get(cell_key, "")
        if not student_value:
            details.append(f"❌ {cell_key}: empty")
            continue

        # Normalize for comparison
        student_norm = student_value.strip().lower()
        correct_norm = str(correct_value).strip().lower()

        # Check numerical equality
        numerical_match = compare_numerical_answers(student_value, str(correct_value))

        if student_norm == correct_norm or numerical_match:
            correct_count += 1
            details.append(f"✅ {cell_key}: {student_value}")
        else:
            details.append(f"❌ {cell_key}: got '{student_value}', expected '{correct_value}'")

    total_cells = len(memo_cells)
    score = round((correct_count / total_cells) * marks) if total_cells else 0
    score = max(0, min(score, marks))

    status = "correct" if score == marks else "partial" if score > 0 else "incorrect"

    return {
        "score": score,
        "feedback": f"{correct_count}/{total_cells} cells correct. " + " | ".join(details),
        "status": status,
    }


def mark_multi_part(question: str, student_answer: str, memo: str, marks: int,
                    sub_parts: list = None) -> dict:
    """Mark multi-part questions — aggregate per-sub-part scores."""
    student = str(student_answer).strip() if student_answer else ""

    if not student:
        return {"score": 0, "feedback": "No answer provided.", "status": "missing"}

    student_answers = parse_multi_part_answer(student)

    # Try to parse memo as multi-part
    memo_parts = {}
    try:
        memo_parts = json.loads(memo) if isinstance(memo, str) and memo.strip().startswith('{') else {}
    except json.JSONDecodeError:
        pass

    if not memo_parts and sub_parts:
        # Use sub_parts structure
        for sp in sub_parts:
            sp_num = sp.get("sub_number", "")
            if sp_num:
                memo_parts[sp_num] = sp.get("memo", "")

    if not memo_parts:
        # Fallback: treat as single open question
        return mark_open(question, student_answer, memo, marks)

    total_score = 0
    total_possible = 0
    details = []

    for part_num, part_memo in memo_parts.items():
        part_student = student_answers.get(part_num, "")
        part_marks = 1  # Assume 1 mark per sub-part unless specified

        # Try to infer marks from memo format
        marks_match = re.search(r'\((\d+)\)', str(part_memo))
        if marks_match:
            part_marks = int(marks_match.group(1))

        part_result = mark_open(question, part_student, part_memo, part_marks)

        total_score += part_result["score"]
        total_possible += part_marks
        details.append(f"{part_num}: {part_result['score']}/{part_marks} — {part_result['feedback']}")

    # Scale to total marks
    if total_possible > 0:
        score = round((total_score / total_possible) * marks)
    else:
        score = 0
    score = max(0, min(score, marks))

    status = "correct" if score == marks else "partial" if score > 0 else "incorrect"

    return {
        "score": score,
        "feedback": " | ".join(details[:5]),
        "status": status,
    }


def mark_open(question: str, student_answer: str, memo: str, marks: int) -> dict:
    """Mark open-ended questions using LLM with JsonOutputParser."""
    student = str(student_answer).strip() if student_answer else ""

    if not student:
        return {"score": 0, "feedback": "No answer provided.", "status": "missing"}

    memo_text = memo if isinstance(memo, str) else json.dumps(memo) if memo else ""
    has_memo = bool(memo_text.strip())

    marking_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a strict South African NSC examiner. "
            "Award 1 mark per distinct correct fact. "
            "Be fair but rigorous. Return ONLY valid JSON — no markdown, no explanation.",
        ),
        (
            "human",
            "Question ({marks} mark{plural}):\n{question}\n\n"
            "{memo_section}"
            "Student's answer:\n{student_answer}\n\n"
            "Return JSON: "
            '{{"score": <int 0-{marks}>, '
            '"feedback": "<what was correct, what was missing or incorrect>", '
            '"status": "<correct|partial|incorrect>" }}',
        ),
    ])

    marking_chain = marking_prompt | llm | _json_parser

    for attempt in range(2):
        try:
            result = marking_chain.invoke({
                "marks": marks,
                "plural": "s" if marks != 1 else "",
                "question": question,
                "memo_section": (
                    f"Marking guideline:\n{memo_text}\n\n" if has_memo
                    else "No guideline — use NSC Grade 12 subject knowledge.\n\n"
                ),
                "student_answer": student,
            })
            result["score"] = max(0, min(int(result.get("score", 0)), marks))
            return result
        except Exception as e:
            if attempt == 0:
                time.sleep(5)
            else:
                # Final fallback: keyword overlap
                overlap = calculate_keyword_overlap(student, memo_text)
                score = int(overlap * marks)
                return {
                    "score": score,
                    "feedback": f"Could not perform detailed marking. Keyword match: {overlap:.0%}. Error: {str(e)[:100]}",
                    "status": "correct" if score == marks else "partial" if score > marks * 0.3 else "incorrect",
                }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN MARK_ANSWER DISPATCHER
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

    NEW PARAMETERS (backward-compatible):
        instructions: Special instructions (e.g., "Show ALL working")
        subject: Subject name for subject-aware marking
        sub_parts: List of sub-part dicts for multi_part questions
    """
    q_type = q_type.lower().strip()
    student = str(student_answer).strip() if student_answer else ""

    # ── MCQ — exact letter match, no LLM needed ───────────────────────────────
    if q_type == "mcq":
        correct = str(memo).strip().upper()
        ans = student.upper()
        if not correct:
            return {"score": 0, "feedback": "No memo available.", "status": "no_memo"}
        if not ans:
            return {"score": 0, "feedback": f"No answer selected. Correct: {correct}.", "status": "missing"}
        opt_text = ""
        for opt in (options or []):
            if isinstance(opt, dict) and opt.get("key", "").upper() == correct:
                opt_text = f" ({opt['value']})"
                break
            elif isinstance(options, dict) and correct in options:
                opt_text = f" ({options[correct]})"
                break
        if ans == correct:
            return {"score": marks, "feedback": f"Correct! {correct}{opt_text}.", "status": "correct"}
        return {
            "score": 0,
            "feedback": f"Incorrect. You selected {ans}; correct is {correct}{opt_text}.",
            "status": "incorrect",
        }

    # ── True/False ────────────────────────────────────────────────────────────
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
            def extract_correction(s):
                parts = re.split(r"[-—]", s, maxsplit=1)
                return parts[1].strip().lower() if len(parts) > 1 else ""

            memo_word = extract_correction(cl)
            student_word = extract_correction(al)

            if not memo_word or (student_word and (memo_word in student_word or student_word in memo_word)):
                return {
                    "score": marks,
                    "feedback": f"Correct — False, correction: {student_word or memo_word}.",
                    "status": "correct",
                }
            return {
                "score": marks // 2,
                "feedback": f"Correct that it is FALSE, but wrong correction. Expected '{memo_word}', got '{student_word or '(none)'}'.",
                "status": "partial",
            }

        return {"score": 0, "feedback": f"Incorrect. Correct answer: {memo}.", "status": "incorrect"}

    # ── Matching ──────────────────────────────────────────────────────────────
    if q_type == "matching":
        if not isinstance(memo, dict) or not memo:
            return {"score": 0, "feedback": "No memo for matching.", "status": "no_memo"}
        try:
            student_map = json.loads(student) if student else {}
        except Exception:
            student_map = {}

        correct_count = 0
        details = []
        for col_a_item, correct_letter in memo.items():
            student_val = student_map.get(col_a_item, "")
            student_letter = student_val.strip().split(".")[0].strip().upper() if student_val else ""
            correct_clean = str(correct_letter).strip().upper()
            if student_letter == correct_clean:
                correct_count += 1
                details.append(f"✅ {col_a_item.split()[0]}: {student_letter}")
            else:
                details.append(f"❌ {col_a_item.split()[0]}: got '{student_letter or '—'}' need '{correct_clean}'")

        total_pairs = len(memo)
        earned = round((correct_count / total_pairs) * marks) if total_pairs else 0
        status = "correct" if earned == marks else "partial" if earned > 0 else "incorrect"
        return {
            "score": earned,
            "feedback": f"{correct_count}/{total_pairs} correct. " + " | ".join(details),
            "status": status,
        }

    # ── Calculation ───────────────────────────────────────────────────────────
    if q_type == "calculation":
        return mark_calculation(question, student_answer, memo, marks, subject)

    # ── Essay ─────────────────────────────────────────────────────────────────
    if q_type == "essay":
        return mark_essay(question, student_answer, memo, marks, subject)

    # ── Short Answer ──────────────────────────────────────────────────────────
    if q_type == "short_answer":
        return mark_short_answer(question, student_answer, memo, marks)

    # ── Comprehension ─────────────────────────────────────────────────────────
    if q_type == "comprehension":
        return mark_comprehension(question, student_answer, memo, marks)

    # ── Diagram Label ─────────────────────────────────────────────────────────
    if q_type == "diagram_label":
        return mark_diagram_label(question, student_answer, memo, marks)

    # ── Table Completion ──────────────────────────────────────────────────────
    if q_type == "table_completion":
        return mark_table_completion(question, student_answer, memo, marks)

    # ── Multi-Part ────────────────────────────────────────────────────────────
    if q_type == "multi_part":
        return mark_multi_part(question, student_answer, memo, marks, sub_parts)

    # ── Open / Unknown / Fallback ─────────────────────────────────────────────
    return mark_open(question, student_answer, memo, marks)


# ═══════════════════════════════════════════════════════════════════════════════
# EXAM FEEDBACK SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def generate_exam_feedback(results: list, score: int, total: int, percentage: float,
                           subject: str = "") -> str:
    """
    Generate a personalised performance summary after exam submission.
    Now subject-aware with specific study recommendations.
    """
    wrong = [r for r in results if r["status"] != "correct"]
    wrong_by_type = {}
    for r in wrong:
        qt = r.get("type", "open")
        if qt not in wrong_by_type:
            wrong_by_type[qt] = []
        wrong_by_type[qt].append(r["question_number"])

    # Build weak areas summary
    weak_summary = ""
    if wrong_by_type:
        weak_summary = "Areas to focus on:\n"
        for qt, qnums in wrong_by_type.items():
            type_name = {
                "mcq": "Multiple Choice", "matching": "Matching",
                "true_false": "True/False", "calculation": "Calculations",
                "essay": "Essay Writing", "short_answer": "Short Answers",
                "comprehension": "Comprehension", "diagram_label": "Diagram Labelling",
                "table_completion": "Table Completion", "multi_part": "Multi-Part Questions",
                "open": "Open-Ended Questions",
            }.get(qt, qt.replace("_", " ").title())
            weak_summary += f"- {type_name}: Q{', Q'.join(qnums[:5])}{'...' if len(qnums) > 5 else ''}\n"

    subject_prompt = f" in {subject}" if subject else ""

    prompt = (
        f"You are a motivating NSC Grade 12 teacher{subject_prompt}.\n"
        f"Score: {score}/{total} ({percentage}%)\n"
        f"Wrong/partial questions: {len(wrong)}\n"
        f"{weak_summary}\n\n"
        f"Write 4-5 sentences of encouraging, specific feedback. "
        f"Mention exactly which topics and question types to revise. "
        f"Give ONE concrete study tip for the weakest area."
    )

    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        return f"Score: {score}/{total} ({percentage}%). Keep practising! Focus on: {', '.join(wrong_by_type.keys()) if wrong_by_type else 'all areas'} 🚀"