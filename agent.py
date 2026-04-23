import os
from typing import TypedDict, Optional

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

import memory as mem
from model import mark_answer

load_dotenv()

# ── LLM ─────────────────────────────────────
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    max_tokens=1024,
    groq_api_key=os.getenv("GROQ_API_KEY"),
)

# ── STATE ───────────────────────────────────
class AgentState(TypedDict):
    student_id: str
    input: str
    response: Optional[str]
    intent: Optional[str]
    context: Optional[str]

# ── RAG ─────────────────────────────────────
_rag = None

def set_rag(rag):
    global _rag
    _rag = rag

# ────────────────────────────────────────────
# 🧭 INTENT ROUTER (VERY IMPORTANT)
# ────────────────────────────────────────────
def classify_intent(state: AgentState):
    prompt = f"""
    Classify the student's request into ONE of:
    - theory
    - weak_topics
    - study_plan
    - hint
    - marking
    - general

    Message: {state['input']}
    """

    result = llm.invoke(prompt).content.lower()

    return {
        **state,
        "intent": result.strip()
    }

# ────────────────────────────────────────────
# 📚 THEORY NODE
# ────────────────────────────────────────────
def handle_theory(state: AgentState):
    if _rag is None:
        return {**state, "response": "Theory search unavailable."}

    chunks = _rag.search(state["input"])
    context = " ".join(str(c) for c in chunks)[:1500]

    answer = llm.invoke(f"""
    Use this context to answer clearly:
    {context}

    Question:
    {state['input']}
    """).content

    return {**state, "response": answer}

# ────────────────────────────────────────────
# 📉 WEAK TOPICS NODE
# ────────────────────────────────────────────
def handle_weak_topics(state: AgentState):
    weak = mem.get_weak_topics(state["student_id"])

    if not weak:
        return {**state, "response": "No weak topics yet."}

    text = "\n".join(
        f"Q{w['question_number']} ({w['q_type']}) - wrong {w['wrong_count']}x"
        for w in weak
    )

    return {**state, "response": text}

# ────────────────────────────────────────────
# 📊 STUDY PLAN NODE
# ────────────────────────────────────────────
def handle_study_plan(state: AgentState):
    weak = mem.get_weak_topics(state["student_id"])
    sessions = mem.get_sessions(state["student_id"])

    plan_prompt = f"""
    Create a personalised CAT study plan.

    Weak topics:
    {weak}

    Past sessions:
    {sessions}
    """

    plan = llm.invoke(plan_prompt).content

    mem.save_study_plan(state["student_id"], plan)

    return {**state, "response": plan}

# ────────────────────────────────────────────
# 💡 HINT NODE
# ────────────────────────────────────────────
def handle_hint(state: AgentState):
    hint = llm.invoke(f"""
    Give a Socratic hint (DO NOT GIVE ANSWER):

    {state['input']}
    """).content

    return {**state, "response": hint}

# ────────────────────────────────────────────
# 📝 MARKING NODE
# ────────────────────────────────────────────
def handle_marking(state: AgentState):
    # You can expand parsing logic here
    result = mark_answer(
        question="",
        question_number="",
        q_type="open",
        student_answer=state["input"],
        memo="",
        marks=5,
    )

    return {**state, "response": str(result)}

# ────────────────────────────────────────────
# 💬 GENERAL CHAT NODE
# ────────────────────────────────────────────
def handle_general(state: AgentState):
    response = llm.invoke(state["input"]).content
    return {**state, "response": response}

# ────────────────────────────────────────────
# 🔀 ROUTING LOGIC
# ────────────────────────────────────────────
def route(state: AgentState):
    intent = state["intent"]

    if "theory" in intent:
        return "theory"
    if "weak" in intent:
        return "weak"
    if "study" in intent:
        return "study"
    if "hint" in intent:
        return "hint"
    if "mark" in intent:
        return "mark"

    return "general"

# ────────────────────────────────────────────
# 🏗 BUILD GRAPH
# ────────────────────────────────────────────
builder = StateGraph(AgentState)

builder.add_node("intent", classify_intent)
builder.add_node("theory", handle_theory)
builder.add_node("weak", handle_weak_topics)
builder.add_node("study", handle_study_plan)
builder.add_node("hint", handle_hint)
builder.add_node("mark", handle_marking)
builder.add_node("general", handle_general)

builder.set_entry_point("intent")

builder.add_conditional_edges("intent", route, {
    "theory": "theory",
    "weak": "weak",
    "study": "study",
    "hint": "hint",
    "mark": "mark",
    "general": "general",
})

for node in ["theory", "weak", "study", "hint", "mark", "general"]:
    builder.add_edge(node, END)

graph = builder.compile()

# ────────────────────────────────────────────
# 🚀 MAIN FUNCTION
# ────────────────────────────────────────────
def run_agent(student_id: str, message: str, rag=None):
    global _rag
    if rag:
        _rag = rag

    mem.ensure_student(student_id)

    result = graph.invoke({
        "student_id": student_id,
        "input": message
    })

    return result["response"]