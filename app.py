"""
app.py — EduCAT Flask API (Universal Exam Pipeline — All Subjects)

Supports ALL NSC subjects with enhanced question rendering:
- Mathematical formulas via MathJax
- Diagram placeholders
- Table rendering
- Calculation working areas
- Essay word counts
- Comprehension passages
- Multi-part questions
"""

from dotenv import load_dotenv
load_dotenv()

import os
import io
import json
import uuid
import re
import traceback
import requests as http_requests

from flask import Flask, request, jsonify
from flask_cors import CORS

# Firebase Admin
import firebase_admin
from firebase_admin import credentials, firestore as fs_admin

def _init_firebase():
    if firebase_admin._apps:
        return
    inline_json = os.getenv("FIREBASE_SERVICE_ACCOUNT")
    if inline_json and inline_json.strip():
        try:
            sa_dict = json.loads(inline_json)
            if sa_dict.get("type") != "service_account":
                raise ValueError(
                    'FIREBASE_SERVICE_ACCOUNT must contain "type": "service_account".'
                )
            cred = credentials.Certificate(sa_dict)
            print("[Firebase] Using inline JSON credentials")
        except json.JSONDecodeError as e:
            raise ValueError(f"FIREBASE_SERVICE_ACCOUNT is not valid JSON: {e}")
    else:
        cred = credentials.ApplicationDefault()
        print("[Firebase] Using Application Default Credentials")
    firebase_admin.initialize_app(cred)

_init_firebase()
db_admin = fs_admin.client()

from model import generate_answer, mark_answer, generate_exam_feedback
from rag import RAGIndex
import memory as mem
import agent
from agent import run_agent

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": [
    "http://localhost:3000",
    "http://localhost:5176",
    "http://localhost:5175",
    "http://localhost:5174"
    "https://edu-cat.netlify.app",
]}})

rag = RAGIndex()
agent.set_rag(rag)

EXAMS_FOLDER = "exams"
sessions = {}
SERVICE_ACCOUNT_EMAIL = os.getenv("SERVICE_ACCOUNT_EMAIL", "")


# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL QUESTION RENDERING
# ═══════════════════════════════════════════════════════════════════════════════

QUESTION_TYPE_LABELS = {
    "mcq": "Multiple Choice",
    "matching": "Matching Items",
    "true_false": "True / False",
    "open": "Open-Ended",
    "calculation": "Calculation",
    "short_answer": "Short Answer",
    "essay": "Essay",
    "comprehension": "Comprehension",
    "diagram_label": "Diagram Label",
    "table_completion": "Table Completion",
    "multi_part": "Multi-Part",
    "unknown": "Question"
}


def render_math_formulas(text):
    if not text:
        return text
    text = re.sub(r'\$\$(.*?)\$\$', r'<div class="math-display">\\[\1\\]</div>', text, flags=re.DOTALL)
    text = re.sub(r'\$(.+?)\$', r'<span class="math-inline">\\(\1\\)</span>', text)
    return text


def render_diagram_refs(diagram_refs):
    if not diagram_refs:
        return ""
    html = '<div class="diagram-container">'
    for ref in diagram_refs:
        html += '<div class="diagram-box">'
        html += '<div class="diagram-icon">&#128202;</div>'
        html += '<div class="diagram-caption">' + ref.get("diagram_id", "Figure") + '</div>'
        html += '<div class="diagram-desc">' + ref.get("caption", "") + '</div>'
        html += '<div class="diagram-note">' + ref.get("description", "") + '</div>'
        html += '</div>'
    html += '</div>'
    return html


def render_table_refs(table_refs):
    if not table_refs:
        return ""
    html = ""
    for ref in table_refs:
        headers = ref.get("headers", [])
        rows = ref.get("rows", [])
        caption = ref.get("caption", "")
        html += '<div class="table-container">'
        html += '<div class="table-caption">' + caption + '</div>'
        html += '<table class="exam-table">'
        if headers:
            html += '<thead><tr>' + ''.join('<th>' + h + '</th>' for h in headers) + '</tr></thead>'
        html += '<tbody>'
        for row in rows:
            html += '<tr>' + ''.join('<td>' + cell + '</td>' for cell in row) + '</tr>'
        html += '</tbody></table></div>'
    return html


def render_question_input(q, saved_answer=""):
    q_type = q.get("type", "open").lower()

    if q_type == "mcq":
        options = q.get("options", {})
        if isinstance(options, list):
            options = {chr(65+i): str(v) for i, v in enumerate(options) if str(v).strip()}
        if not options:
            return '<textarea id="openAnswerBox" placeholder="Your answer...">' + saved_answer + '</textarea>'
        html = '<div id="mcqOptions" style="margin-top:12px">'
        for key, value in sorted(options.items()):
            chk = "checked" if saved_answer == key else ""
            html += '<label class="option-label">'
            html += '<input type="radio" name="mcq_answer" value="' + key + '" ' + chk + '> '
            html += '<b>' + key + '.</b> ' + value
            html += '</label>'
        html += '</div>'
        return html

    elif q_type == "true_false":
        is_f = saved_answer.startswith("False")
        corr = saved_answer.split("—", 1)[1].strip() if is_f and "—" in saved_answer else ""
        html = '<div id="tfOptions" style="margin-top:12px">'
        html += '<label class="tf-label"><input type="radio" name="tf_answer" value="True" ' + ("checked" if saved_answer=="True" else "") + '> &#9989; True</label>'
        html += '<label class="tf-label"><input type="radio" name="tf_answer" value="False" ' + ("checked" if is_f else "") + '> &#10060; False</label>'
        html += '</div>'
        html += '<div id="tfCorrection" style="margin-top:10px;' + ("" if is_f else "display:none") + '">'
        html += '<label style="font-size:12px;color:#555">If FALSE — corrected word/phrase:</label>'
        html += '<input type="text" id="tfCorrectionBox" placeholder="e.g. secondary memory" value="' + corr + '" style="width:100%;margin-top:4px;padding:8px;border:1px solid #ddd;border-radius:6px;font-size:13px">'
        html += '</div>'
        return html

    elif q_type == "matching":
        col_a = q.get("column_a", [])
        col_b = q.get("column_b", [])
        if not col_a or not col_b:
            return '<textarea id="openAnswerBox" placeholder="Your answer...">' + saved_answer + '</textarea>'
        saved = {}
        try:
            if saved_answer:
                saved = json.loads(saved_answer)
        except:
            pass
        html = '<p style="margin-top:10px;font-size:12px;color:#555"><i>Match each COLUMN A item to COLUMN B.</i></p>'
        html += '<table class="match-table"><thead><tr><th style="width:55%">COLUMN A</th><th>COLUMN B</th></tr></thead><tbody>'
        for item in col_a:
            sv = saved.get(item, "")
            html += '<tr><td>' + item + '</td><td><select class="match-select" data-item="' + item + '">'
            html += '<option value="">-- Select --</option>'
            for b in col_b:
                sel = "selected" if sv == b else ""
                html += '<option value="' + b + '" ' + sel + '>' + b + '</option>'
            html += '</select></td></tr>'
        html += '</tbody></table>'
        return html

    elif q_type == "calculation":
        instructions = q.get("instructions", "Show ALL calculations.")
        html = '<div class="calculation-box">'
        html += '<div class="calc-instructions">&#128221; ' + instructions + '</div>'
        html += '<textarea id="openAnswerBox" placeholder="Show your working here...">' + saved_answer + '</textarea>'
        html += '</div>'
        return html

    elif q_type == "essay":
        instructions = q.get("instructions", "Write a well-structured response.")
        html = '<div class="essay-box">'
        html += '<div class="essay-instructions">&#128221; ' + instructions + '</div>'
        html += '<textarea id="openAnswerBox" placeholder="Write your essay here..." style="min-height:200px">' + saved_answer + '</textarea>'
        html += '<div class="essay-word-count" id="wordCount">0 words</div>'
        html += '</div>'
        return html

    elif q_type == "diagram_label":
        diagram_refs = q.get("diagram_refs", [])
        html = render_diagram_refs(diagram_refs)
        html += '<textarea id="openAnswerBox" placeholder="Label the diagram parts...">' + saved_answer + '</textarea>'
        return html

    elif q_type == "table_completion":
        table_refs = q.get("table_refs", [])
        html = render_table_refs(table_refs)
        html += '<textarea id="openAnswerBox" placeholder="Complete the table...">' + saved_answer + '</textarea>'
        return html

    elif q_type == "comprehension":
        parent_context = q.get("parent_context", "")
        html = ""
        if parent_context:
            html += '<div class="comprehension-passage">' + render_math_formulas(parent_context) + '</div>'
        html += '<textarea id="openAnswerBox" placeholder="Refer to the passage and answer...">' + saved_answer + '</textarea>'
        return html

    elif q_type == "multi_part":
        sub_parts = q.get("sub_parts", [])
        html = '<div class="multi-part-container">'
        for i, sub in enumerate(sub_parts):
            sub_num = sub.get("sub_number", str(q.get("question_number", "")) + "." + str(i+1))
            sub_q = sub.get("question", "")
            sub_saved = saved_answer.get(sub_num, "") if isinstance(saved_answer, dict) else ""
            html += '<div class="sub-part">'
            html += '<div class="sub-part-label">' + sub_num + '</div>'
            html += '<div class="sub-part-text">' + sub_q + '</div>'
            html += '<input type="text" class="sub-part-answer" data-sub="' + sub_num + '" value="' + sub_saved + '" placeholder="Answer...">'
            html += '</div>'
        html += '</div>'
        return html

    else:
        return '<textarea id="openAnswerBox" placeholder="Write your answer here...">' + saved_answer + '</textarea>'


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def load_exam_local(exam_name):
    path = os.path.join(EXAMS_FOLDER, exam_name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_exam_from_firestore(exam_id):
    try:
        doc = db_admin.collection("exams").document(exam_id).get()
        if not doc.exists:
            return None, []
        meta = doc.to_dict()
        meta["id"] = doc.id
        q_docs = db_admin.collection("exam_questions").where("examId", "==", exam_id).order_by("questionNumber").stream()
        questions = []
        for q in q_docs:
            d = q.to_dict()
            questions.append({
                "id": d.get("questionNumber", q.id),
                "question_number": str(d.get("questionNumber", "")),
                "parent_question": d.get("parentQuestion", ""),
                "parent_context": d.get("parentContext"),
                "section": d.get("section", "A"),
                "section_title": d.get("sectionTitle", ""),
                "section_instructions": d.get("sectionInstructions", ""),
                "section_total_marks": d.get("sectionTotalMarks"),
                "question": d.get("questionText", "Question text missing"),
                "type": d.get("type", "open").lower(),
                "options": d.get("options"),
                "column_a": d.get("columnA"),
                "column_b": d.get("columnB"),
                "diagram_refs": d.get("diagramRefs", []),
                "table_refs": d.get("tableRefs", []),
                "formula": d.get("formula"),
                "instructions": d.get("instructions"),
                "marks": d.get("marks", 1),
                "memo": d.get("memo", ""),
                "saved_answer": "",
            })
        return meta, questions
    except Exception as e:
        traceback.print_exc()
        return None, []


def download_from_drive_bytes(file_id):
    try:
        from google.oauth2 import service_account
        from google.auth.transport.requests import Request as GoogleRequest
        inline_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
        if inline_json:
            sa_info = json.loads(inline_json)
            creds = service_account.Credentials.from_service_account_info(
                sa_info, scopes=["https://www.googleapis.com/auth/drive.readonly"]
            )
        else:
            sa_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "serviceAccountKey.json")
            creds = service_account.Credentials.from_service_account_file(
                sa_path, scopes=["https://www.googleapis.com/auth/drive.readonly"]
            )
        creds.refresh(GoogleRequest())
        token = creds.token
        url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"
        res = http_requests.get(url, headers={"Authorization": f"Bearer {token}"})
        if res.status_code == 200:
            return res.content
        print(f"[Drive] download failed: {res.status_code} {res.text[:200]}")
        return None
    except Exception as e:
        traceback.print_exc()
        return None


def flatten_exam(exam):
    flat = []
    sections = exam.get("sections", [])
    metadata = exam.get("metadata", {})
    if not sections and "questions" in exam:
        sections = [{"section": "A", "section_title": None, "section_instructions": None, "total_marks": None, "questions": exam["questions"]}]
    for section in sections:
        sec_label = section.get("section", "")
        sec_title = section.get("section_title") or ""
        sec_instructions = section.get("section_instructions") or ""
        sec_marks = section.get("total_marks")
        for q in section.get("questions", []):
            q_type = q.get("type", "open").lower()
            q_text = q.get("question", "").strip()
            marks = q.get("marks", 1)
            memo = q.get("memo", "")
            q_id = q.get("id")
            question_number = q.get("question_number", f"Q{q_id}" if q_id else "")
            parent_question = q.get("parent_question", "")
            parent_context = q.get("parent_context")
            options = None
            column_a = column_b = None
            diagram_refs = q.get("diagram_refs", [])
            table_refs = q.get("table_refs", [])
            formula = q.get("formula")
            instructions = q.get("instructions")
            sub_parts = q.get("sub_parts", [])
            if q_type == "mcq":
                raw_opts = q.get("options")
                if isinstance(raw_opts, dict):
                    options = {k: str(v).strip() for k, v in raw_opts.items() if str(v).strip()}
                elif isinstance(raw_opts, list):
                    options = {chr(65+i): str(v).strip() for i, v in enumerate(raw_opts) if str(v).strip()}
            if q_type == "matching":
                column_a = q.get("column_a", [])
                column_b = q.get("column_b", [])
            flat.append({
                "id": q_id, "question_number": question_number,
                "parent_question": parent_question, "parent_context": parent_context,
                "section": sec_label, "section_title": sec_title,
                "section_instructions": sec_instructions, "section_total_marks": sec_marks,
                "question": q_text or "Question text missing",
                "type": q_type, "options": options,
                "column_a": column_a, "column_b": column_b,
                "diagram_refs": diagram_refs, "table_refs": table_refs,
                "formula": formula, "instructions": instructions,
                "sub_parts": sub_parts,
                "marks": marks, "memo": memo, "saved_answer": "",
            })
    return flat

def extract_questions_from_pdf(pdf_bytes, exam_meta):
    """Use Groq to parse PDF bytes into structured question JSON."""
    import base64
    import pdfplumber
    import io

    # Extract text from PDF bytes using pdfplumber (already in your requirements)
    text = ""
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                text += page_text + "\n"

    if not text.strip():
        raise ValueError("No text could be extracted from PDF")

    # Trim to avoid Groq token limits (keep first 12000 chars ~ 20 pages)
    text = text[:12000]

    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    prompt = f"""You are an expert exam parser for South African NSC (CAPS) exams.
Extract ALL questions from this exam text into structured JSON.

Return ONLY a valid JSON array. No extra text, no markdown fences.
Each object must have:
- question_number (string e.g. "1.1", "2", "3.2.1")
- parent_question (string, section heading if any, else "")
- section (string e.g. "A", "B", "1")
- question (string, full question text)
- type (one of: mcq, true_false, matching, short_answer, calculation, essay, open)
- marks (integer)
- options (object with A/B/C/D keys, ONLY for mcq, else null)
- column_a (array, ONLY for matching, else null)
- column_b (array, ONLY for matching, else null)
- memo (string if answer visible in text, else null)

Subject: {exam_meta.get('subject', '')}
Grade: {exam_meta.get('grade', '')}

EXAM TEXT:
{text}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",   # free, handles long context well
        messages=[{"role": "user", "content": prompt}],
        max_tokens=8000,
        temperature=0.1,
    )

    raw = response.choices[0].message.content.strip()
    # Strip markdown fences if Groq adds them
    raw = re.sub(r"^```json\s*|^```\s*|```$", "", raw, flags=re.MULTILINE).strip()
    return json.loads(raw)


# ═══════════════════════════════════════════════════════════════════════════════
# HOME UI (with MathJax + Universal Styles)
# ═══════════════════════════════════════════════════════════════════════════════
# Use the 'r' prefix to prevent Python from eating your backslashes!
HOME_HTML = r"""<!DOCTYPE html>
<html>
<head>
<title>EduCAT — AI Tutor & Exam Mocker</title>
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<script>
window.MathJax = {
  tex: { inlineMath: [['$', '$'], ['\\(', '\\)']], displayMath: [['$$', '$$'], ['\\[', '\\]']] },
  svg: { fontCache: 'global' }
};
</script>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:Arial,sans-serif;background:#f4f6f9;color:#2c3e50}
  .app{display:flex;height:100vh}
  .sidebar{width:240px;background:#2c3e50;color:#ecf0f1;padding:20px;display:flex;flex-direction:column;gap:12px;flex-shrink:0}
  .sidebar h2{font-size:18px;font-weight:700;margin-bottom:8px}
  .sidebar .student-id{font-size:11px;opacity:.5;word-break:break-all}
  .nav-btn{padding:10px 14px;background:rgba(255,255,255,.08);border:none;color:#ecf0f1;border-radius:8px;cursor:pointer;text-align:left;font-size:13px;width:100%}
  .nav-btn:hover,.nav-btn.active{background:rgba(255,255,255,.18)}
  .sep{height:1px;background:rgba(255,255,255,.1);margin:4px 0}
  .main{flex:1;display:flex;flex-direction:column;overflow:hidden}
  .panel{flex:1;display:none;flex-direction:column;overflow:hidden}
  .panel.active{display:flex}
  .chat-messages{flex:1;overflow-y:auto;padding:20px;display:flex;flex-direction:column;gap:12px}
  .msg{max-width:72%;padding:12px 16px;border-radius:12px;font-size:14px;line-height:1.6}
  .msg.user{background:#3498db;color:#fff;align-self:flex-end;border-bottom-right-radius:4px}
  .msg.agent{background:#fff;border:1px solid #e0e0e0;align-self:flex-start;border-bottom-left-radius:4px}
  .msg.agent.thinking{opacity:.6;font-style:italic}
  .chat-input-row{padding:16px 20px;background:#fff;border-top:1px solid #e0e0e0;display:flex;gap:10px}
  .chat-input-row input{flex:1;padding:10px 14px;border:1px solid #ddd;border-radius:8px;font-size:14px}
  .chat-input-row button{padding:10px 18px;background:#3498db;color:#fff;border:none;border-radius:8px;cursor:pointer;font-size:14px}
  .exam-setup{padding:24px;background:#fff;border-bottom:1px solid #e0e0e0}
  .exam-setup h3{margin-bottom:12px}
  .exam-setup select,.exam-setup button{padding:9px 14px;border-radius:7px;border:1px solid #ddd;font-size:13px;margin-right:8px}
  .exam-setup button{background:#27ae60;color:#fff;border-color:#27ae60;cursor:pointer}
  .memo-status{margin-top:8px;font-size:12px;color:#888}
  .exam-area{flex:1;overflow-y:auto;padding:24px}
  .sec-header{background:#eaf0fb;padding:10px 14px;border-radius:8px;margin-bottom:14px;border-left:4px solid #3498db}
  .sec-label{font-weight:bold;font-size:15px}
  .q-row{display:flex;gap:10px;margin-top:10px}
  .q-num{font-weight:bold;min-width:30px}
  .q-mark{color:#e74c3c;font-weight:bold}
  .option-label{display:block;margin:8px 0;padding:10px;border:1px solid #ddd;border-radius:6px;cursor:pointer}
  .nav-bar{margin-top:20px;display:flex;gap:10px}
  .nav-bar button{padding:10px 20px;border:none;border-radius:6px;cursor:pointer;background:#3498db;color:#fff}
  .submit-btn{background:#e74c3c !important;margin-left:auto}
  .results-area{padding:24px;overflow-y:auto}
  .score-banner{text-align:center;padding:20px;background:#fff;border-radius:10px;margin-bottom:20px}
  .result-card{background:#fff;padding:15px;border-radius:8px;margin-bottom:10px;border:1px solid #eee}
  .dash-card{background:#fff;padding:20px;border-radius:10px;margin-bottom:15px;border:1px solid #e0e0e0}
</style>
</head>
<body>
<div class="app">
  <div class="sidebar">
    <h2>&#127891; EduCAT</h2>
    <div class="student-id" id="sidDisplay"></div>
    <div class="sep"></div>
    <button class="nav-btn active" onclick="showPanel('chat',event)">&#128172; AI Tutor</button>
    <button class="nav-btn" onclick="showPanel('exam',event)">&#128221; Exam Mocker</button>
    <button class="nav-btn" onclick="showPanel('dashboard',event);loadDashboard()">&#128202; My Dashboard</button>
    <div class="sep"></div>
    <button class="nav-btn" onclick="clearHistory()">&#128465; Clear chat</button>
  </div>
  <div class="main">
    <div class="panel active" id="panel-chat">
      <div class="chat-messages" id="chatMessages"></div>
      <div class="chat-input-row">
        <input id="chatInput" placeholder="Ask anything..." onkeydown="if(event.key==='Enter')sendChat()">
        <button onclick="sendChat()">Send</button>
      </div>
    </div>
    <div class="panel" id="panel-exam">
      <div class="exam-setup">
        <h3>&#128221; Exam Mocker</h3>
        <select id="examSelect"></select>
        <button onclick="startExam()">&#9654; Start</button>
        <div class="memo-status" id="memoStatus"></div>
      </div>
      <div class="exam-area" id="examArea"></div>
    </div>
    <div class="panel" id="panel-results">
      <div class="results-area" id="resultsArea"></div>
    </div>
    <div class="panel" id="panel-dashboard">
      <div class="dashboard" id="dashboardArea" style="padding:24px; overflow-y:auto;">
        <p>Loading dashboard...</p>
      </div>
    </div>
  </div>
</div>

<script>
// Use window. to ensure global scope for onclick handlers
const studentId = localStorage.getItem('educat_sid') || 'stu_'+Math.random().toString(36).slice(2,10);
localStorage.setItem('educat_sid', studentId);
document.getElementById('sidDisplay').textContent = 'ID: ' + studentId;

let sessionId = null, currentIdx = 0, totalQ = 0, currentType = null;

window.showPanel = function(name, e) {
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
  const target = document.getElementById('panel-' + name);
  if(target) target.classList.add('active');
  if(e && e.currentTarget) e.currentTarget.classList.add('active');
};

window.addMsg = function(role, text) {
  const box = document.getElementById('chatMessages');
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.innerHTML = text.replace(/\n/g, '<br>');
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
  return div;
};

window.sendChat = async function() {
  const inp = document.getElementById('chatInput');
  const msg = inp.value.trim();
  if(!msg) return;
  inp.value = '';
  addMsg('user', msg);
  const thinking = addMsg('agent thinking', 'Thinking...');
  try {
    const res = await fetch('/agent-chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({student_id: studentId, message: msg})
    });
    const data = await res.json();
    thinking.remove();
    addMsg('agent', data.response || 'No response');
  } catch(e) {
    thinking.remove();
    addMsg('agent', 'Error: ' + e.message);
  }
};

window.clearHistory = async function() {
  await fetch('/clear-history', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({student_id: studentId})
  });
  document.getElementById('chatMessages').innerHTML = '';
  addMsg('agent', 'Chat history cleared.');
};

window.addEventListener('load', async () => {
  try {
    const res = await fetch('/exams');
    const data = await res.json();
    const sel = document.getElementById('examSelect');
    sel.innerHTML = '<option value="">-- select exam --</option>';
    (data.exams || []).forEach(e => {
      const o = document.createElement('option');
      o.value = typeof e === 'string' ? e : e.id;
      o.text = typeof e === 'string' ? e.replace(/_/g, ' ') : (e.name || 'Exam');
      sel.appendChild(o);
    });
  } catch(e) { console.error("Load exams error:", e); }
});

window.startExam = async function() {
  const examVal = document.getElementById('examSelect').value;
  if(!examVal) return alert('Select an exam.');
  const res = await fetch('/start-exam', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({exam: examVal, student_id: studentId})
  });
  const data = await res.json();
  if(data.error) return alert(data.error);
  sessionId = data.session_id;
  totalQ = data.total_questions;
  currentIdx = 0;
  document.getElementById('memoStatus').innerHTML = data.memo_merged ? '✅ Memo loaded' : '⚠️ No memo';
  loadQuestion();
};

window.loadQuestion = async function() {
  const res = await fetch('/question', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({session_id: sessionId, index: currentIdx})
  });
  const q = await res.json();
  currentType = q.type;
  let html = `<div class="sec-header"><div class="sec-label">Question ${q.question_number}</div></div>`;
  html += `<div class="q-row"><div class="q-text">${q.question} <span class="q-mark">(${q.marks})</span></div></div>`;

  if(q.type === 'mcq' && q.options) {
    for(const [key, val] of Object.entries(q.options)) {
      const chk = q.saved_answer === key ? 'checked' : '';
      html += `<label class="option-label"><input type="radio" name="mcq" value="${key}" ${chk}> ${key}. ${val}</label>`;
    }
  } else {
    html += `<textarea id="ansBox" placeholder="Type answer...">${q.saved_answer || ''}</textarea>`;
  }

  html += `<div class="nav-bar">
    <button onclick="saveAndGo(-1)">Back</button>
    <button onclick="saveAndGo(1)">Next</button>
    <button class="submit-btn" onclick="submitExam()">Submit</button>
  </div>`;
  document.getElementById('examArea').innerHTML = html;
};

window.saveAndGo = async function(dir) {
  const ans = currentType === 'mcq' ? (document.querySelector('input[name="mcq"]:checked')?.value || '') : (document.getElementById('ansBox')?.value || '');
  await fetch('/answer', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({session_id: sessionId, index: currentIdx, answer: ans})
  });
  const next = currentIdx + dir;
  if(next >= 0 && next < totalQ) { currentIdx = next; loadQuestion(); }
};

window.submitExam = async function() {
  if(!confirm('Submit?')) return;
  const res = await fetch('/submit', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({session_id: sessionId, student_id: studentId})
  });
  const data = await res.json();
  let html = `<div class="score-banner"><h2>${data.score} / ${data.total}</h2><p>${data.percentage}%</p></div>`;
  document.getElementById('resultsArea').innerHTML = html;
  showPanel('results');
};

window.loadDashboard = async function() {
  const res = await fetch('/dashboard', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({student_id: studentId})
  });
  const data = await res.json();
  let html = '<div class="dash-card"><h3>History</h3>';
  (data.sessions || []).forEach(s => {
    html += `<div class="session-row"><span>${s.exam_name}</span><span>${s.percentage}%</span></div>`;
  });
  html += '</div>';
  document.getElementById('dashboardArea').innerHTML = html;
};
</script>
</body>
</html>"""

@app.route("/")
def home():
    return HOME_HTML


# ═══════════════════════════════════════════════════════════════════════════════
# API ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/exams", methods=["GET"])
def list_exams():
    exams = []
    try:
        docs = db_admin.collection("exams").where("status", "==", "ready").stream()
        for doc in docs:
            d = doc.to_dict()
            exams.append({
                "id": doc.id,
                "name": d.get("title", doc.id),
                "subject": d.get("subject", ""),
                "subject_code": d.get("subjectCode", ""),
                "grade": d.get("grade", ""),
                "year": d.get("year", ""),
                "paper": d.get("paper", ""),
                "curriculum": d.get("curriculum", "CAPS"),
                "source": "firestore",
            })
    except Exception as e:
        print(f"[list_exams] Firestore error: {e}")

    try:
        if os.path.isdir(EXAMS_FOLDER):
            for fname in sorted(os.listdir(EXAMS_FOLDER)):
                if fname.endswith("_exam.json"):
                    display = fname.replace("_exam.json", "").replace("_", " ")
                    already = any(e.get("name", "").lower() == display.lower() for e in exams if isinstance(e, dict))
                    if not already:
                        try:
                            with open(os.path.join(EXAMS_FOLDER, fname)) as f:
                                data = json.load(f)
                            meta = data.get("metadata", {})
                            exams.append({
                                "id": fname,
                                "name": meta.get("subject", display) + " " + meta.get("year", ""),
                                "subject": meta.get("subject", ""),
                                "grade": meta.get("grade", ""),
                                "year": meta.get("year", ""),
                                "paper": meta.get("paper_number", ""),
                                "source": "local",
                            })
                        except Exception:
                            exams.append(fname)
    except Exception as e:
        print(f"[list_exams] Local folder error: {e}")

    return jsonify({"exams": exams})

@app.route("/admin/uploads", methods=["GET"])
def admin_uploads():
    try:
        docs = db_admin.collection("teacherExamUploads").stream()
        uploads = []
        for doc in docs:
            d = doc.to_dict()
            d["examId"] = doc.id
            uploads.append(d)
        # Sort in Python instead — no Firestore index needed
        uploads.sort(key=lambda x: x.get("uploadedAt", ""), reverse=True)
        return jsonify({"uploads": uploads})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/extract-exam", methods=["POST"])
def extract_exam():
    try:
        data = request.get_json()
        exam_id = data.get("exam_id", "").strip()
        if not exam_id:
            return jsonify({"error": "exam_id required"}), 400

        # 1. Load the upload doc
        upload_ref = db_admin.collection("teacherExamUploads").document(exam_id)
        upload_doc = upload_ref.get()
        if not upload_doc.exists:
            return jsonify({"error": "Upload not found"}), 404

        meta = upload_doc.to_dict()

        # 2. Mark as processing
        upload_ref.update({"status": "processing"})

        # 3. Download exam PDF from Drive
        exam_file_id = meta.get("examDriveFileId")
        if not exam_file_id:
            upload_ref.update({"status": "error", "errorMessage": "No examDriveFileId"})
            return jsonify({"error": "No exam Drive file ID"}), 400

        pdf_bytes = download_from_drive_bytes(exam_file_id)
        if not pdf_bytes:
            upload_ref.update({"status": "error", "errorMessage": "Failed to download from Drive"})
            return jsonify({"error": "Drive download failed"}), 500

        # 4. Extract questions via AI
        questions = extract_questions_from_pdf(pdf_bytes, meta)

        # 5. Also download memo if available
        memo_map = {}
        memo_file_id = meta.get("memoDriveFileId")
        if memo_file_id:
            memo_bytes = download_from_drive_bytes(memo_file_id)
            if memo_bytes:
                # Parse memo separately and match by question number
                memo_questions = extract_questions_from_pdf(memo_bytes, {
                    **meta, "subject": meta.get("subject", "") + " MEMO"
                })
                memo_map = {q.get("question_number"): q.get("memo") or q.get("question", "")
                            for q in memo_questions if q.get("question_number")}

        # 6. Merge memo answers into questions
        for q in questions:
            qn = q.get("question_number")
            if qn and qn in memo_map and not q.get("memo"):
                q["memo"] = memo_map[qn]

        # 7. Write exam doc to `exams` collection
        exam_doc = {
            "title": meta.get("title", meta.get("examFileName", "Exam")),
            "subject": meta.get("subject", ""),
            "grade": meta.get("grade", ""),
            "year": meta.get("year", ""),
            "curriculum": meta.get("curriculum", "CAPS"),
            "teacherName": meta.get("teacherName", ""),
            "uploadedBy": meta.get("uploadedBy", ""),
            "examDriveFileId": exam_file_id,
            "memoDriveFileId": memo_file_id,
            "status": "ready",
            "totalQuestions": len(questions),
            "extractedAt": fs_admin.SERVER_TIMESTAMP,
            "sourceUploadId": exam_id,
        }
        exam_ref = db_admin.collection("exams").document(exam_id)
        exam_ref.set(exam_doc)

        # 8. Write each question to `exam_questions` collection
        batch = db_admin.batch()
        for i, q in enumerate(questions):
            q_ref = db_admin.collection("exam_questions").document(f"{exam_id}_{i:04d}")
            batch.set(q_ref, {
                "examId": exam_id,
                "questionNumber": q.get("question_number", str(i + 1)),
                "parentQuestion": q.get("parent_question", ""),
                "parentContext": q.get("parent_context"),
                "section": q.get("section", "A"),
                "questionText": q.get("question", ""),
                "type": q.get("type", "open"),
                "marks": q.get("marks", 1),
                "options": q.get("options"),
                "columnA": q.get("column_a"),
                "columnB": q.get("column_b"),
                "memo": q.get("memo", ""),
                "order": i,
            })
        batch.commit()

        # 9. Update upload doc to extracted
        upload_ref.update({
            "status": "extracted",
            "extractedAt": fs_admin.SERVER_TIMESTAMP,
            "totalQuestions": len(questions),
        })

        return jsonify({
            "ok": True,
            "exam_id": exam_id,
            "questions_extracted": len(questions),
        })

    except Exception as e:
        traceback.print_exc()
        # Mark as error so admin can retry
        try:
            upload_ref.update({"status": "error", "errorMessage": str(e)})
        except Exception:
            pass
        return jsonify({"error": str(e)}), 500

@app.route("/admin/trigger-extract/<exam_id>", methods=["GET"])
def trigger_extract_get(exam_id):
    """Temporary GET trigger for testing — remove after extraction works."""
    try:
        # reuse your extract_exam logic directly
        upload_ref = db_admin.collection("teacherExamUploads").document(exam_id)
        upload_doc = upload_ref.get()
        if not upload_doc.exists:
            return jsonify({"error": "Upload not found"}), 404
        meta = upload_doc.to_dict()
        upload_ref.update({"status": "processing"})
        pdf_bytes = download_from_drive_bytes(meta.get("examDriveFileId"))
        if not pdf_bytes:
            upload_ref.update({"status": "error", "errorMessage": "Drive download failed"})
            return jsonify({"error": "Drive download failed"}), 500
        questions = extract_questions_from_pdf(pdf_bytes, meta)
        exam_doc = {
            "title": meta.get("title", "Exam"),
            "subject": meta.get("subject", ""),
            "grade": meta.get("grade", ""),
            "year": meta.get("year", ""),
            "curriculum": meta.get("curriculum", "CAPS"),
            "teacherName": meta.get("teacherName", ""),
            "uploadedBy": meta.get("uploadedBy", ""),
            "examDriveFileId": meta.get("examDriveFileId"),
            "memoDriveFileId": meta.get("memoDriveFileId"),
            "status": "ready",
            "totalQuestions": len(questions),
            "extractedAt": fs_admin.SERVER_TIMESTAMP,
            "sourceUploadId": exam_id,
        }
        exam_ref = db_admin.collection("exams").document(exam_id)
        exam_ref.set(exam_doc)
        batch = db_admin.batch()
        for i, q in enumerate(questions):
            q_ref = db_admin.collection("exam_questions").document(f"{exam_id}_{i:04d}")
            batch.set(q_ref, {
                "examId": exam_id,
                "questionNumber": q.get("question_number", str(i+1)),
                "parentQuestion": q.get("parent_question", ""),
                "section": q.get("section", "A"),
                "questionText": q.get("question", ""),
                "type": q.get("type", "open"),
                "marks": q.get("marks", 1),
                "options": q.get("options"),
                "columnA": q.get("column_a"),
                "columnB": q.get("column_b"),
                "memo": q.get("memo", ""),
                "order": i,
            })
        batch.commit()
        upload_ref.update({"status": "extracted", "extractedAt": fs_admin.SERVER_TIMESTAMP, "totalQuestions": len(questions)})
        return jsonify({"ok": True, "questions_extracted": len(questions)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/admin/list-raw", methods=["GET"])
def list_raw():
    """Temp diagnostic — lists all teacherExamUploads doc IDs."""
    try:
        docs = db_admin.collection("teacherExamUploads").stream()
        return jsonify({"doc_ids": [doc.id for doc in docs]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/agent-chat", methods=["POST"])
def agent_chat():
    try:
        data = request.get_json()
        student_id = data.get("student_id", "anonymous")
        message = data.get("message", "").strip()
        if not message:
            return jsonify({"response": "Please enter a message."})
        response = run_agent(student_id, message, rag=rag)
        return jsonify({"response": response})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"response": f"Agent error: {e}"})


@app.route("/clear-history", methods=["POST"])
def clear_history():
    data = request.get_json()
    mem.clear_history(data.get("student_id", ""))
    return jsonify({"status": "cleared"})


@app.route("/start-exam", methods=["POST"])
def start_exam():
    try:
        data = request.get_json()
        exam_value = data.get("exam", "").strip()
        student_id = data.get("student_id", "anonymous")
        if not exam_value:
            return jsonify({"error": "No exam specified"})

        flat = []
        memo_merged = False
        exam_label = exam_value
        subject = ""

        if not exam_value.endswith("_exam.json"):
            meta, flat = load_exam_from_firestore(exam_value)
            if meta is None:
                return jsonify({"error": f"Exam '{exam_value}' not found in Firestore"})
            memo_merged = bool(meta.get("memoDriveFileId"))
            exam_label = meta.get("title", exam_value)
            subject = meta.get("subject", "")
            if not flat:
                status = meta.get("status", "unknown")
                return jsonify({"error": f"Exam is not ready yet (status: {status})."})
        else:
            exam = load_exam_local(exam_value)
            if not exam:
                return jsonify({"error": f"Exam file '{exam_value}' not found"})
            flat = flatten_exam(exam)
            memo_merged = exam.get("memo_merged", False)
            subject = exam.get("metadata", {}).get("subject", "")
            if not flat:
                return jsonify({"error": "No questions found in exam file"})

        mem.ensure_student(student_id)
        sid = str(uuid.uuid4())
        sessions[sid] = {
            "exam": exam_label,
            "subject": subject,
            "student_id": student_id,
            "questions": flat,
            "answers": {},
        }
        return jsonify({
            "session_id": sid,
            "total_questions": len(flat),
            "memo_merged": memo_merged,
            "subject": subject,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)})


@app.route("/question", methods=["POST"])
def get_question():
    try:
        data = request.get_json()
        sid = data.get("session_id")
        idx = data.get("index", 0)
        session = sessions.get(sid)
        if not session:
            return jsonify({"error": "Invalid session"})
        flat = session["questions"]
        if idx < 0 or idx >= len(flat):
            return jsonify({"error": "Index out of range"})
        q = flat[idx].copy()
        q["saved_answer"] = session["answers"].get(str(idx), "")
        q["rendered_input"] = render_question_input(q, q["saved_answer"])
        return jsonify(q)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)})


@app.route("/answer", methods=["POST"])
def save_answer():
    try:
        data = request.get_json()
        sid = data.get("session_id")
        idx = data.get("index")
        answer = data.get("answer", "")
        session = sessions.get(sid)
        if not session:
            return jsonify({"error": "Invalid session"})
        session["answers"][str(idx)] = answer
        return jsonify({"status": "saved"})
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/submit", methods=["POST"])
def submit_exam():
    try:
        data = request.get_json()
        sid = data.get("session_id")
        student_id = data.get("student_id", "anonymous")
        session = sessions.get(sid)
        if not session:
            return jsonify({"error": "Invalid session"})

        exam_name = session["exam"]
        subject = session.get("subject", "")
        flat = session["questions"]
        answers = session["answers"]

        results = []
        total_score = 0
        total_marks = 0

        for i, q in enumerate(flat):
            q_num = q.get("question_number", f"Q{i+1}")
            q_type = q.get("type", "open").lower()
            marks = int(q.get("marks", 1))
            q_text = q.get("question", "")
            memo = q.get("memo", "")
            student = answers.get(str(i), "").strip()
            options = q.get("options")
            instructions = q.get("instructions", "")

            result = mark_answer(
                question=q_text, question_number=q_num, q_type=q_type,
                student_answer=student, memo=memo, marks=marks, options=options,
                instructions=instructions, subject=subject,
            )

            if result.get("status") in ("incorrect", "missing"):
                topic = q.get("parent_question", "").split(":")[1].strip() if ":" in q.get("parent_question", "") else subject
                mem.record_wrong(student_id, q_num, q_text, q_type, topic)
            elif result.get("status") == "correct":
                mem.record_correct(student_id, q_num)

            correct_display = "Not available"
            if memo:
                if q_type == "mcq" and options:
                    cl = str(memo).strip().upper()
                    correct_display = cl
                    for opt_key, opt_val in (options.items() if isinstance(options, dict) else []):
                        if str(opt_key).upper() == cl:
                            correct_display = f"{cl}. {opt_val}"
                            break
                elif q_type == "matching" and isinstance(memo, dict):
                    correct_display = " | ".join(f"{k} → {v}" for k, v in memo.items())
                else:
                    correct_display = str(memo)

            result["question_number"] = q_num
            result["question"] = q_text
            result["type"] = q_type
            result["marks"] = marks
            result["student_answer"] = student or "No answer"
            result["correct_answer"] = correct_display
            result["earned"] = result.get("score", 0)

            results.append(result)
            total_score += result["earned"]
            total_marks += marks

        percentage = round((total_score / total_marks * 100), 1) if total_marks else 0
        mem.save_session(student_id, exam_name, total_score, total_marks, percentage, subject=subject)
        feedback = generate_exam_feedback(results, total_score, total_marks, percentage, subject=subject)

        weak = mem.get_weak_topics(student_id)
        if weak:
            try:
                run_agent(
                    student_id,
                    f"I just scored {percentage}% on {exam_name} ({subject}). Please update my study plan based on my weak areas.",
                    rag=rag,
                )
            except Exception:
                pass

        return jsonify({
            "score": total_score, "total": total_marks,
            "percentage": percentage, "results": results, "feedback": feedback,
            "subject": subject,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)})


@app.route("/dashboard", methods=["POST"])
def dashboard():
    try:
        data = request.get_json()
        student_id = data.get("student_id", "anonymous")
        mem.ensure_student(student_id)
        return jsonify({
            "weak": mem.get_weak_topics(student_id),
            "sessions": mem.get_sessions(student_id, limit=8),
            "study_plan": mem.get_study_plan(student_id),
        })
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True, port=8000)