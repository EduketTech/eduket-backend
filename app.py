"""
app.py  —  EduCAT Flask API  (Firestore-native exam pipeline)

WHAT CHANGED FROM PREVIOUS VERSION
────────────────────────────────────
1. Exams now live in Firestore (/exams collection) instead of local JSON files.
   Teachers upload PDFs via the React frontend → Google Drive → Firestore record.
   The scheduled extraction task (scheduled_task.py) processes PDFs and sets
   status="ready". Only ready exams are served to students.

2. /exams  — reads Firestore, returns {id, name, subject, grade, year, curriculum}
             instead of a flat list of filenames.

3. /start-exam — accepts exam_id (Firestore doc ID) instead of a filename.
                 Loads questions from /exam_questions subcollection.
                 Falls back to local JSON files (EXAMS_FOLDER) for backward
                 compatibility during the migration period.

4. A Drive helper (download_from_drive_bytes) is added so the extraction
   scheduled task can fetch PDFs using the service account without needing
   a user OAuth token.

5. Everything else (session management, /submit marking, /dashboard,
   /agent-chat, model.py, memory.py, agent.py, home UI) is unchanged.

ENVIRONMENT VARIABLES REQUIRED
───────────────────────────────
  GOOGLE_APPLICATION_CREDENTIALS  path to your service account JSON file
    OR
  FIREBASE_SERVICE_ACCOUNT_JSON   JSON string of the service account (for
                                  PythonAnywhere env-var based config)

  GROQ_API_KEY                    already set
  SERVICE_ACCOUNT_EMAIL           e.g. educat@your-project.iam.gserviceaccount.com
                                  (used to share uploaded Drive files with backend)
"""

from dotenv import load_dotenv
load_dotenv()

import os
import io
import json
import uuid
import re
import traceback
import requests as http_requests   # renamed to avoid clash with Flask's request

from flask import Flask, request, jsonify
from flask_cors import CORS

# ── Firebase Admin ────────────────────────────────────────────────────────────
import firebase_admin
from firebase_admin import credentials, firestore as fs_admin


def _init_firebase():
    """
    Initialises Firebase Admin SDK.
    Accepts either a path (GOOGLE_APPLICATION_CREDENTIALS env var) or an
    inline JSON string (FIREBASE_SERVICE_ACCOUNT_JSON).
    """
    if firebase_admin._apps:
        return

    inline_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
    if inline_json:
        try:
            sa_dict = json.loads(inline_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"FIREBASE_SERVICE_ACCOUNT_JSON is not valid JSON: {e}")

        # Debug print — safe because sa_dict is guaranteed to exist here
        print("[Firebase] Using inline JSON credentials")
        print("[Firebase] Keys found:", list(sa_dict.keys()))
        print("[Firebase] Type field:", sa_dict.get('type'))

        if sa_dict.get("type") != "service_account":
            raise ValueError(
                'FIREBASE_SERVICE_ACCOUNT_JSON must contain "type": "service_account". '
                f'Got type={sa_dict.get("type")!r}. '
                'Did you accidentally paste a file path instead of the JSON content?'
            )
        cred = credentials.Certificate(sa_dict)
    else:
        # Falls back to GOOGLE_APPLICATION_CREDENTIALS path automatically
        print("[Firebase] Using Application Default Credentials")
        cred = credentials.ApplicationDefault()

    firebase_admin.initialize_app(cred)


# Call it at module level — UNINDENTED
_init_firebase()
db_admin = fs_admin.client()


# ── Local modules ─────────────────────────────────────────────────────────────
from model import generate_answer, mark_answer, generate_exam_feedback
from rag import RAGIndex
import memory as mem
import agent
from agent import run_agent

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": [
    "http://localhost:3000",
    "http://localhost:5176",
    "https://edu-cat.netlify.app",
]}})

# ── Initialise RAG and inject into agent tools ────────────────────────────────
rag = RAGIndex()
agent.set_rag(rag)

# ── Fallback for exams still stored as local JSON (migration period) ──────────
EXAMS_FOLDER = "exams"
sessions     = {}   # in-memory exam sessions

# ── Service-account email (backend shares Drive files to this address) ────────
SERVICE_ACCOUNT_EMAIL = os.getenv(
    "SERVICE_ACCOUNT_EMAIL",
    ""   # set this in your PythonAnywhere env vars
)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

# ── Legacy: local JSON loader (kept for migration compatibility) ───────────────
def load_exam_local(exam_name: str) -> dict | None:
    path = os.path.join(EXAMS_FOLDER, exam_name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ── Firestore: load exam metadata + questions ─────────────────────────────────
def load_exam_from_firestore(exam_id: str) -> tuple[dict | None, list]:
    """
    Returns (exam_metadata_dict, flat_questions_list).
    Questions come from the /exam_questions collection filtered by examId.
    If no questions are found in Firestore, returns (meta, []) — caller can
    then attempt the local JSON fallback.
    """
    try:
        doc = db_admin.collection("exams").document(exam_id).get()
        if not doc.exists:
            return None, []

        meta = doc.to_dict()
        meta["id"] = doc.id

        q_docs = (
            db_admin.collection("exam_questions")
            .where("examId", "==", exam_id)
            .order_by("questionNumber")
            .stream()
        )

        questions = []
        for q in q_docs:
            d = q.to_dict()
            questions.append({
                "id":               d.get("questionNumber", q.id),
                "question_number":  str(d.get("questionNumber", "")),
                "parent_question":  d.get("parentQuestion", ""),
                "parent_context":   d.get("parentContext"),
                "section":          d.get("section", "A"),
                "section_title":    d.get("sectionTitle", ""),
                "section_instructions": d.get("sectionInstructions", ""),
                "section_total_marks":  d.get("sectionTotalMarks"),
                "question":         d.get("questionText", "⚠️ Question text missing"),
                "type":             d.get("type", "open").lower(),
                "options":          d.get("options"),       # list of {key, value}
                "column_a":         d.get("columnA"),
                "column_b":         d.get("columnB"),
                "marks":            d.get("marks", 1),
                "memo":             d.get("memo", ""),
                "saved_answer":     "",
            })

        return meta, questions

    except Exception as e:
        traceback.print_exc()
        return None, []


# ── Download a Drive file as bytes using the service account ──────────────────
def download_from_drive_bytes(file_id: str) -> bytes | None:
    """
    Downloads a Drive file by ID using the service account's OAuth token.
    Used by the extraction pipeline (scheduled_task.py).
    Not called during normal exam serving — questions come from Firestore.
    """
    try:
        from google.oauth2 import service_account
        from google.auth.transport.requests import Request as GoogleRequest

        inline_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
        if inline_json:
            sa_info = json.loads(inline_json)
            creds = service_account.Credentials.from_service_account_info(
                sa_info,
                scopes=["https://www.googleapis.com/auth/drive.readonly"]
            )
        else:
            sa_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "serviceAccountKey.json")
            creds = service_account.Credentials.from_service_account_file(
                sa_path,
                scopes=["https://www.googleapis.com/auth/drive.readonly"]
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


def flatten_exam(exam: dict) -> list:
    """Convert sections → flat list of question dicts (local JSON path only)."""
    flat     = []
    sections = exam.get("sections", [])
    if not sections and "questions" in exam:
        sections = [{
            "section": "A", "section_title": None,
            "section_instructions": None, "total_marks": None,
            "questions": exam["questions"],
        }]

    for section in sections:
        sec_label        = section.get("section", "")
        sec_title        = section.get("section_title") or ""
        sec_instructions = section.get("section_instructions") or ""
        sec_marks        = section.get("total_marks")

        for q in section.get("questions", []):
            q_type          = q.get("type", "open").lower()
            q_text          = q.get("question", "").strip()
            marks           = q.get("marks", 1)
            memo            = q.get("memo", "")
            q_id            = q.get("id")
            question_number = q.get("question_number", f"Q{q_id}" if q_id else "")
            parent_question = q.get("parent_question", "")
            parent_context  = q.get("parent_context")
            options         = None
            column_a = column_b = None

            if q_type == "mcq":
                raw_opts = q.get("options")
                if isinstance(raw_opts, dict):
                    options = [{"key": k, "value": v} for k, v in sorted(raw_opts.items()) if str(v).strip()]
                elif isinstance(raw_opts, list):
                    options = [{"key": chr(65 + i), "value": str(v).strip()} for i, v in enumerate(raw_opts) if str(v).strip()]

            if q_type == "matching":
                column_a = q.get("column_a", [])
                column_b = q.get("column_b", [])

            flat.append({
                "id": q_id, "question_number": question_number,
                "parent_question": parent_question, "parent_context": parent_context,
                "section": sec_label, "section_title": sec_title,
                "section_instructions": sec_instructions, "section_total_marks": sec_marks,
                "question": q_text or "⚠️ Question text missing",
                "type": q_type, "options": options,
                "column_a": column_a, "column_b": column_b,
                "marks": marks, "memo": memo, "saved_answer": "",
            })
    return flat


# ═══════════════════════════════════════════════════════════════════════════════
# HOME UI  (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def home():
    return r"""<!DOCTYPE html>
<html>
<head>
<title>Eduket — AI Agent</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:Arial,sans-serif;background:#f4f6f9;color:#2c3e50}
.app{display:flex;height:100vh}
.sidebar{width:240px;background:#2c3e50;color:#ecf0f1;padding:20px;display:flex;flex-direction:column;gap:12px;flex-shrink:0}
.sidebar h2{font-size:18px;font-weight:700;margin-bottom:8px}
.sidebar .student-id{font-size:11px;opacity:.5;word-break:break-all}
.nav-btn{padding:10px 14px;background:rgba(255,255,255,.08);border:none;color:#ecf0f1;border-radius:8px;cursor:pointer;text-align:left;font-size:13px}
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
.sec-header .sec-label{font-weight:bold;font-size:15px}
.sec-header .sec-sub{font-size:12px;color:#555;margin-top:3px}
.parent-heading{font-size:12px;font-weight:bold;color:#7f8c8d;text-transform:uppercase;margin-bottom:4px}
.parent-context{background:#fefae0;border-left:3px solid #f1c40f;padding:8px 12px;border-radius:5px;font-size:13px;color:#555;margin-bottom:10px}
.q-row{display:flex;gap:10px;margin-top:10px;align-items:flex-start}
.q-num{font-weight:bold;min-width:40px;color:#2c3e50}
.q-text{flex:1;font-size:14px;line-height:1.5}
.q-mark{color:#e74c3c;font-weight:bold;white-space:nowrap}
.option-label{display:block;margin:7px 0;padding:8px 13px;border:1px solid #ddd;border-radius:6px;cursor:pointer;font-size:13px}
.option-label:hover{background:#f0f4ff}
.option-label input{margin-right:8px}
.tf-label{display:inline-block;margin-right:16px;padding:8px 16px;border:1px solid #ddd;border-radius:6px;cursor:pointer;font-size:13px}
.tf-label:hover{background:#f0f4ff}
.match-table{width:100%;border-collapse:collapse;margin-top:8px;font-size:13px}
.match-table th{text-align:left;padding:7px 9px;border-bottom:2px solid #ddd;background:#f5f5f5}
.match-table td{padding:7px 9px;border-bottom:1px solid #eee;vertical-align:middle}
.match-table select{width:100%;padding:5px;border-radius:4px;border:1px solid #ccc}
textarea{width:100%;height:110px;margin-top:12px;padding:10px;border:1px solid #ddd;border-radius:7px;font-size:13px;resize:vertical}
.nav-bar{margin-top:16px;display:flex;gap:8px;flex-wrap:wrap}
.nav-bar button{padding:9px 16px;border:none;border-radius:7px;cursor:pointer;font-size:13px;background:#3498db;color:#fff}
.nav-bar .submit-btn{margin-left:auto;background:#e74c3c}
.progress{margin-top:8px;color:#888;font-size:12px}
.results-area{flex:1;overflow-y:auto;padding:24px}
.score-banner{background:#fff;border:1px solid #e0e0e0;border-radius:10px;padding:20px;margin-bottom:16px;text-align:center}
.score-banner h2{font-size:28px;margin-bottom:6px}
.feedback-box{background:#eafaf1;border-left:4px solid #27ae60;padding:14px;border-radius:8px;margin-bottom:16px;font-size:13px;line-height:1.6}
.result-card{border:1px solid #ddd;border-radius:8px;padding:14px;margin-bottom:10px;font-size:13px;line-height:1.7}
.dashboard{flex:1;overflow-y:auto;padding:24px;display:flex;flex-direction:column;gap:16px}
.dash-card{background:#fff;border:1px solid #e0e0e0;border-radius:10px;padding:18px}
.dash-card h3{font-size:15px;margin-bottom:10px;font-weight:600}
.weak-item{display:flex;align-items:center;gap:10px;padding:7px 0;border-bottom:1px solid #f0f0f0;font-size:13px}
.weak-bar-bg{flex:1;height:6px;background:#f0f0f0;border-radius:3px}
.weak-bar{height:6px;background:#e74c3c;border-radius:3px}
.session-row{display:flex;justify-content:space-between;font-size:13px;padding:6px 0;border-bottom:1px solid #f0f0f0}
.plan-text{font-size:13px;line-height:1.7;white-space:pre-wrap;color:#555}
.hint-btn{padding:5px 10px;background:#f39c12;color:#fff;border:none;border-radius:5px;cursor:pointer;font-size:12px}
</style>
</head>
<body>
<div class="app">
  <div class="sidebar">
    <h2>🎓 EduCAT</h2>
    <div class="student-id" id="sidDisplay"></div>
    <div class="sep"></div>
    <button class="nav-btn active" onclick="showPanel('chat',event)">💬 AI Tutor</button>
    <button class="nav-btn" onclick="showPanel('exam',event)">📝 Exam Mocker</button>
    <button class="nav-btn" onclick="showPanel('dashboard',event);loadDashboard()">📊 My Dashboard</button>
    <div class="sep"></div>
    <button class="nav-btn" onclick="clearHistory()">🗑 Clear chat</button>
  </div>
  <div class="main">
    <div class="panel active" id="panel-chat">
      <div class="chat-messages" id="chatMessages"></div>
      <div class="chat-input-row">
        <input id="chatInput" placeholder="Ask anything about CAT, or say 'what should I study?'" onkeydown="if(event.key==='Enter')sendChat()">
        <button onclick="sendChat()">Send</button>
      </div>
    </div>
    <div class="panel" id="panel-exam">
      <div class="exam-setup">
        <h3>📝 Exam Mocker</h3>
        <select id="examSelect"></select>
        <button onclick="startExam()">▶ Start</button>
        <div class="memo-status" id="memoStatus"></div>
      </div>
      <div class="exam-area" id="examArea"></div>
    </div>
    <div class="panel" id="panel-results">
      <div class="results-area" id="resultsArea"></div>
    </div>
    <div class="panel" id="panel-dashboard">
      <div class="dashboard" id="dashboardArea">
        <p style="color:#888;font-size:13px">Loading dashboard...</p>
      </div>
    </div>
  </div>
</div>
<script>
const studentId=localStorage.getItem('educat_sid')||'stu_'+Math.random().toString(36).slice(2,10);
localStorage.setItem('educat_sid',studentId);
document.getElementById('sidDisplay').textContent='ID: '+studentId;

/*
 * examIdMap: maps the <option> value shown in the UI to the Firestore examId.
 * For Firestore exams the value IS the examId.
 * For legacy local-file exams value is the filename (ends in _exam.json).
 */
let examIdMap = {};
let sessionId=null,currentIdx=0,totalQ=0,currentType=null;

function showPanel(name,e){
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(b=>b.classList.remove('active'));
  const p=document.getElementById('panel-'+name);if(p)p.classList.add('active');
  if(e&&e.target)e.target.classList.add('active');
}
function addMsg(role,text){
  const box=document.getElementById('chatMessages');
  const div=document.createElement('div');div.className='msg '+role;
  div.innerHTML=text.replace(/\n/g,'<br>');box.appendChild(div);box.scrollTop=box.scrollHeight;return div;
}
async function sendChat(){
  const inp=document.getElementById('chatInput');const msg=inp.value.trim();if(!msg)return;
  inp.value='';addMsg('user',msg);const thinking=addMsg('agent thinking','🤔 Thinking...');
  try{
    const res=await fetch('/agent-chat',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({student_id:studentId,message:msg})});
    const data=await res.json();thinking.remove();addMsg('agent',data.response||'⚠️ No response');
  }catch(e){thinking.remove();addMsg('agent','⚠️ Error: '+e.message);}
}
async function clearHistory(){
  await fetch('/clear-history',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({student_id:studentId})});
  document.getElementById('chatMessages').innerHTML='';
  addMsg('agent','🗑 Chat history cleared. How can I help you?');
}

/* Load exam list — now returns objects with id, name, subject, grade */
window.addEventListener('load',async()=>{
  const res=await fetch('/exams');
  const data=await res.json();
  const sel=document.getElementById('examSelect');
  sel.innerHTML='<option value="">— select exam —</option>';

  (data.exams||[]).forEach(e=>{
    const o=document.createElement('option');
    const isLegacy = typeof e === 'string';          // old local-file format
    if(isLegacy){
      o.value = e;
      o.text  = e.replace('_exam.json','').replace(/_/g,' ');
    } else {
      o.value = e.id;                                // Firestore doc ID
      o.text  = `${e.name}${e.grade?' — Gr '+e.grade:''}${e.year?' ('+e.year+')':''}`;
    }
    sel.appendChild(o);
  });
});

async function startExam(){
  const examVal=document.getElementById('examSelect').value;
  if(!examVal){alert('Select an exam first.');return;}
  const res=await fetch('/start-exam',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({exam:examVal,student_id:studentId})});
  const data=await res.json();if(data.error){alert(data.error);return;}
  sessionId=data.session_id;totalQ=data.total_questions;currentIdx=0;
  document.getElementById('memoStatus').innerHTML=data.memo_merged?'✅ Memo loaded':'⚠️ No memo — AI feedback only';
  document.getElementById('examArea').innerHTML='';showPanel('exam');loadQuestion();
}

/* --- rest of UI JS unchanged --- */
async function loadQuestion(){
  const res=await fetch('/question',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({session_id:sessionId,index:currentIdx})});
  const q=await res.json();if(q.error){document.getElementById('examArea').innerHTML='<p style="color:red">'+q.error+'</p>';return;}
  currentType=q.type;
  let html=`<div class="sec-header"><div class="sec-label">SECTION ${q.section}${q.section_title?' — '+q.section_title:''}</div>${q.section_instructions?`<div class="sec-sub">${q.section_instructions}</div>`:''}${q.section_total_marks?`<div class="sec-sub">Total: <b>${q.section_total_marks}</b> marks</div>`:''}</div>`;
  if(q.parent_question)html+=`<div class="parent-heading">${q.parent_question}</div>`;
  if(q.parent_context)html+=`<div class="parent-context">📌 ${q.parent_context}</div>`;
  html+=`<div class="q-row"><span class="q-num">${q.question_number}.</span><span class="q-text">${q.question}</span><span class="q-mark">(${q.marks})</span></div>`;
  if(q.type==='mcq'&&Array.isArray(q.options)&&q.options.length){
    html+='<div id="mcqOptions" style="margin-top:12px">';
    q.options.forEach(opt=>{const chk=q.saved_answer===opt.key?'checked':'';html+=`<label class="option-label"><input type="radio" name="mcq_answer" value="${opt.key}" ${chk}> <b>${opt.key}.</b> ${opt.value}</label>`;});
    html+='</div>';
  }else if(q.type==='true_false'){
    const sf=q.saved_answer||'';const isF=sf.startsWith('False');const corr=isF&&sf.includes('—')?sf.split('—').slice(1).join('—').trim():'';
    html+=`<div id="tfOptions" style="margin-top:12px"><label class="tf-label"><input type="radio" name="tf_answer" value="True" ${sf==='True'?'checked':''}> ✅ True</label><label class="tf-label"><input type="radio" name="tf_answer" value="False" ${isF?'checked':''}> ❌ False</label></div><div id="tfCorrection" style="margin-top:10px;${isF?'':'display:none'}"><label style="font-size:12px;color:#555">If FALSE — corrected word/phrase:</label><input type="text" id="tfCorrectionBox" placeholder="e.g. secondary memory" value="${corr}" style="width:100%;margin-top:4px;padding:8px;border:1px solid #ddd;border-radius:6px;font-size:13px"></div>`;
  }else if(q.type==='matching'&&Array.isArray(q.column_a)&&q.column_a.length){
    let saved={};try{saved=typeof q.saved_answer==='string'&&q.saved_answer?JSON.parse(q.saved_answer):{};}catch(e){}
    html+=`<p style="margin-top:10px;font-size:12px;color:#555"><i>Match each COLUMN A item to COLUMN B.</i></p><table class="match-table"><thead><tr><th style="width:55%">COLUMN A</th><th>COLUMN B</th></tr></thead><tbody>`;
    q.column_a.forEach((item,i)=>{const sv=saved[item]||'';html+=`<tr><td>${item}</td><td><select class="match-select" data-item="${encodeURIComponent(item)}"><option value="">-- Select --</option>`;q.column_b.forEach(b=>{html+=`<option value="${b}" ${sv===b?'selected':''}>${b}</option>`;});html+=`</select></td></tr>`;});
    html+='</tbody></table>';
  }else{html+=`<textarea id="openAnswerBox" placeholder="Write your answer here...">${q.saved_answer||''}</textarea>`;}
  if((q.type==='open'||q.type==='true_false')&&q.memo){html+=`<div style="margin-top:8px"><button class="hint-btn" onclick="askHint(${JSON.stringify(q.question).replace(/"/g,'&quot;')},'${q.question_number}',${JSON.stringify(String(q.memo)).replace(/"/g,'&quot;')})">💡 Get a hint</button></div>`;}
  html+=`<div class="nav-bar"><button onclick="saveAndGo(-1)">⬅ Back</button><button onclick="saveOnly()">💾 Save</button><button onclick="saveAndGo(1)">Next ➡</button><button class="submit-btn" onclick="submitExam()">✅ Submit</button></div><p class="progress">Question ${currentIdx+1} of ${totalQ}</p>`;
  document.getElementById('examArea').innerHTML=html;
  document.querySelectorAll('input[name="tf_answer"]').forEach(r=>{r.addEventListener('change',()=>{const cb=document.getElementById('tfCorrection');if(cb)cb.style.display=(r.value==='False'&&r.checked)?'block':'none';});});
}
function collectAnswer(){
  if(currentType==='mcq'){const s=document.querySelector('input[name="mcq_answer"]:checked');return s?s.value:'';}
  if(currentType==='true_false'){const s=document.querySelector('input[name="tf_answer"]:checked');if(!s)return '';if(s.value==='False'){const c=(document.getElementById('tfCorrectionBox')?.value||'').trim();return c?`False — ${c}`:'False';}return 'True';}
  if(currentType==='matching'){const obj={};document.querySelectorAll('.match-select').forEach(s=>{if(s.value)obj[decodeURIComponent(s.dataset.item)]=s.value;});return Object.keys(obj).length?JSON.stringify(obj):'';}
  return(document.getElementById('openAnswerBox')?.value||'').trim();
}
async function saveCurrentAnswer(){const answer=collectAnswer();await fetch('/answer',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:sessionId,index:currentIdx,answer})});return answer;}
async function saveOnly(){await saveCurrentAnswer();const b=document.querySelector("button[onclick='saveOnly()']");if(b){b.textContent='✅ Saved!';setTimeout(()=>b.textContent='💾 Save',1500);}}
async function saveAndGo(dir){await saveCurrentAnswer();const next=currentIdx+dir;if(next>=0&&next<totalQ){currentIdx=next;loadQuestion();}}
async function askHint(qText,qNum,memo){
  showPanel('chat');addMsg('user',`💡 I need a hint for question ${qNum}`);
  const thinking=addMsg('agent thinking','🤔 Generating hint...');
  const res=await fetch('/agent-chat',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({student_id:studentId,message:`Give me a Socratic hint for question ${qNum}: "${qText}". The memo answer is: "${memo}". Do not reveal the full answer.`})});
  const data=await res.json();thinking.remove();addMsg('agent',data.response||'⚠️ Could not generate hint');
}
async function submitExam(){
  await saveCurrentAnswer();if(!confirm('Submit exam?'))return;
  const res=await fetch('/submit',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:sessionId,student_id:studentId})});
  const data=await res.json();if(data.error){alert(data.error);return;}
  let html=`<div class="score-banner"><h2>${data.score} / ${data.total}</h2><p style="font-size:20px;font-weight:600">${data.percentage}%</p></div>`;
  if(data.feedback)html+=`<div class="feedback-box">🤖 <b>AI Feedback:</b><br>${data.feedback}</div>`;
  data.results.forEach(r=>{
    const bg=r.status==='correct'?'#d4edda':r.status==='partial'?'#fff3cd':'#f8d7da';
    const ic=r.status==='correct'?'✅':r.status==='partial'?'⚠️':'❌';
    let sd=r.student_answer||'<i>No answer</i>';
    if(r.type==='matching'&&r.student_answer){try{const obj=JSON.parse(r.student_answer);sd=Object.entries(obj).map(([k,v])=>`${k} → ${v}`).join('<br>');}catch(e){}}
    html+=`<div class="result-card" style="background:${bg}"><b>${ic} ${r.question_number} [${r.marks}]:</b> ${r.question}<br><b>Your answer:</b> ${sd}<br><b>Correct:</b> ${r.correct_answer||'<i>Not available</i>'}<br><b>Feedback:</b> ${r.feedback||'—'}<br><b>Earned:</b> ${r.earned}/${r.marks}</div>`;
  });
  document.getElementById('resultsArea').innerHTML=html;showPanel('results');
}
async function loadDashboard(){
  const res=await fetch('/dashboard',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({student_id:studentId})});
  const data=await res.json();
  const maxWrong=Math.max(1,...(data.weak||[]).map(w=>w.wrong_count));
  let html=`<div class="dash-card"><h3>📉 Weak areas</h3>`;
  if(data.weak&&data.weak.length){data.weak.forEach(w=>{const pct=Math.round((w.wrong_count/maxWrong)*100);html+=`<div class="weak-item"><span style="min-width:50px;font-weight:600">Q${w.question_number}</span><div class="weak-bar-bg"><div class="weak-bar" style="width:${pct}%"></div></div><span style="min-width:60px;color:#888">${w.wrong_count}x wrong</span><span style="font-size:12px;color:#aaa;max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${w.question_text||''}</span></div>`;});}else{html+=`<p style="color:#888;font-size:13px">No weak areas yet — take some exams!</p>`;}
  html+=`</div><div class="dash-card"><h3>📅 Recent sessions</h3>`;
  if(data.sessions&&data.sessions.length){data.sessions.forEach(s=>{const col=s.percentage>=70?'#27ae60':s.percentage>=50?'#f39c12':'#e74c3c';html+=`<div class="session-row"><span>${s.exam_name.replace('_exam.json','').replace(/_/g,' ')}</span><span style="color:${col};font-weight:600">${s.score}/${s.total} (${s.percentage}%)</span><span style="color:#aaa;font-size:11px">${s.played_at.split(' ')[0]}</span></div>`;});}else{html+=`<p style="color:#888;font-size:13px">No sessions recorded yet.</p>`;}
  html+=`</div><div class="dash-card"><h3>📋 Study plan</h3>`;
  if(data.study_plan){html+=`<p style="font-size:11px;color:#aaa;margin-bottom:6px">Updated: ${data.study_plan.updated_at}</p><div class="plan-text">${data.study_plan.plan}</div>`;}else{html+=`<p style="color:#888;font-size:13px">No study plan yet. Ask the AI Tutor: "Create a study plan for me"</p>`;}
  html+=`</div>`;document.getElementById('dashboardArea').innerHTML=html;
}
</script>
</body>
</html>"""


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/exams", methods=["GET"])
def list_exams():
    """
    Returns all available exams from TWO sources, merged:

    1. Firestore /exams where status == "ready"
       → returned as rich objects: {id, name, subject, grade, year, curriculum}

    2. Local JSON files in EXAMS_FOLDER (legacy / migration fallback)
       → returned as plain strings (filename) so the existing UI still works

    The React AIExamMocker and the embedded home UI both handle both shapes.
    Once all exams are migrated to Firestore you can remove the local fallback.
    """
    exams = []

    # ── Source 1: Firestore ready exams ──────────────────────────────────────
    try:
        docs = (
            db_admin.collection("exams")
            .where("status", "==", "ready")
            .stream()
        )
        for doc in docs:
            d = doc.to_dict()
            exams.append({
                "id":         doc.id,
                "name":       d.get("title", doc.id),
                "subject":    d.get("subject", ""),
                "grade":      d.get("grade", ""),
                "year":       d.get("year", ""),
                "curriculum": d.get("curriculum", "CAPS"),
                "source":     "firestore",
            })
    except Exception as e:
        print(f"[list_exams] Firestore error: {e}")

    # ── Source 2: Local JSON files (legacy fallback) ──────────────────────────
    try:
        if os.path.isdir(EXAMS_FOLDER):
            for fname in sorted(os.listdir(EXAMS_FOLDER)):
                if fname.endswith("_exam.json"):
                    # Skip if a Firestore exam with same title already listed
                    display = fname.replace("_exam.json", "").replace("_", " ")
                    already = any(
                        e.get("name", "").lower() == display.lower()
                        for e in exams
                        if isinstance(e, dict)
                    )
                    if not already:
                        exams.append(fname)   # plain string — legacy shape
    except Exception as e:
        print(f"[list_exams] Local folder error: {e}")

    return jsonify({"exams": exams})


@app.route("/agent-chat", methods=["POST"])
def agent_chat():
    try:
        data       = request.get_json()
        student_id = data.get("student_id", "anonymous")
        message    = data.get("message", "").strip()
        if not message:
            return jsonify({"response": "⚠️ Please enter a message."})
        response = run_agent(student_id, message, rag=rag)
        return jsonify({"response": response})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"response": f"⚠️ Agent error: {e}"})


@app.route("/clear-history", methods=["POST"])
def clear_history():
    data = request.get_json()
    mem.clear_history(data.get("student_id", ""))
    return jsonify({"status": "cleared"})


@app.route("/start-exam", methods=["POST"])
def start_exam():
    """
    Accepts either:
      • A Firestore examId  (new path) — loads questions from /exam_questions
      • A legacy filename   (old path) — loads from local JSON file

    Detection: if the value ends with "_exam.json" it is a legacy filename.
    Otherwise it is treated as a Firestore doc ID.
    """
    try:
        data       = request.get_json()
        exam_value = data.get("exam", "").strip()
        student_id = data.get("student_id", "anonymous")

        if not exam_value:
            return jsonify({"error": "❌ No exam specified"})

        flat       = []
        memo_merged = False
        exam_label  = exam_value   # used for session recording

        # ── Path A: Firestore exam ────────────────────────────────────────────
        if not exam_value.endswith("_exam.json"):
            meta, flat = load_exam_from_firestore(exam_value)

            if meta is None:
                return jsonify({"error": f"❌ Exam '{exam_value}' not found in Firestore"})

            memo_merged = bool(meta.get("memoDriveFileId"))
            exam_label  = meta.get("title", exam_value)

            # If extraction pipeline hasn't stored questions yet, surface a
            # clear error rather than serving an empty exam.
            if not flat:
                status = meta.get("status", "unknown")
                return jsonify({
                    "error": (
                        f"❌ Exam is not ready yet (status: {status}). "
                        "The extraction pipeline may still be processing. "
                        "Please try again in a few minutes."
                    )
                })

        # ── Path B: Legacy local JSON ─────────────────────────────────────────
        else:
            exam = load_exam_local(exam_value)
            if not exam:
                return jsonify({"error": f"❌ Exam file '{exam_value}' not found"})
            flat        = flatten_exam(exam)
            memo_merged = exam.get("memo_merged", False)
            if not flat:
                return jsonify({"error": "❌ No questions found in exam file"})

        mem.ensure_student(student_id)
        sid = str(uuid.uuid4())
        sessions[sid] = {
            "exam":       exam_label,
            "student_id": student_id,
            "questions":  flat,
            "answers":    {},
        }

        return jsonify({
            "session_id":      sid,
            "total_questions": len(flat),
            "memo_merged":     memo_merged,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)})


@app.route("/question", methods=["POST"])
def get_question():
    try:
        data    = request.get_json()
        sid     = data.get("session_id")
        idx     = data.get("index", 0)
        session = sessions.get(sid)
        if not session:
            return jsonify({"error": "Invalid session"})
        flat = session["questions"]
        if idx < 0 or idx >= len(flat):
            return jsonify({"error": "Index out of range"})
        q = flat[idx].copy()
        q["saved_answer"] = session["answers"].get(str(idx), "")
        return jsonify(q)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)})


@app.route("/answer", methods=["POST"])
def save_answer():
    try:
        data    = request.get_json()
        sid     = data.get("session_id")
        idx     = data.get("index")
        answer  = data.get("answer", "")
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
        data       = request.get_json()
        sid        = data.get("session_id")
        student_id = data.get("student_id", "anonymous")
        session    = sessions.get(sid)
        if not session:
            return jsonify({"error": "Invalid session"})

        exam_name = session["exam"]
        flat      = session["questions"]
        answers   = session["answers"]

        results     = []
        total_score = 0
        total_marks = 0

        for i, q in enumerate(flat):
            q_num   = q.get("question_number", f"Q{i+1}")
            q_type  = q.get("type", "open").lower()
            marks   = int(q.get("marks", 1))
            q_text  = q.get("question", "")
            memo    = q.get("memo", "")
            student = answers.get(str(i), "").strip()
            options = q.get("options")

            result = mark_answer(
                question=q_text, question_number=q_num, q_type=q_type,
                student_answer=student, memo=memo, marks=marks, options=options,
            )

            if result.get("status") in ("incorrect", "missing"):
                topic = q.get("parent_question", "").split(":")[1].strip() if ":" in q.get("parent_question", "") else ""
                mem.record_wrong(student_id, q_num, q_text, q_type, topic)
            elif result.get("status") == "correct":
                mem.record_correct(student_id, q_num)

            if isinstance(memo, dict) and memo:
                correct_display = " | ".join(f"{k.split()[0]} → {v}" for k, v in memo.items())
            elif memo:
                if q_type == "mcq" and options:
                    cl = str(memo).strip().upper()
                    correct_display = cl
                    for opt in options:
                        if isinstance(opt, dict) and opt.get("key", "").upper() == cl:
                            correct_display = f"{cl}. {opt['value']}"
                            break
                else:
                    correct_display = str(memo)
            else:
                correct_display = "Not available"

            result["question_number"] = q_num
            result["question"]        = q_text
            result["type"]            = q_type
            result["marks"]           = marks
            result["student_answer"]  = student or "No answer"
            result["correct_answer"]  = correct_display
            result["earned"]          = result.get("score", 0)

            results.append(result)
            total_score += result["earned"]
            total_marks += marks

        percentage = round((total_score / total_marks * 100), 1) if total_marks else 0
        mem.save_session(student_id, exam_name, total_score, total_marks, percentage)
        feedback = generate_exam_feedback(results, total_score, total_marks, percentage)

        weak = mem.get_weak_topics(student_id)
        if weak:
            try:
                run_agent(
                    student_id,
                    f"I just scored {percentage}% on {exam_name}. "
                    f"Please update my study plan based on my weak areas.",
                    rag=rag,
                )
            except Exception:
                pass

        return jsonify({
            "score": total_score, "total": total_marks,
            "percentage": percentage, "results": results, "feedback": feedback,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)})


@app.route("/dashboard", methods=["POST"])
def dashboard():
    try:
        data       = request.get_json()
        student_id = data.get("student_id", "anonymous")
        mem.ensure_student(student_id)
        return jsonify({
            "weak":       mem.get_weak_topics(student_id),
            "sessions":   mem.get_sessions(student_id, limit=8),
            "study_plan": mem.get_study_plan(student_id),
        })
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True, port=8000)
