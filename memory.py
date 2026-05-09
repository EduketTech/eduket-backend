"""
memory.py  —  Persistent student memory  (LangChain migration)

WHAT CHANGED
────────────
1. A new table  'lc_message_store' is created alongside the existing tables.
   LangChain's SQLChatMessageHistory writes to this table automatically.
   The table schema follows LangChain's convention (session_id, message columns)
   and is managed entirely by LangChain — we do not touch it directly.

2. get_history() and append_message() are KEPT because app.py's /dashboard
   and the old home-UI still reference them.  They now read from the original
   'conversation_history' table which remains intact.

3. Everything else (students, sessions, weak_questions, study_plan tables,
   and all helper functions) is completely unchanged.  No existing callers
   need to be modified.

WHAT DID NOT CHANGE
───────────────────
- DB_PATH, _conn(), init_db(), ensure_student()
- save_session(), get_sessions()
- record_wrong(), record_correct(), get_weak_topics(), get_weak_summary()
- append_message(), get_history(), clear_history()
- save_study_plan(), get_study_plan()
"""

import sqlite3
import json
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "student_memory.db")


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    with _conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS students (
            student_id  TEXT PRIMARY KEY,
            created_at  TEXT DEFAULT (datetime('now')),
            name        TEXT DEFAULT 'Student'
        );

        CREATE TABLE IF NOT EXISTS sessions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id  TEXT,
            exam_name   TEXT,
            score       INTEGER,
            total       INTEGER,
            percentage  REAL,
            subject     TEXT DEFAULT '',
            played_at   TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS weak_questions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id      TEXT,
            question_number TEXT,
            question_text   TEXT,
            q_type          TEXT,
            topic           TEXT,
            wrong_count     INTEGER DEFAULT 1,
            last_seen       TEXT DEFAULT (datetime('now')),
            UNIQUE(student_id, question_number)
        );

        CREATE TABLE IF NOT EXISTS conversation_history (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id   TEXT,
            role         TEXT,
            content      TEXT,
            tool_call_id TEXT,
            tool_name    TEXT,
            created_at   TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS study_plan (
            student_id  TEXT PRIMARY KEY,
            plan        TEXT,
            updated_at  TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS lc_message_store (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            message    TEXT NOT NULL
        );
        """)

        # Migration — adds subject to existing databases that already
        # have the sessions table without the column. Safe to run every
        # startup; silently ignored if the column already exists.
        try:
            c.execute("ALTER TABLE sessions ADD COLUMN subject TEXT DEFAULT ''")
        except Exception:
            pass  # column already exists — nothing to do


# ── Student ───────────────────────────────────────────────────────────────────
def ensure_student(student_id: str, name: str = "Student"):
    with _conn() as c:
        c.execute(
            "INSERT OR IGNORE INTO students(student_id, name) VALUES(?,?)",
            (student_id, name),
        )


# ── Session history ───────────────────────────────────────────────────────────
def save_session(student_id, exam, score, total, percentage, subject=""):
    with _conn() as c:
        c.execute(
            "INSERT INTO sessions (student_id, exam_name, score, total, percentage, subject) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (student_id, exam, score, total, percentage, subject)
        )



def get_sessions(student_id: str, limit: int = 5) -> list:
    with _conn() as c:
        rows = c.execute(
            "SELECT exam_name,score,total,percentage,played_at FROM sessions "
            "WHERE student_id=? ORDER BY played_at DESC LIMIT ?",
            (student_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


# ── Weak question tracking ────────────────────────────────────────────────────
def record_wrong(student_id: str, question_number: str, question_text: str, q_type: str, topic: str = ""):
    with _conn() as c:
        c.execute(
            """
            INSERT INTO weak_questions
                (student_id,question_number,question_text,q_type,topic,wrong_count,last_seen)
            VALUES(?,?,?,?,?,1,datetime('now'))
            ON CONFLICT(student_id,question_number) DO UPDATE SET
                wrong_count   = wrong_count + 1,
                last_seen     = datetime('now'),
                question_text = excluded.question_text
            """,
            (student_id, question_number, question_text, q_type, topic),
        )


def record_correct(student_id: str, question_number: str):
    """Reduce wrong_count by 1 when the student answers correctly (floor 0)."""
    with _conn() as c:
        c.execute(
            """
            UPDATE weak_questions
            SET wrong_count = MAX(0, wrong_count - 1)
            WHERE student_id=? AND question_number=?
            """,
            (student_id, question_number),
        )


def get_weak_topics(student_id: str, limit: int = 10) -> list:
    with _conn() as c:
        rows = c.execute(
            """
            SELECT question_number, question_text, q_type, topic, wrong_count, last_seen
            FROM weak_questions
            WHERE student_id=? AND wrong_count > 0
            ORDER BY wrong_count DESC, last_seen DESC
            LIMIT ?
            """,
            (student_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def get_weak_summary(student_id: str) -> str:
    """Returns a short text summary of weak areas for agent context injection."""
    weak = get_weak_topics(student_id)
    if not weak:
        return "No weak areas recorded yet."
    lines = [
        f"Q{w['question_number']} ({w['q_type']}): wrong {w['wrong_count']}x"
        for w in weak
    ]
    return "Weak questions: " + ", ".join(lines)


# ── Conversation history (original table — kept for dashboard + legacy UI) ────
def append_message(student_id: str, role: str, content, tool_call_id=None, tool_name=None):
    """
    Write to the original conversation_history table.
    The LangChain agent writes to lc_message_store instead, but the
    dashboard and the home-UI still use this table for display.
    """
    with _conn() as c:
        c.execute(
            "INSERT INTO conversation_history"
            "(student_id,role,content,tool_call_id,tool_name) VALUES(?,?,?,?,?)",
            (
                student_id,
                role,
                content if isinstance(content, str) else json.dumps(content),
                tool_call_id,
                tool_name,
            ),
        )


def get_history(student_id: str, limit: int = 20) -> list:
    """
    Returns last N messages from the original conversation_history table
    as a list of dicts suitable for display in the dashboard / home-UI.

    Note: the LangChain agent reads from lc_message_store via
    SQLChatMessageHistory — not from this function.
    """
    with _conn() as c:
        rows = c.execute(
            "SELECT role,content,tool_call_id,tool_name FROM conversation_history "
            "WHERE student_id=? ORDER BY id DESC LIMIT ?",
            (student_id, limit),
        ).fetchall()

    messages = []
    for r in reversed(rows):
        msg = {"role": r["role"]}
        try:
            msg["content"] = json.loads(r["content"])
        except Exception:
            msg["content"] = r["content"]
        if r["tool_call_id"]:
            msg["tool_call_id"] = r["tool_call_id"]
        if r["tool_name"]:
            msg["name"] = r["tool_name"]
        messages.append(msg)
    return messages


def clear_history(student_id: str):
    """
    Clears both the original conversation_history table AND the
    LangChain lc_message_store table for this student.
    Called by app.py /clear-history endpoint.
    """
    with _conn() as c:
        c.execute("DELETE FROM conversation_history WHERE student_id=?", (student_id,))
        # Also clear LangChain's history so the agent starts fresh
        c.execute("DELETE FROM lc_message_store WHERE session_id=?",   (student_id,))


# ── Study plan ────────────────────────────────────────────────────────────────
def save_study_plan(student_id: str, plan_text: str):
    with _conn() as c:
        c.execute(
            """
            INSERT INTO study_plan(student_id, plan, updated_at)
            VALUES(?,?,datetime('now'))
            ON CONFLICT(student_id) DO UPDATE SET
                plan       = excluded.plan,
                updated_at = datetime('now')
            """,
            (student_id, plan_text),
        )


def get_study_plan(student_id: str) -> dict | None:
    with _conn() as c:
        row = c.execute(
            "SELECT plan, updated_at FROM study_plan WHERE student_id=?",
            (student_id,),
        ).fetchone()
    return dict(row) if row else None


# Init on import — creates all tables including lc_message_store
init_db()