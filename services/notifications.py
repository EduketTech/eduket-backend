"""
services/notifications.py
Handles user onboarding emails and principal approval alerts.
Uses urllib (built-in) instead of requests/httpx/resend SDK
to avoid gevent recursion errors in gunicorn gevent workers.
"""

import os
import json
import urllib.request
import urllib.error
from flask import request, jsonify
from firebase_admin import firestore as fs_admin


# ── Shared email sender — uses urllib to avoid gevent recursion ───────────

def _send_email(to: str, subject: str, html: str,
                from_addr: str = "Eduket OS <onboarding@resend.dev>") -> dict:
    """
    Send email via Resend REST API using Python's built-in urllib.
    Does NOT use requests, httpx, or the resend SDK — all three cause
    'maximum recursion depth exceeded' when running inside gunicorn
    gevent workers because gevent patches their socket layer.
    urllib is unaffected.
    """
    resend_key = os.getenv("RESEND_API_KEY", "").strip()
    if not resend_key:
        print("[Email] RESEND_API_KEY not set — skipping")
        return {"success": False, "error": "RESEND_API_KEY not set"}

    payload = json.dumps({
        "from":    from_addr,
        "to":      [to],
        "subject": subject,
        "html":    html,
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.resend.com/emails",
        data=payload,
        headers={
            "Authorization": f"Bearer {resend_key}",
            "Content-Type":  "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode())
            print(f"[Email] Sent to {to} — id: {result.get('id')}")
            return {"success": True, "id": result.get("id")}
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"[Email] HTTP error {e.code}: {body}")
        return {"success": False, "error": f"HTTP {e.code}: {body}"}
    except Exception as e:
        print(f"[Email] Failed: {e}")
        return {"success": False, "error": str(e)}


# ── Welcome email — sent to new user on registration ─────────────────────

def send_welcome_email_handler(data=None):
    if data is None:

        try:
            data        = request.get_json() or {}
            email       = data.get("email",       "").strip()
            name        = data.get("displayName") or data.get("firstName", "")
            role        = data.get("role",        "student")
            school_name = data.get("schoolName",  "")
            subjects    = data.get("subjects",    [])
            grade       = data.get("grade",       "")
            dashboard   = data.get("dashboardUrl", "https://eduket.tech")

            if not email:
                return jsonify({"error": "Email required"}), 400

            role_config = {
                "principal": {
                    "colour":   "#7c3aed",
                    "icon":     "🏫",
                    "subtitle": "Your school is live on Eduket OS!",
                    "body":     f"Your school <strong>{school_name}</strong> has been successfully registered. You can now invite teachers and students.",
                    "btn":      "Go to Principal Dashboard",
                },
                "teacher": {
                    "colour":   "#059669",
                    "icon":     "📚",
                    "subtitle": "Welcome, Teacher!",
                    "body":     f"You have been set up as a teacher at <strong>{school_name}</strong>. Start by uploading your first exam.",
                    "btn":      "Go to Teacher Dashboard",
                },
                "student": {
                    "colour":   "#1d4ed8",
                    "icon":     "🎓",
                    "subtitle": "Welcome to Eduket OS!",
                    "body":     f"You are enrolled at <strong>{school_name}</strong>{f', Grade {grade}' if grade else ''}. Your exams will appear when teachers upload them.",
                    "btn":      "Go to My Exams",
                },
            }
            cfg = role_config.get(role, role_config["student"])

            # Build details rows
            rows = ""
            if name:
                rows += f"<tr><td style='padding:7px 0;color:#6b7280;font-size:13px;width:35%'>Name</td><td style='padding:7px 0;font-weight:700;font-size:13px'>{name}</td></tr>"
            if email:
                rows += f"<tr><td style='padding:7px 0;color:#6b7280;font-size:13px'>Email</td><td style='padding:7px 0;font-weight:700;font-size:13px'>{email}</td></tr>"
            if role:
                rows += f"<tr><td style='padding:7px 0;color:#6b7280;font-size:13px'>Role</td><td style='padding:7px 0;font-weight:700;font-size:13px;text-transform:capitalize'>{role}</td></tr>"
            if school_name:
                rows += f"<tr><td style='padding:7px 0;color:#6b7280;font-size:13px'>School</td><td style='padding:7px 0;font-weight:700;font-size:13px'>{school_name}</td></tr>"
            if grade:
                rows += f"<tr><td style='padding:7px 0;color:#6b7280;font-size:13px'>Grade</td><td style='padding:7px 0;font-weight:700;font-size:13px'>{grade}</td></tr>"
            if subjects and isinstance(subjects, list):
                rows += f"<tr><td style='padding:7px 0;color:#6b7280;font-size:13px'>Subjects</td><td style='padding:7px 0;font-weight:700;font-size:13px'>{', '.join(subjects)}</td></tr>"

            html = f"""<!DOCTYPE html>
    <html>
    <head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/></head>
    <body style="margin:0;padding:0;background:#f1f5f9;font-family:'Segoe UI',Arial,sans-serif;">
    <table width="100%" cellpadding="0" cellspacing="0" style="background:#f1f5f9;padding:40px 16px;">
    <tr><td align="center">
    <table width="100%" cellpadding="0" cellspacing="0"
           style="max-width:560px;background:#fff;border-radius:20px;
                  overflow:hidden;box-shadow:0 4px 32px rgba(0,0,0,0.08);">
      <tr>
        <td style="background:linear-gradient(135deg,{cfg['colour']},{cfg['colour']}cc);
                   padding:36px 32px;text-align:center;">
          <p style="margin:0 0 8px;font-size:32px;">{cfg['icon']}</p>
          <h1 style="margin:0;font-size:24px;font-weight:900;color:#fff;">Eduket OS</h1>
          <p style="margin:8px 0 0;font-size:13px;color:rgba(255,255,255,0.8);">
            AI-Powered School Assessment Platform
          </p>
        </td>
      </tr>
      <tr>
        <td style="padding:32px;">
          <p style="margin:0 0 4px;font-size:11px;font-weight:700;
                     color:{cfg['colour']};text-transform:uppercase;letter-spacing:1px;">
            {cfg['subtitle']}
          </p>
          <p style="margin:8px 0 20px;font-size:14px;color:#374151;line-height:1.7;">
            Hi <strong>{name or 'there'}</strong>, {cfg['body']}
          </p>
          <div style="background:#f8fafc;border-radius:12px;padding:16px 20px;
                      border:1px solid #e2e8f0;margin-bottom:28px;">
            <p style="margin:0 0 10px;font-size:10px;font-weight:700;
                       color:#94a3b8;text-transform:uppercase;letter-spacing:1px;">
              Your Registration Details
            </p>
            <table width="100%" cellpadding="0" cellspacing="0">{rows}</table>
          </div>
          <table width="100%" cellpadding="0" cellspacing="0">
            <tr><td align="center">
              <a href="{dashboard}"
                 style="display:inline-block;padding:14px 36px;
                        background:{cfg['colour']};color:#fff;font-size:14px;
                        font-weight:900;text-decoration:none;border-radius:12px;">
                {cfg['btn']} →
              </a>
            </td></tr>
          </table>
          <p style="margin:24px 0 0;font-size:12px;color:#94a3b8;text-align:center;">
            If you did not create this account, please ignore this email.
          </p>
        </td>
      </tr>
      <tr>
        <td style="background:#f8fafc;padding:16px 32px;
                   border-top:1px solid #e2e8f0;text-align:center;">
          <p style="margin:0;font-size:11px;color:#94a3b8;">
            © 2026 Nextgen Skills · Eduket OS · eduket.tech
          </p>
        </td>
      </tr>
    </table>
    </td></tr>
    </table>
    </body>
    </html>"""

            result = _send_email(
                to=email,
                subject=f"{cfg['icon']} {cfg['subtitle']}",
                html=html,
                from_addr="Eduket OS <onboarding@resend.dev>",
            )
            return jsonify(result), 200

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[Welcome Email] Error: {e}")
            return jsonify({"success": False, "error": str(e)}), 200


# ── Principal notification — sent when teacher/student joins ──────────────

def notify_principal_signup_handler(db, data=None):
    # data is passed directly when called from background thread
    # (request context not available in threads)
    if data is None:
        try:
            data = request.get_json() or {}
        except Exception:
            data = {}

    try:
        data        = request.get_json() or {}
        school_id   = data.get("schoolId",   "").strip()
        new_email   = data.get("email",      "").strip()
        new_name    = data.get("displayName") or data.get("firstName", "New User")
        new_role    = data.get("role",       "student")
        school_name = data.get("schoolName", "Your School")
        grade       = data.get("grade",      "")
        subjects    = data.get("subjects",   [])
        uid         = data.get("uid",        "")

        if not school_id or not new_email:
            return jsonify({"error": "schoolId and email required"}), 400

        # 1. Log activity to Firestore
        db.collection("schoolActivity").document().set({
            "schoolId":    school_id,
            "schoolName":  school_name,
            "type":        "user_joined",
            "actorUid":    uid,
            "actorName":   new_name,
            "actorEmail":  new_email,
            "actorRole":   new_role,
            "grade":       grade,
            "subjects":    subjects if isinstance(subjects, list) else [],
            "description": f"{new_name} joined as {new_role}",
            "timestamp":   fs_admin.SERVER_TIMESTAMP,
            "read":        False,
        })

        # 2. Get principal details
        school_doc = db.collection("schools").document(school_id).get()
        if not school_doc.exists:
            return jsonify({"error": "School not found"}), 404

        principal_uid = school_doc.to_dict().get("principalUid", "")
        if not principal_uid:
            return jsonify({"success": False, "reason": "No principal linked"}), 200

        # Try users collection first, fallback to principals
        principal_doc = db.collection("users").document(principal_uid).get()
        if not principal_doc.exists:
            principal_doc = db.collection("principals").document(principal_uid).get()

        if not principal_doc.exists:
            return jsonify({"success": False, "reason": "Principal not found"}), 200

        principal_data  = principal_doc.to_dict()
        principal_email = principal_data.get("email", "")
        principal_name  = principal_data.get("firstName", "Principal")

        if not principal_email:
            return jsonify({"success": False, "reason": "Principal has no email"}), 200

        # 3. Build and send email
        role_colour = {
            "teacher":   "#059669",
            "student":   "#1d4ed8",
            "principal": "#7c3aed",
        }.get(new_role, "#6b7280")

        role_icon = {"teacher": "👩‍🏫", "student": "🎓"}.get(new_role, "👤")

        details_rows = f"""
<tr>
  <td style="padding:7px 0;color:#6b7280;font-size:13px;width:35%">Name</td>
  <td style="padding:7px 0;font-weight:700;font-size:13px">{new_name}</td>
</tr>
<tr>
  <td style="padding:7px 0;color:#6b7280;font-size:13px">Email</td>
  <td style="padding:7px 0;font-weight:700;font-size:13px">{new_email}</td>
</tr>
<tr>
  <td style="padding:7px 0;color:#6b7280;font-size:13px">Role</td>
  <td style="padding:7px 0;font-weight:700;font-size:13px;
             text-transform:capitalize;color:{role_colour}">{new_role}</td>
</tr>"""

        if grade:
            details_rows += f"""
<tr>
  <td style="padding:7px 0;color:#6b7280;font-size:13px">Grade</td>
  <td style="padding:7px 0;font-weight:700;font-size:13px">{grade}</td>
</tr>"""

        if subjects and isinstance(subjects, list) and len(subjects) > 0:
            details_rows += f"""
<tr>
  <td style="padding:7px 0;color:#6b7280;font-size:13px">Subjects</td>
  <td style="padding:7px 0;font-weight:700;font-size:13px">{', '.join(subjects)}</td>
</tr>"""

        from datetime import datetime
        timestamp = datetime.utcnow().strftime("%d %B %Y at %H:%M UTC")

        # URL-encode for mailto link
        mailto_subject = f"Unknown user registered: {new_email}"
        mailto_body    = f"Principal: {principal_name}%0ASchool: {school_name}%0AUnknown user: {new_name} ({new_email})"

        html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/></head>
<body style="margin:0;padding:0;background:#f1f5f9;font-family:'Segoe UI',Arial,sans-serif;">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#f1f5f9;padding:40px 16px;">
<tr><td align="center">
<table width="100%" cellpadding="0" cellspacing="0"
       style="max-width:580px;background:#fff;border-radius:20px;
              overflow:hidden;box-shadow:0 4px 32px rgba(0,0,0,0.08);">
  <tr>
    <td style="background:linear-gradient(135deg,#1e293b,#334155);
               padding:28px 32px;text-align:center;">
      <h1 style="margin:0;font-size:22px;font-weight:900;color:#fff;">
        Eduket OS · School Alert
      </h1>
      <p style="margin:6px 0 0;font-size:12px;color:rgba(255,255,255,0.6);">
        {school_name}
      </p>
    </td>
  </tr>
  <tr>
    <td style="background:{role_colour}18;border-bottom:3px solid {role_colour};
               padding:14px 32px;">
      <p style="margin:0;font-size:14px;font-weight:700;color:{role_colour};">
        {role_icon} New {new_role.title()} joined your school
      </p>
      <p style="margin:4px 0 0;font-size:12px;color:#6b7280;">
        Registered on {timestamp}
      </p>
    </td>
  </tr>
  <tr>
    <td style="padding:28px 32px;">
      <p style="margin:0 0 20px;font-size:14px;color:#374151;line-height:1.7;">
        Hi <strong>{principal_name}</strong>, a new <strong>{new_role}</strong>
        has completed registration for <strong>{school_name}</strong>.
      </p>
      <div style="background:#f8fafc;border-radius:12px;padding:16px 20px;
                  border:1px solid #e2e8f0;margin-bottom:20px;">
        <p style="margin:0 0 10px;font-size:10px;font-weight:700;
                   color:#94a3b8;text-transform:uppercase;letter-spacing:1px;">
          New User Details
        </p>
        <table width="100%" cellpadding="0" cellspacing="0">
          {details_rows}
        </table>
      </div>
      <div style="background:#fef3c7;border:1px solid #fcd34d;
                  border-radius:12px;padding:14px 18px;margin-bottom:24px;">
        <p style="margin:0 0 6px;font-size:12px;font-weight:700;color:#92400e;">
          ⚠️ Do you recognise this person?
        </p>
        <p style="margin:0;font-size:12px;color:#92400e;line-height:1.6;">
          If you do not recognise <strong>{new_name} ({new_email})</strong>
          as a member of {school_name}, contact Eduket OS support immediately.
        </p>
      </div>
      <table width="100%" cellpadding="0" cellspacing="0">
        <tr>
          <td style="padding-right:6px;">
            <a href="https://eduket.tech/principal-dashboard"
               style="display:block;text-align:center;padding:13px 0;
                      background:#7c3aed;color:#fff;font-size:13px;
                      font-weight:900;text-decoration:none;border-radius:10px;">
              View Dashboard →
            </a>
          </td>
          <td style="padding-left:6px;">
            <a href="mailto:support@eduket.tech?subject={mailto_subject}&body={mailto_body}"
               style="display:block;text-align:center;padding:13px 0;
                      background:#dc2626;color:#fff;font-size:13px;
                      font-weight:900;text-decoration:none;border-radius:10px;">
              🚨 Report Unknown User
            </a>
          </td>
        </tr>
      </table>
    </td>
  </tr>
  <tr>
    <td style="background:#f8fafc;padding:16px 32px;
               border-top:1px solid #e2e8f0;text-align:center;">
      <p style="margin:0;font-size:11px;color:#94a3b8;">
        Eduket OS · eduket.tech · support@eduket.tech
      </p>
    </td>
  </tr>
</table>
</td></tr>
</table>
</body>
</html>"""

        result = _send_email(
            to=principal_email,
            subject=f"{role_icon} {new_name} joined {school_name} as {new_role}",
            html=html,
            from_addr="Eduket OS Alerts <onboarding@resend.dev>",
        )

        print(f"[Notify] Principal alert {'sent' if result.get('success') else 'failed'}: {principal_email}")
        return jsonify({"success": result.get("success", False)})

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[Notify] Error: {e}")
        return jsonify({"success": False, "error": str(e)}), 200


# ── School activity feed ──────────────────────────────────────────────────

def get_school_activity_handler(db):
    """GET /school-activity?schoolId=xxx"""
    try:
        from google.cloud.firestore_v1.base_query import FieldFilter
        school_id = request.args.get("schoolId", "")
        if not school_id:
            return jsonify({"error": "schoolId required"}), 400

        events = []
        try:
            docs = (
                db.collection("schoolActivity")
                  .where(filter=FieldFilter("schoolId", "==", school_id))
                  .order_by("timestamp", direction="DESCENDING")
                  .limit(50)
                  .stream()
            )
            for doc in docs:
                d  = doc.to_dict()
                ts = d.get("timestamp")
                events.append({
                    "id":          doc.id,
                    "type":        d.get("type",        ""),
                    "actorName":   d.get("actorName",   ""),
                    "actorEmail":  d.get("actorEmail",  ""),
                    "actorRole":   d.get("actorRole",   ""),
                    "description": d.get("description", ""),
                    "grade":       d.get("grade",       ""),
                    "subjects":    d.get("subjects",    []),
                    "timestamp":   ts.isoformat() if hasattr(ts, "isoformat") else str(ts or ""),
                    "read":        d.get("read", False),
                })
        except Exception as idx_err:
            if "requires an index" in str(idx_err):
                print(f"[Activity] Index building: {idx_err}")
                return jsonify({"events": [], "indexBuilding": True})
            raise

        return jsonify({"events": events})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def mark_activity_read_handler(db):
    """POST /school-activity/mark-read"""
    try:
        data      = request.get_json() or {}
        event_ids = data.get("eventIds", [])
        if not event_ids:
            return jsonify({"success": True})
        batch = db.batch()
        for eid in event_ids:
            batch.update(db.collection("schoolActivity").document(eid), {"read": True})
        batch.commit()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500