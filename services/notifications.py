"""
services/notifications.py
Handles user onboarding emails and principal approval alerts via Resend.
"""

import os
import resend
from flask import request, jsonify
import firebase_admin.firestore as fs_admin

def notify_principal_signup_handler(db):
    if request.method == "OPTIONS":
        return "", 200

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

        # 1. Log activity record to Firestore
        activity_ref = db.collection("schoolActivity").document()
        activity_ref.set({
            "schoolId":    school_id,
            "schoolName":  school_name,
            "type":        "user_joined_pending",
            "actorUid":    uid,
            "actorName":   new_name,
            "actorEmail":  new_email,
            "actorRole":   new_role,
            "grade":       grade,
            "subjects":    subjects if isinstance(subjects, list) else [],
            "description": f"{new_name} requested approval as {new_role}",
            "timestamp":   fs_admin.SERVER_TIMESTAMP,
            "read":        False,
        })

        # 2. Fetch principal details
        school_doc = db.collection("schools").document(school_id).get()
        if not school_doc.exists:
            return jsonify({"error": "School not found"}), 404

        principal_uid = school_doc.to_dict().get("principalUid", "")
        if not principal_uid:
            return jsonify({"success": False, "reason": "No principal linked"}), 200

        # Try fetching from 'users' first, fallback to 'teachers' if not found
        principal_doc = db.collection("users").document(principal_uid).get()
        if not principal_doc.exists:
            principal_doc = db.collection("teachers").document(principal_uid).get()

        if not principal_doc.exists:
            return jsonify({"success": False, "reason": "Principal user document not found"}), 200

        principal_data  = principal_doc.to_dict() or {}
        principal_email = principal_data.get("email", "")
        principal_name  = principal_data.get("firstName") or principal_data.get("displayName") or "Principal"

        if not principal_email:
            return jsonify({"success": False, "reason": "Principal has no email"}), 200

        resend_key = os.getenv("RESEND_API_KEY")
        if not resend_key:
            return jsonify({"success": False, "reason": "RESEND_API_KEY not set"}), 200

        # 3. Construct HTML email
        html = f"""<!DOCTYPE html><html><body>
        <h2>Pending Approval Request</h2>
        <p>Hi <strong>{principal_name}</strong>,</p>
        <p><strong>{new_name}</strong> ({new_email}) has registered as a <strong>{new_role}</strong> at {school_name}.</p>
        <p><a href="https://eduket.tech/principal-dashboard" style="background:#059669;color:#fff;padding:10px 18px;text-decoration:none;border-radius:6px;display:inline-block;">Review in Dashboard →</a></p>
        </body></html>"""

        # 4. Send email via Resend module
        resend.api_key = resend_key
        resend.Emails.send({
            "from": "Eduket OS Alerts <no-reply@eduket.tech>",
            "to": [principal_email],
            "subject": f"⏳ Approval Required: {new_name} requested to join {school_name}",
            "html": html,
        })

        return jsonify({"success": True}), 200

    except Exception as e:
        print(f"[Notify] Principal alert failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 200


def send_welcome_email_handler():
    if request.method == "OPTIONS":
        return "", 200

    try:
        data        = request.get_json() or {}
        email       = data.get("email", "").strip()
        name        = data.get("displayName") or data.get("firstName", "")
        role        = data.get("role", "student")
        school_name = data.get("schoolName", "")
        dashboard   = data.get("dashboardUrl", "https://eduket.tech")

        if not email:
            return jsonify({"error": "Email required"}), 400

        resend_key = os.getenv("RESEND_API_KEY")
        if not resend_key:
            return jsonify({"error": "Email service not configured"}), 503

        # Set API key directly on the imported resend module
        resend.api_key = resend_key

        params = {
            "from": "Eduket OS <no-reply@eduket.tech>",
            "to": [email],
            "subject": "Welcome to Eduket OS!",
            "html": f"<p>Hi {name}, welcome to Eduket OS for {school_name}. Account status: pending principal verification.</p>",
        }

        # Use resend.Emails.send (or resend.emails.send depending on SDK version)
        email_response = resend.Emails.send(params)

        # Safely extract ID without risk of serializing non-primitive objects
        msg_id = getattr(email_response, "id", None) or (
            email_response.get("id") if isinstance(email_response, dict) else str(email_response)
        )

        return jsonify({"success": True, "messageId": msg_id}), 200

    except Exception as e:
        print(f"[Welcome Email] Failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 200