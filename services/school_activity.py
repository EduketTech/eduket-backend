"""
services/school_activity.py
Handles fetching and updating principal activity audit logs.
"""

import traceback
from datetime import datetime, timezone
from flask import request, jsonify
from google.cloud.firestore_v1.base_query import FieldFilter


def get_school_activity_handler(db):
    """GET /school-activity?schoolId=xxx"""
    if request.method == "OPTIONS":
        return "", 204

    try:
        school_id = request.args.get("schoolId", "").strip()
        if not school_id:
            return jsonify({"error": "schoolId required"}), 400

        # ── Try ordered query (requires composite index) ───────────────────
        try:
            docs = (
                db.collection("schoolActivity")
                  .where(filter=FieldFilter("schoolId", "==", school_id))
                  .order_by("timestamp", direction="DESCENDING")
                  .limit(50)
                  .stream()
            )
            raw_docs = list(docs)

        except Exception as index_err:
            err_str = str(index_err)
            if "requires an index" in err_str or "FAILED_PRECONDITION" in err_str:
                # Index still building — fall back to unordered query + client sort
                print(f"[Activity] Index not ready — using fallback sort: {err_str[:80]}")
                raw_docs = list(
                    db.collection("schoolActivity")
                      .where(filter=FieldFilter("schoolId", "==", school_id))
                      .limit(100)
                      .stream()
                )

                def _sort_key(doc):
                    ts = doc.to_dict().get("timestamp")
                    if ts is None:
                        return datetime(1970, 1, 1, tzinfo=timezone.utc)
                    if hasattr(ts, 'tzinfo') and ts.tzinfo is None:
                        return ts.replace(tzinfo=timezone.utc)
                    return ts

                raw_docs.sort(key=_sort_key, reverse=True)
                raw_docs = raw_docs[:50]
            else:
                raise

        # ── Build response ─────────────────────────────────────────────────
        events = []
        for doc in raw_docs:
            d  = doc.to_dict()
            ts = d.get("timestamp")

            try:
                ts_formatted = ts.isoformat() if hasattr(ts, "isoformat") else str(ts or "")
            except Exception:
                ts_formatted = ""

            events.append({
                "id":             doc.id,
                "type":           d.get("type",           ""),
                "actorUid":       d.get("actorUid",       ""),
                "actorName":      d.get("actorName",      ""),
                "actorEmail":     d.get("actorEmail",     ""),
                "actorRole":      d.get("actorRole",      ""),
                "description":    d.get("description",    ""),
                "grade":          d.get("grade",          ""),
                "subjects":       d.get("subjects",       []),
                "timestamp":      ts_formatted,
                "read":           d.get("read",           False),
                "approvalStatus": d.get("approvalStatus", None),
                "approvedBy":     d.get("approvedBy",     None),
                "declineReason":  d.get("declineReason",  None),
                "schoolName":     d.get("schoolName",     ""),
            })

        return jsonify({"events": events, "success": True})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def mark_activity_read_handler(db):
    """POST /school-activity/mark-read"""
    if request.method == "OPTIONS":
        return "", 204

    try:
        data      = request.get_json() or {}
        event_ids = data.get("eventIds", [])

        if not event_ids or not isinstance(event_ids, list):
            return jsonify({"success": True, "updated": 0})

        CHUNK_SIZE    = 500
        total_updated = 0

        for i in range(0, len(event_ids), CHUNK_SIZE):
            chunk = event_ids[i:i + CHUNK_SIZE]
            batch = db.batch()
            for eid in chunk:
                if eid and isinstance(eid, str):
                    batch.update(
                        db.collection("schoolActivity").document(eid),
                        {"read": True}
                    )
                    total_updated += 1
            batch.commit()

        return jsonify({"success": True, "updated": total_updated})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def approve_school_user_handler(db, req):
    """POST /approve-school-user"""
    if req.method == "OPTIONS":
        return "", 204

    try:
        data           = req.get_json() or {}
        activity_id    = data.get("activityId",    "").strip()
        actor_uid      = data.get("actorUid",      "").strip()
        actor_email    = data.get("actorEmail",    "").strip()
        actor_name     = data.get("actorName",     "")
        actor_role     = data.get("actorRole",     "student")
        school_id      = data.get("schoolId",      "").strip()
        action         = data.get("action",        "")
        decline_reason = data.get("declineReason", "").strip()

        if not activity_id or not school_id or action not in ("approved", "declined"):
            return jsonify({"error": "activityId, schoolId and valid action required"}), 400

        # ── Get principal name from token ──────────────────────────────────
        principal_name = "Principal"
        auth_header    = req.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            try:
                from firebase_admin import auth as fb_auth
                token   = auth_header.split("Bearer ", 1)[1].strip()
                decoded = fb_auth.verify_id_token(token)
                p_uid   = decoded.get("uid", "")
                if p_uid:
                    p_doc = db.collection("users").document(p_uid).get()
                    if p_doc.exists:
                        pd = p_doc.to_dict()
                        principal_name = (
                            pd.get("displayName") or
                            f"{pd.get('firstName', '')} {pd.get('lastName', '')}".strip() or
                            "Principal"
                        )
            except Exception:
                pass  # non-fatal

        # ── Update original activity document ──────────────────────────────
        from firebase_admin import firestore as fs_admin

        update_data = {
            "approvalStatus": action,
            "approvedBy":     principal_name,
            "approvedAt":     fs_admin.SERVER_TIMESTAMP,
            "read":           True,
        }
        if action == "declined" and decline_reason:
            update_data["declineReason"] = decline_reason

        db.collection("schoolActivity").document(activity_id).update(update_data)

        # ── Log the principal's action as a new activity entry ─────────────
        db.collection("schoolActivity").add({
            "schoolId":      school_id,
            "type":          f"user_{action}",
            "actorName":     principal_name,
            "targetName":    actor_name,
            "targetEmail":   actor_email,
            "targetRole":    actor_role,
            "description":   f"{principal_name} {action} {actor_name} as {actor_role}",
            "declineReason": decline_reason if action == "declined" else None,
            "timestamp":     fs_admin.SERVER_TIMESTAMP,
            "read":          True,
        })

        # ── Update user's approval status ──────────────────────────────────
        if actor_uid:
            role_col = (
                "teachers" if actor_role == "teacher" else
                "students" if actor_role == "student" else
                "users"
            )
            try:
                db.collection(role_col).document(actor_uid).update({
                    "approvalStatus": action,
                    "approvedBy":     principal_name,
                    "approvedAt":     fs_admin.SERVER_TIMESTAMP,
                })
                db.collection("users").document(actor_uid).update({
                    "approvalStatus": action,
                })
            except Exception as e:
                print(f"[Approve] Could not update user doc (non-fatal): {e}")

        print(f"[Approve] {principal_name} {action} {actor_email}")
        return jsonify({"success": True, "action": action})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "success": False}), 500