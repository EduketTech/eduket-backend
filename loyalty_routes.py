"""
loyalty_routes.py — Loyalty / trial-bypass code API
═══════════════════════════════════════════════════════════════════════════════
Matches the conventions of billing_routes.py: no blueprint url_prefix (full
paths per route), inline token verification, _db()/_audit() helpers, OPTIONS
handling for CORS preflight, try/except with traceback.print_exc() -> 500.

Register in app.py alongside billing_bp:
    from loyalty_routes import loyalty_bp
    app.register_blueprint(loyalty_bp)
"""

import logging
import traceback

from flask import Blueprint, request, jsonify

from firebase_admin import firestore as fs_admin, auth as fb_auth

from pricing import redeem_loyalty_code, is_loyalty_subscription_active

logger     = logging.getLogger(__name__)
loyalty_bp = Blueprint("loyalty", __name__)


# ══════════════════════════════════════════════════════════════════════════════
# AUTH HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _db():
    return fs_admin.client()

def _verify_token(req) -> tuple:
    """Same shape as billing_routes.py's _verify_token -- duplicated here
    rather than imported so this file has no import-order dependency on
    billing_routes.py. If you'd rather share one copy, move this (and
    _get_school_id_for_uid) into a small shared auth_helpers.py and import
    it from both files instead."""
    header = req.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        return None, (jsonify({"error": "Missing or malformed Authorization header"}), 401)
    try:
        decoded = fb_auth.verify_id_token(header.split("Bearer ", 1)[1].strip())
        return decoded["uid"], None
    except Exception as e:
        logger.warning("[Loyalty Auth] Token verification failed: %s", e)
        return None, (jsonify({"error": "Invalid or expired token"}), 401)

def _get_school_id_for_uid(uid: str):
    try:
        doc = _db().collection("users").document(uid).get()
        return doc.to_dict().get("schoolId") if doc.exists else None
    except Exception as e:
        logger.error("[Loyalty] schoolId lookup failed for %s: %s", uid, e)
        return None

def _audit(action: str, actor: str, target: str, details: dict = None):
    try:
        _db().collection("auditLog").add({
            "action":    action,
            "actorUid":  actor,
            "target":    target,
            "details":   details or {},
            "timestamp": fs_admin.SERVER_TIMESTAMP,
            "ip": request.headers.get(
                "X-Forwarded-For", request.remote_addr or "unknown"
            ).split(",")[0].strip(),
        })
    except Exception as e:
        logger.error("[Audit] Loyalty log failed: %s", e)


# ── Admin check ─────────────────────────────────────────────────────────────
# Matches app.py's require_admin exactly: authentication alone isn't
# enough, the caller's email must have a document in the `admins`
# collection. Duplicated here (rather than imported from app.py) to avoid
# a circular import -- app.py is what imports and registers this blueprint,
# so this module can't import back from it. If that becomes annoying,
# move require_admin's logic into a shared auth_helpers.py both files
# import from instead.

def _verify_admin_token(req) -> tuple:
    uid, err = _verify_token(req)
    if err:
        return None, err
    try:
        user_record = fb_auth.get_user(uid)
        admin_doc = _db().collection("admins").document(user_record.email or "").get()
        if not admin_doc.exists:
            return None, (jsonify({"error": "Admin access required"}), 403)
    except Exception:
        return None, (jsonify({"error": "Admin verification failed"}), 403)
    return uid, None


# ══════════════════════════════════════════════════════════════════════════════
# ROUTE: Redeem a loyalty code (school-scoped, any signed-in user of that school)
# ══════════════════════════════════════════════════════════════════════════════

@loyalty_bp.route("/api/billing/loyalty/redeem", methods=["POST", "OPTIONS"])
def loyalty_redeem():
    if request.method == "OPTIONS":
        return "", 204
    try:
        uid, err = _verify_token(request)
        if err:
            return err

        school_id = _get_school_id_for_uid(uid)
        if not school_id:
            return jsonify({"error": "No school associated with this account"}), 400

        data = request.get_json() or {}
        code = str(data.get("code", "")).strip().upper()
        if not code:
            return jsonify({"error": "Missing loyalty code."}), 400

        result = redeem_loyalty_code(_db(), school_id=school_id, code=code)

        _audit("loyalty_code_redeemed", uid, school_id, {
            "code": code,
            "cycleEnd": result.get("cycle_end"),
        })

        return jsonify(result), 200

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Could not redeem loyalty code."}), 500


# ══════════════════════════════════════════════════════════════════════════════
# ROUTE: Current loyalty status for the caller's school
# ══════════════════════════════════════════════════════════════════════════════

@loyalty_bp.route("/api/billing/loyalty/status", methods=["GET", "OPTIONS"])
def loyalty_status():
    if request.method == "OPTIONS":
        return "", 204
    try:
        uid, err = _verify_token(request)
        if err:
            return err

        school_id = _get_school_id_for_uid(uid)
        if not school_id:
            return jsonify({"active": False, "cycleEnd": None}), 200

        db = _db()
        active = is_loyalty_subscription_active(db, school_id=school_id)

        cycle_end = None
        if active:
            school_snap = db.collection("schools").document(school_id).get()
            loyalty = (school_snap.to_dict() or {}).get("loyaltySubscription") or {}
            cycle_end_dt = loyalty.get("cycleEnd")
            cycle_end = cycle_end_dt.isoformat() if cycle_end_dt else None

        return jsonify({"active": active, "cycleEnd": cycle_end}), 200

    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Could not check loyalty status."}), 500


# ══════════════════════════════════════════════════════════════════════════════
# ROUTE: Admin -- create a code
# ══════════════════════════════════════════════════════════════════════════════

@loyalty_bp.route("/api/billing/loyalty/admin/create", methods=["POST", "OPTIONS"])
def loyalty_admin_create():
    if request.method == "OPTIONS":
        return "", 204
    try:
        admin_uid, err = _verify_admin_token(request)
        if err:
            return err

        data = request.get_json() or {}
        code = str(data.get("code") or "").strip().upper()
        if not code:
            import secrets
            alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # no O/0, I/1
            code = "".join(secrets.choice(alphabet) for _ in range(10))

        db = _db()
        code_ref = db.collection("loyaltyCodes").document(code)
        if code_ref.get().exists:
            return jsonify({"error": f"Code '{code}' already exists."}), 409

        from datetime import datetime, timezone
        code_ref.set({
            "active":            True,
            "maxRedemptions":    data.get("maxRedemptions"),
            "expiresAt":         data.get("expiresAt"),
            "description":       data.get("description", ""),
            "redeemedBySchools": [],
            "createdAt":         datetime.now(timezone.utc),
        })

        _audit("loyalty_code_created", admin_uid, code, {
            "maxRedemptions": data.get("maxRedemptions"),
            "description": data.get("description", ""),
        })

        return jsonify({"code": code, "status": "created"}), 201

    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Could not create loyalty code."}), 500


# ══════════════════════════════════════════════════════════════════════════════
# ROUTE: Admin -- list all codes with redemption state
# ══════════════════════════════════════════════════════════════════════════════

@loyalty_bp.route("/api/billing/loyalty/admin/list", methods=["GET", "OPTIONS"])
def loyalty_admin_list():
    if request.method == "OPTIONS":
        return "", 204
    try:
        _, err = _verify_admin_token(request)
        if err:
            return err

        db = _db()
        docs = list(db.collection("loyaltyCodes").stream())

        all_school_ids = {
            sid
            for d in docs
            for sid in (d.to_dict().get("redeemedBySchools") or [])
        }
        school_names = {}
        for sid in all_school_ids:
            snap = db.collection("schools").document(sid).get()
            school_names[sid] = snap.to_dict().get("name", sid) if snap.exists else sid

        codes = []
        for d in docs:
            data = d.to_dict()
            redeemed_ids = data.get("redeemedBySchools") or []
            codes.append({
                "code": d.id,
                "active": data.get("active", False),
                "description": data.get("description", ""),
                "maxRedemptions": data.get("maxRedemptions"),
                "redeemedCount": len(redeemed_ids),
                "redeemedBySchools": [
                    {"schoolId": sid, "schoolName": school_names.get(sid, sid)}
                    for sid in redeemed_ids
                ],
                "expiresAt": data.get("expiresAt").isoformat() if data.get("expiresAt") else None,
                "createdAt": data.get("createdAt").isoformat() if data.get("createdAt") else None,
            })

        codes.sort(key=lambda c: c.get("createdAt") or "", reverse=True)

        return jsonify({"codes": codes}), 200

    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Could not list loyalty codes."}), 500


# ══════════════════════════════════════════════════════════════════════════════
# ROUTE: Admin -- deactivate a code
# ══════════════════════════════════════════════════════════════════════════════

@loyalty_bp.route("/api/billing/loyalty/admin/<code>/deactivate", methods=["POST", "OPTIONS"])
def loyalty_admin_deactivate(code):
    if request.method == "OPTIONS":
        return "", 204
    try:
        admin_uid, err = _verify_admin_token(request)
        if err:
            return err

        db = _db()
        code_ref = db.collection("loyaltyCodes").document(code.strip().upper())
        if not code_ref.get().exists:
            return jsonify({"error": "Code not found."}), 404

        code_ref.update({"active": False})

        _audit("loyalty_code_deactivated", admin_uid, code, {})

        return jsonify({"code": code, "status": "deactivated"}), 200

    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Could not deactivate loyalty code."}), 500