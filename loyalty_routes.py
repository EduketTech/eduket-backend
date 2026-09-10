"""
loyalty_routes.py

Flask blueprint for the loyalty-code flow. Register with:
    from loyalty_routes import loyalty_bp
    app.register_blueprint(loyalty_bp)

Assumes:
  - `db` is a firebase_admin firestore client, available via get_db()
    (swap this import for however your app currently gets its Firestore
    handle -- e.g. the same place billing_routes.py gets it).
  - Requests are authenticated the same way your other /api/billing routes
    are (a decorator that verifies the Firebase ID token and attaches
    request.school_id / request.uid). Adjust the decorator names below to
    match what you already use elsewhere (e.g. in the billing blueprint).
"""

import secrets
import string

from flask import Blueprint, jsonify, request

from pricing import redeem_loyalty_code, is_loyalty_subscription_active

# Adjust these imports to match your existing auth decorators/db accessor.
# They almost certainly already exist in your billing blueprint -- reuse
# the same ones rather than duplicating auth logic.
from auth_decorators import require_auth, require_admin  # noqa: adjust path
from firebase_setup import get_db  # noqa: adjust path

loyalty_bp = Blueprint("loyalty", __name__, url_prefix="/api/billing/loyalty")


def _generate_code(length: int = 10) -> str:
    """Human-typeable code: uppercase letters + digits, no ambiguous chars."""
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # no O/0, I/1
    return "".join(secrets.choice(alphabet) for _ in range(length))


@loyalty_bp.route("/redeem", methods=["POST"])
@require_auth  # school-scoped auth: sets request.school_id from the verified token
def redeem():
    """
    School redeems a loyalty code for the current school (school_id is
    derived server-side from the auth token -- never trust a school_id
    passed in the request body).
    """
    data = request.get_json(force=True) or {}
    code = (data.get("code") or "").strip().upper()

    if not code:
        return jsonify({"error": "Missing loyalty code."}), 400

    db = get_db()
    try:
        result = redeem_loyalty_code(db, school_id=request.school_id, code=code)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    return jsonify(result), 200


@loyalty_bp.route("/status", methods=["GET"])
@require_auth
def status():
    """
    Frontend calls this alongside the normal quote fetch so it knows
    whether to render the redeemed/active state or the code-entry box.
    """
    db = get_db()
    active = is_loyalty_subscription_active(db, school_id=request.school_id)

    cycle_end = None
    if active:
        school_snap = db.collection("schools").document(request.school_id).get()
        loyalty = (school_snap.to_dict() or {}).get("loyaltySubscription") or {}
        cycle_end_dt = loyalty.get("cycleEnd")
        cycle_end = cycle_end_dt.isoformat() if cycle_end_dt else None

    return jsonify({"active": active, "cycleEnd": cycle_end}), 200


# ─── Admin-only: creating new codes ───────────────────────────────────────

@loyalty_bp.route("/admin/create", methods=["POST"])
@require_admin  # must verify a custom claim (e.g. admin: true) on YOUR account
def admin_create_code():
    """
    Mint a new loyalty code. Optional body:
      { "code": "PILOT2026",       # omit to auto-generate
        "maxRedemptions": 5,       # omit for unlimited schools
        "expiresAt": "2026-12-31T00:00:00Z",  # omit for no expiry
        "description": "Q4 conference giveaway" }
    """
    data = request.get_json(force=True) or {}
    code = (data.get("code") or _generate_code()).strip().upper()

    db = get_db()
    code_ref = db.collection("loyaltyCodes").document(code)
    if code_ref.get().exists:
        return jsonify({"error": f"Code '{code}' already exists."}), 409

    code_ref.set({
        "active": True,
        "maxRedemptions": data.get("maxRedemptions"),  # None == unlimited schools
        "expiresAt": data.get("expiresAt"),            # store as Firestore Timestamp if provided
        "description": data.get("description", ""),
        "redeemedBySchools": [],
    })

    return jsonify({"code": code, "status": "created"}), 201


@loyalty_bp.route("/admin/<code>/deactivate", methods=["POST"])
@require_admin
def admin_deactivate_code(code):
    """Kill a code immediately -- schools currently mid-cycle keep access
    until their cycleEnd, but can't re-redeem this code afterward."""
    db = get_db()
    code_ref = db.collection("loyaltyCodes").document(code.strip().upper())
    if not code_ref.get().exists:
        return jsonify({"error": "Code not found."}), 404

    code_ref.update({"active": False})
    return jsonify({"code": code, "status": "deactivated"}), 200