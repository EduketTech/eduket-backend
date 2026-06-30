"""
billing_routes.py
------------------
Flask routes that expose pricing_engine.py to the frontend:

  POST /api/billing/quote     - read-only price preview, one tier
  GET  /api/billing/quotes    - read-only price preview, all tiers
  POST /api/billing/initiate  - creates the AUTHORITATIVE pending
                                 transaction and returns everything
                                 PaymentForm needs to redirect to PayFast

Registering the blueprint - already done in app.py:

    from billing_routes import billing_bp
    app.register_blueprint(billing_bp)

AUTH MODEL (matches your existing /exams/upload and /exams/usage pattern):
every route here requires a valid Firebase ID token via the Authorization
header, and `schoolId` is ALWAYS derived server-side from
`users/{uid}.schoolId` - never trusted from the request body or URL. The
earlier version of this file accepted a client-supplied schoolId directly,
which would have let anyone query or initiate a checkout against a school
they don't belong to. `_verify_request_token` below is duplicated from
app.py rather than imported, specifically so this file doesn't require any
edit to that already-working function - extract it into a shared module
later if you want to de-duplicate.
"""

import logging
import os
import uuid
from datetime import datetime, timezone

from flask import Blueprint, jsonify, request
from firebase_admin import firestore, auth as fb_auth

from pricing_engine import (
    BASE_TIER_PRICES_ZAR,
    compute_subscription_price,
)

logger = logging.getLogger(__name__)
billing_bp = Blueprint("billing", __name__)

# ADJUST: these should already exist as env vars on your Render service
# (same ones PaymentManager.jsx used client-side before, just without the
# VITE_ prefix).
PAYFAST_MERCHANT_ID = os.environ.get("PAYFAST_MERCHANT_ID")
PAYFAST_MERCHANT_KEY = os.environ.get("PAYFAST_MERCHANT_KEY")
PAYFAST_URL = "https://www.payfast.co.za/eng/process"

# Real production frontend domain, confirmed from app.py's CORS list.
FRONTEND_BASE_URL = os.environ.get("FRONTEND_BASE_URL", "https://eduket.tech")

# ADJUST - SEE pricing_engine.py ASSUMPTION #1: I don't have confident
# knowledge of PayFast's exact field name for specifying a non-ZAR
# settlement currency on a multi-currency-enabled account. "currency" is a
# guess based on common gateway convention (Stripe/PayPal both use this
# name) - check your PayFast multi-currency docs/dashboard and fix this one
# constant if it's wrong.
PAYFAST_CURRENCY_FIELD = "currency"


def _verify_request_token(req):
    """Duplicated from app.py's verify_request_token - see module docstring."""
    auth_header = req.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return None, (jsonify({"error": "Missing or malformed Authorization header"}), 401)

    id_token = auth_header.split("Bearer ", 1)[1].strip()
    try:
        decoded = fb_auth.verify_id_token(id_token)
        return decoded["uid"], None
    except Exception as exc:  # noqa: BLE001
        logger.warning("[_verify_request_token] %s", exc)
        return None, (jsonify({"error": "Invalid or expired token"}), 401)


def _get_authoritative_school(req):
    """Verifies the caller's token, resolves their schoolId from their own
    user doc (same pattern as /exams/upload), and returns the full school
    dict. Returns (school_id, school_dict, error_response_or_None)."""
    uid, err = _verify_request_token(req)
    if err:
        return None, None, err

    db = firestore.client()
    user_doc = db.collection("users").document(uid).get()
    if not user_doc.exists:
        return None, None, (jsonify({"error": "User profile not found"}), 404)

    school_id = user_doc.to_dict().get("schoolId")
    if not school_id:
        return None, None, (jsonify({"error": "No school associated with this account"}), 400)

    school_snap = db.collection("schools").document(school_id).get()
    if not school_snap.exists:
        return None, None, (jsonify({"error": "School not found"}), 404)

    return school_id, (school_snap.to_dict() or {}), None


def _pricing_inputs_from_school(school):
    """Field names confirmed against your actual schools schema:
    `countryCode` (ISO alpha-2) ahead of `country` (full name);
    `teachingPhases` is an array, handled directly by
    pricing_engine.resolve_institution_type."""
    country_raw = school.get("countryCode") or school.get("country")
    institution_raw = school.get("teachingPhases") or school.get("institutionType")
    return country_raw, institution_raw


def _to_camel_case(quote_dict):
    """pricing_engine.py stays snake_case (idiomatic Python); convert at the
    API boundary so the frontend gets normal-looking JS object keys."""
    return {
        "tierId": quote_dict["tier_id"],
        "billingCycle": quote_dict["billing_cycle"],
        "baseAmountZar": quote_dict["base_amount_zar"],
        "institutionType": quote_dict["institution_type"],
        "institutionMultiplier": quote_dict["institution_multiplier"],
        "currencyMultiplier": quote_dict["currency_multiplier"],
        "exchangeRateSource": quote_dict["exchange_rate_source"],
        "chargeCurrency": quote_dict["charge_currency"],
        "chargeAmount": quote_dict["charge_amount"],
        "amountZarEquivalent": quote_dict["amount_zar_equivalent"],
    }


# ---------------------------------------------------------------------------
# Read-only previews - safe to call anytime, create nothing.
# ---------------------------------------------------------------------------
@billing_bp.route("/api/billing/quote", methods=["POST"])
def billing_quote():
    """Single-tier quote for the calling user's own school.
    Body: { tierId, billingCycle }"""
    school_id, school, err = _get_authoritative_school(request)
    if err:
        return err

    data = request.get_json() or {}
    tier_id = data.get("tierId")
    billing_cycle = data.get("billingCycle", "monthly")

    if tier_id not in BASE_TIER_PRICES_ZAR:
        return jsonify({"error": f"Unknown tierId '{tier_id}'"}), 400

    country_raw, institution_raw = _pricing_inputs_from_school(school)
    quote = compute_subscription_price(tier_id, billing_cycle, institution_raw, country_raw)
    return jsonify(_to_camel_case(quote)), 200


@billing_bp.route("/api/billing/quotes", methods=["GET"])
def billing_quotes_all_tiers():
    """Bulk quote for every tier at once, for the calling user's own school
    (cheaper than N round trips from TierSelection).
    Query param: ?billingCycle=monthly|annual"""
    school_id, school, err = _get_authoritative_school(request)
    if err:
        return err

    billing_cycle = request.args.get("billingCycle", "monthly")
    country_raw, institution_raw = _pricing_inputs_from_school(school)

    quotes = [
        _to_camel_case(compute_subscription_price(tier_id, billing_cycle, institution_raw, country_raw))
        for tier_id in BASE_TIER_PRICES_ZAR.keys()
    ]
    return jsonify(quotes), 200


# ---------------------------------------------------------------------------
# Mutating - creates the pending transaction record that payfast-itn.js will
# later verify against. Call this only when the user actually clicks "Pay".
# ---------------------------------------------------------------------------
@billing_bp.route("/api/billing/initiate", methods=["POST"])
def billing_initiate():
    """Body: { tierId, billingCycle }. schoolId/schoolName/currentTier all
    come from the verified school record, never from the request body.
    Returns everything PaymentForm needs to build the PayFast redirect form."""
    school_id, school, err = _get_authoritative_school(request)
    if err:
        return err

    data = request.get_json() or {}
    tier_id = data.get("tierId")
    billing_cycle = data.get("billingCycle", "monthly")

    if tier_id not in BASE_TIER_PRICES_ZAR:
        return jsonify({"error": f"Unknown tierId '{tier_id}'"}), 400

    country_raw, institution_raw = _pricing_inputs_from_school(school)
    school_name = school.get("name", school_id)
    current_tier = school.get("tier", "free")

    quote = compute_subscription_price(tier_id, billing_cycle, institution_raw, country_raw)
    payment_id = f"PAY-{uuid.uuid4().hex}"

    db = firestore.client()
    batch = db.batch()

    # 'billing' doc - same shape your SubscriptionManager already reads,
    # plus the new currency context.
    batch.set(db.collection("billing").document(), {
        "schoolId": school_id,
        "tierId": tier_id,
        "billingCycle": quote["billing_cycle"],
        "amount": quote["charge_amount"],
        "currency": quote["charge_currency"],
        "status": "pending",
        "transactionRef": payment_id,
        "pricing": {
            "baseAmountZar": quote["base_amount_zar"],
            "institutionType": quote["institution_type"],
            "institutionMultiplier": quote["institution_multiplier"],
            "currencyMultiplier": quote["currency_multiplier"],
            "amountZarEquivalent": quote["amount_zar_equivalent"],
        },
        "createdAt": firestore.SERVER_TIMESTAMP,
    })

    # 'paymentTransactions/{payment_id}' - this is what payfast-itn.js looks
    # up by m_payment_id to verify the ITN amount before upgrading anything.
    # expectedAmount/expectedCurrency are never touched again after this -
    # the ITN writes the ACTUAL paid amount to separate fields, so the two
    # stay comparable instead of one overwriting the other.
    batch.set(db.collection("paymentTransactions").document(payment_id), {
        "schoolId": school_id,
        "tierId": tier_id,
        "fromTier": current_tier,
        "billingCycle": quote["billing_cycle"],
        "expectedAmount": quote["charge_amount"],
        "expectedCurrency": quote["charge_currency"],
        "status": "pending",
        "createdAt": firestore.SERVER_TIMESTAMP,
    })

    batch.commit()

    payment_data = {
        "merchant_id": PAYFAST_MERCHANT_ID,
        "merchant_key": PAYFAST_MERCHANT_KEY,
        "return_url": f"{FRONTEND_BASE_URL}/payment-success",
        "cancel_url": f"{FRONTEND_BASE_URL}/payment-cancelled",
        "m_payment_id": payment_id,
        "amount": f"{quote['charge_amount']:.2f}",
        PAYFAST_CURRENCY_FIELD: quote["charge_currency"],
        "item_name": f"{tier_id.capitalize()} Plan",
        "item_description": f"{tier_id.capitalize()} Subscription ({quote['billing_cycle']})",
        "custom_str1": school_id,
        "custom_str2": tier_id,
        "custom_str3": current_tier,
        "custom_str4": quote["billing_cycle"],
    }

    return jsonify({
        "paymentId": payment_id,
        "paymentData": payment_data,
        "quote": _to_camel_case(quote),
    }), 200