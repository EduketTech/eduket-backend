"""
billing_routes.py — Eduket OS  Billing & Subscription API  v3.2
═══════════════════════════════════════════════════════════════════════════════
Registered in app.py:
    from billing_routes import billing_bp
    app.register_blueprint(billing_bp)

Routes
──────
    POST /api/billing/quote       Single-tier price quote
    GET  /api/billing/quotes      All-tier price quotes
    POST /api/billing/initiate    Create pending transaction + PayFast form data
    POST /api/payfast/itn         PayFast ITN webhook (tier upgrade on payment)

Security controls applied
─────────────────────────
    CRIT-04  PayFast ITN idempotency check — duplicate ITN POST ignored
    CRIT-04  PayFast IP allowlist — non-PayFast IPs logged and flagged
    CRIT-04  ITN signature verification — MD5 + passphrase
    CRIT-04  Amount verification — paid amount checked against pending record
    HIGH-01  Audit log on every tier upgrade
    Tier writes happen EXCLUSIVELY through the ITN handler via Admin SDK.
    No client-side tier write is possible.

Environment variables required
────────────────────────────────
    PAYFAST_MERCHANT_ID   — from PayFast merchant dashboard
    PAYFAST_MERCHANT_KEY  — from PayFast merchant dashboard
    PAYFAST_PASSPHRASE    — set in PayFast Settings → Integration → Passphrase
    FRONTEND_BASE_URL     — e.g. https://eduket.tech  (no trailing slash)
    BACKEND_BASE_URL      — e.g. https://chatbot-backend-educat.onrender.com
    EXCHANGE_RATE_API_KEY — optional; open.er-api.com free tier used if absent
"""

from dotenv import load_dotenv
load_dotenv()

import os
import uuid
import hashlib
import logging
import traceback
from datetime import datetime, timezone, timedelta
from urllib.parse import quote_plus

import requests as http_requests
from flask import Blueprint, request, jsonify
from urllib.parse import urlencode, quote_plus

import firebase_admin
from firebase_admin import firestore as fs_admin, auth as fb_auth

logger     = logging.getLogger(__name__)
billing_bp = Blueprint("billing", __name__)


# ══════════════════════════════════════════════════════════════════════════════
# CREDENTIALS & CONFIG
# ══════════════════════════════════════════════════════════════════════════════

PAYFAST_MERCHANT_ID  = os.getenv("PAYFAST_MERCHANT_ID",  "").strip()
PAYFAST_MERCHANT_KEY = os.getenv("PAYFAST_MERCHANT_KEY", "").strip()
PAYFAST_PASSPHRASE   = os.getenv("PAYFAST_PASSPHRASE",   "").strip()

if not all([PAYFAST_MERCHANT_ID, PAYFAST_MERCHANT_KEY]):
    raise RuntimeError(
        "Missing PayFast credentials. Set PAYFAST_MERCHANT_ID and "
        "PAYFAST_MERCHANT_KEY in your environment variables."
    )

FRONTEND_BASE_URL = os.environ.get(
    "FRONTEND_BASE_URL", "https://eduket.tech"
).rstrip("/")

BACKEND_BASE_URL = os.environ.get(
    "BACKEND_BASE_URL", "https://chatbot-backend-educat.onrender.com"
).rstrip("/")

# PayFast known IP ranges
# Source: https://developers.payfast.co.za/docs#step_4_confirm_payment
# Logged but not hard-rejected — IP ranges can expand without notice.
PAYFAST_IPS = {
    "197.97.145.144", "197.97.145.145", "197.97.145.146", "197.97.145.147",
    "197.97.145.148", "197.97.145.149", "197.97.145.150", "197.97.145.151",
    "41.74.179.194",  "41.74.179.195",  "41.74.179.196",  "41.74.179.197",
    "197.97.144.128",
}


# ══════════════════════════════════════════════════════════════════════════════
# FIREBASE CLIENT
# ══════════════════════════════════════════════════════════════════════════════

def _db():
    """Return a Firestore client using the already-initialised Firebase app."""
    return fs_admin.client()


# ══════════════════════════════════════════════════════════════════════════════
# AUTH HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _verify_token(req) -> tuple:
    """
    Verify Firebase ID token from Authorization header.
    Returns (uid, None) on success or (None, error_response) on failure.
    """
    header = req.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        return None, (jsonify({"error": "Missing or malformed Authorization header"}), 401)
    try:
        decoded = fb_auth.verify_id_token(header.split("Bearer ", 1)[1].strip())
        return decoded["uid"], None
    except Exception as e:
        logger.warning("[Billing Auth] Token verification failed: %s", e)
        return None, (jsonify({"error": "Invalid or expired token"}), 401)


def _get_school_id_for_uid(uid: str):
    """
    Derive schoolId server-side from Firestore.
    The schoolId in the request body is NEVER trusted.
    """
    try:
        doc = _db().collection("users").document(uid).get()
        return doc.to_dict().get("schoolId") if doc.exists else None
    except Exception as e:
        logger.error("[Billing] schoolId lookup failed for %s: %s", uid, e)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# AUDIT LOG
# ══════════════════════════════════════════════════════════════════════════════

def _audit(action: str, actor: str, target: str, details: dict = {}):
    """Write a billing audit entry. Never raises."""
    try:
        _db().collection("auditLog").add({
            "action":    action,
            "actorUid":  actor,
            "target":    target,
            "details":   details,
            "timestamp": fs_admin.SERVER_TIMESTAMP,
            "ip": request.headers.get(
                "X-Forwarded-For", request.remote_addr or "unknown"
            ).split(",")[0].strip(),
        })
    except Exception as e:
        logger.error("[Audit] Billing log failed: %s", e)


# ══════════════════════════════════════════════════════════════════════════════
# SUBSCRIPTION TIERS
# ══════════════════════════════════════════════════════════════════════════════

SUBSCRIPTION_TIERS = {
    "free": {
        "id": "free", "name": "Free", "price_zar": 0, "exam_limit": 5,
        "description": "5 exam uploads per month. Perfect for a trial.",
        "features": ["5 exam uploads/month", "AI marking",
                     "Student results", "Basic analytics"],
    },
    "silver": {
        "id": "silver", "name": "Silver", "price_zar": 799, "exam_limit": 30,
        "description": "Ideal for individual teachers.",
        "features": ["30 exam uploads/month", "All Free features",
                     "Memo-based marking", "Class performance reports",
                     "AI study coach"],
    },
    "gold": {
        "id": "gold", "name": "Gold", "price_zar": 1399, "exam_limit": 120,
        "description": "For departments and small schools.",
        "features": ["120 exam uploads/month", "All Silver features",
                     "Multi-subject analytics", "PDF report export",
                     "Priority support"],
    },
    "platinum": {
        "id": "platinum", "name": "Platinum", "price_zar": 2999, "exam_limit": 500,
        "description": "For medium-sized schools.",
        "features": ["500 exam uploads/month", "All Gold features",
                     "Principal dashboard", "Predictive analytics",
                     "Dedicated support"],
    },
    "diamond": {
        "id": "diamond", "name": "Diamond", "price_zar": 5998, "exam_limit": 1000,
        "description": "For large schools and college networks.",
        "features": ["1,000 exam uploads/month", "All Platinum features",
                     "Multi-campus visibility", "API access", "SLA guarantee"],
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# PRICING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

INSTITUTION_MULTIPLIERS = {
    "primary_school": 1.0, "secondary_school": 1.0, "high_school": 1.0,
    "college": 1.5, "university": 2.0, "private_college": 1.8, "other": 1.0,
}

CURRENCY_STRENGTH = {
    "ZAR": 1.0, "USD": 3.0, "GBP": 3.5, "EUR": 3.2,
    "KES": 1.2, "NGN": 1.1, "GHS": 1.2, "UGX": 1.1,
    "ZWG": 1.3, "BWP": 1.3, "NAD": 1.0, "MWK": 1.1,
    "TZS": 1.2, "ZMW": 1.2, "ETB": 1.1, "XOF": 1.2,
}

_FX_CACHE: dict = {}
_FX_CACHE_TS: float = 0
_FX_CACHE_TTL = 86_400  # 24 hours


def _get_exchange_rates() -> dict:
    """
    Fetch live USD exchange rates, caching for 24 hours.
    Falls through: Firestore cache → open.er-api.com → hardcoded emergency rates.
    """
    import time
    global _FX_CACHE, _FX_CACHE_TS

    if _FX_CACHE and (time.time() - _FX_CACHE_TS) < _FX_CACHE_TTL:
        return _FX_CACHE

    # 1. Firestore cache
    try:
        doc = _db().collection("system_config").document("exchange_rates").get()
        if doc.exists:
            data = doc.to_dict()
            if (time.time() - data.get("updatedAt", 0)) < _FX_CACHE_TTL:
                _FX_CACHE    = data.get("rates", {})
                _FX_CACHE_TS = time.time()
                return _FX_CACHE
    except Exception as e:
        logger.warning("[FX] Firestore cache read failed: %s", e)

    # 2. open.er-api.com
    try:
        api_key = os.environ.get("EXCHANGE_RATE_API_KEY", "")
        url = (f"https://v6.exchangerate-api.com/v6/{api_key}/latest/USD"
               if api_key else "https://open.er-api.com/v6/latest/USD")
        resp = http_requests.get(url, timeout=10)
        if resp.status_code == 200:
            rates        = resp.json().get("rates", {})
            _FX_CACHE    = rates
            _FX_CACHE_TS = time.time()
            try:
                _db().collection("system_config").document("exchange_rates").set({
                    "rates": rates, "updatedAt": time.time(), "source": "open.er-api.com",
                }, merge=True)
            except Exception:
                pass
            return rates
    except Exception as e:
        logger.warning("[FX] Live rate fetch failed: %s", e)

    # 3. Emergency hardcoded fallback
    logger.warning("[FX] Using emergency hardcoded rates")
    emergency = {
        "ZAR": 18.5, "USD": 1.0,  "GBP": 0.79, "EUR": 0.92,
        "KES": 130,  "NGN": 1550, "GHS": 15.2, "ZWG": 13.5,
        "BWP": 13.6, "NAD": 18.5, "UGX": 3750, "TZS": 2550,
    }
    _FX_CACHE    = emergency
    _FX_CACHE_TS = time.time()
    return emergency


def _calculate_price(base_price_zar: float, currency_code: str,
                     billing_cycle: str, institution_type: str = "secondary_school") -> dict:
    """
    Calculate the price adjusted for billing cycle, institution type,
    currency strength, and live exchange rate.

    PayFast ALWAYS receives amount_zar_equivalent — never the local display amount.
    Annual billing = 10 months (2 months free ≈ 17% discount).
    """
    rates      = _get_exchange_rates()
    fx_rate    = rates.get("ZAR", 18.5) / rates.get(currency_code, 1.0)
    inst_mult  = INSTITUTION_MULTIPLIERS.get(institution_type, 1.0)
    curr_mult  = CURRENCY_STRENGTH.get(currency_code, 1.0)
    months     = 10 if billing_cycle == "annual" else 1

    price_zar_base    = base_price_zar * months * inst_mult
    amount_zar_equiv  = round(price_zar_base * curr_mult, 2)
    charge_amount     = round(amount_zar_equiv / fx_rate, 2)

    return {
        "charge_amount":          charge_amount,
        "charge_currency":        currency_code,
        "amount_zar_equivalent":  amount_zar_equiv,
        "billing_cycle":          billing_cycle,
        "fx_rate":                round(fx_rate, 4),
        "institution_multiplier": inst_mult,
        "currency_multiplier":    curr_mult,
        "months":                 months,
    }


def _get_school_pricing_context(school_id: str) -> dict:
    """Fetch the school's country, currency, and institution type from Firestore."""
    defaults = {
        "country": "South Africa", "currency": "ZAR",
        "institution_type": "secondary_school", "billing_cycle": "monthly",
    }
    try:
        doc = _db().collection("schools").document(school_id).get()
        if doc.exists:
            d = doc.to_dict()
            return {
                "country":          d.get("country",         defaults["country"]),
                "currency":         d.get("currency",        defaults["currency"]).upper(),
                "institution_type": d.get("institutionType", defaults["institution_type"]),
                "billing_cycle":    defaults["billing_cycle"],
            }
    except Exception as e:
        logger.warning("[Billing] School context lookup failed: %s", e)
    return defaults


# ══════════════════════════════════════════════════════════════════════════════
# PAYFAST SIGNATURE
# ══════════════════════════════════════════════════════════════════════════════

def _generate_payfast_signature(params: dict) -> str:
    """
    Generate PayFast MD5 signature.
    Sort alphabetically, exclude empty values and 'signature'.
    Append passphrase only if configured.
    Use same encoding PayFast uses: quote_plus with %20 converted to +.
    """
    # Filter and sort alphabetically
    filtered = {
        k: str(v).strip()
        for k, v in params.items()
        if k != "signature"
        and v is not None
        and str(v).strip() != ""
    }
    sorted_items = sorted(filtered.items(), key=lambda x: x[0])

    # Build param string — spaces as +, special chars as %XX
    parts = []
    for k, v in sorted_items:
        encoded_v = quote_plus(v)
        parts.append(f"{k}={encoded_v}")

    param_string = "&".join(parts)

    # Append passphrase only if set
    if PAYFAST_PASSPHRASE and PAYFAST_PASSPHRASE.strip():
        param_string += f"&passphrase={quote_plus(PAYFAST_PASSPHRASE.strip())}"

    return hashlib.md5(param_string.encode("utf-8")).hexdigest()

def _verify_payfast_signature(data: dict) -> bool:
    """
    Verify a PayFast ITN signature.
    Uses same alphabetical sort as _generate_payfast_signature.
    """
    received = data.get("signature", "")
    check    = {k: v for k, v in data.items() if k != "signature"}
    expected = _generate_payfast_signature(check)
    # print(f"[PayFast VERIFY] received={received} expected={expected}")
    return received == expected

# ══════════════════════════════════════════════════════════════════════════════
# ROUTE: Single tier price quote
# ══════════════════════════════════════════════════════════════════════════════

@billing_bp.route("/api/billing/quote", methods=["POST", "OPTIONS"])
def billing_quote():
    """Return a price quote for a single tier."""
    if request.method == "OPTIONS":
        return "", 204
    try:
        uid, err = _verify_token(request)
        if err:
            return err

        data          = request.get_json() or {}
        tier_id       = data.get("tierId", "").lower().strip()
        billing_cycle = data.get("billingCycle", "monthly").lower()

        if tier_id not in SUBSCRIPTION_TIERS:
            return jsonify({"error": f"Unknown tier: {tier_id}"}), 400
        if billing_cycle not in ("monthly", "annual"):
            return jsonify({"error": "billingCycle must be 'monthly' or 'annual'"}), 400

        school_id = _get_school_id_for_uid(uid)
        if not school_id:
            return jsonify({"error": "No school associated with this account"}), 400

        tier          = SUBSCRIPTION_TIERS[tier_id]
        context       = _get_school_pricing_context(school_id)
        billing_cycle = data.get("billingCycle", context.get("billing_cycle", "monthly"))

        quote = _calculate_price(
            base_price_zar   = tier["price_zar"],
            currency_code    = context["currency"],
            billing_cycle    = billing_cycle,
            institution_type = context["institution_type"],
        )

        return jsonify({
            "tierId":              tier_id,
            "tierName":            tier["name"],
            "chargeAmount":        quote["charge_amount"],
            "chargeCurrency":      quote["charge_currency"],
            "amountZarEquivalent": quote["amount_zar_equivalent"],
            "billingCycle":        billing_cycle,
            "country":             context["country"],
            "fxRate":              quote["fx_rate"],
            "institutionMultiplier": quote["institution_multiplier"],
            "currencyMultiplier":    quote["currency_multiplier"],
        })
    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Could not calculate price quote."}), 500


# ══════════════════════════════════════════════════════════════════════════════
# ROUTE: All tier quotes at once
# ══════════════════════════════════════════════════════════════════════════════

@billing_bp.route("/api/billing/quotes", methods=["GET", "OPTIONS"])
def billing_quotes_all():
    """Return price quotes for all five tiers simultaneously."""
    if request.method == "OPTIONS":
        return "", 204
    try:
        uid, err = _verify_token(request)
        if err:
            return err

        billing_cycle = request.args.get("billingCycle", "monthly").lower()
        if billing_cycle not in ("monthly", "annual"):
            billing_cycle = "monthly"

        school_id = _get_school_id_for_uid(uid)
        context   = _get_school_pricing_context(school_id) if school_id else {
            "currency": "ZAR", "institution_type": "secondary_school",
            "country": "South Africa",
        }

        quotes = {}
        for tier_id, tier in SUBSCRIPTION_TIERS.items():
            if tier["price_zar"] == 0:
                quotes[tier_id] = {
                    "tierId": tier_id, "tierName": tier["name"],
                    "chargeAmount": 0, "chargeCurrency": context["currency"],
                    "amountZarEquivalent": 0, "billingCycle": billing_cycle,
                    "examLimit": tier["exam_limit"], "features": tier["features"],
                }
                continue

            q = _calculate_price(
                base_price_zar   = tier["price_zar"],
                currency_code    = context["currency"],
                billing_cycle    = billing_cycle,
                institution_type = context["institution_type"],
            )
            quotes[tier_id] = {
                "tierId":              tier_id,
                "tierName":            tier["name"],
                "chargeAmount":        q["charge_amount"],
                "chargeCurrency":      q["charge_currency"],
                "amountZarEquivalent": q["amount_zar_equivalent"],
                "billingCycle":        billing_cycle,
                "examLimit":           tier["exam_limit"],
                "features":            tier["features"],
                "country":             context["country"],
                "institutionMultiplier": q["institution_multiplier"],
                "currencyMultiplier":    q["currency_multiplier"],
            }

        return jsonify({"quotes": quotes, "billingCycle": billing_cycle})
    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Could not retrieve tier quotes."}), 500


# ══════════════════════════════════════════════════════════════════════════════
# ROUTE: Initiate payment
# ══════════════════════════════════════════════════════════════════════════════

@billing_bp.route("/api/billing/initiate", methods=["POST", "OPTIONS"])
def billing_initiate():
    """
    Create a pending payment transaction in Firestore and return the
    PayFast form fields for the frontend to POST to PayFast.

    amount sent to PayFast is ALWAYS amount_zar_equivalent — never
    the local display amount. PayFast processes in ZAR only.
    """
    if request.method == "OPTIONS":
        return "", 204
    try:
        uid, err = _verify_token(request)
        if err:
            return err

        data          = request.get_json() or {}
        tier_id       = data.get("tierId", "").lower().strip()
        billing_cycle = data.get("billingCycle", "monthly").lower()

        if tier_id not in SUBSCRIPTION_TIERS:
            return jsonify({"error": f"Unknown tier: {tier_id}"}), 400
        if billing_cycle not in ("monthly", "annual"):
            return jsonify({"error": "billingCycle must be 'monthly' or 'annual'"}), 400

        school_id = _get_school_id_for_uid(uid)
        if not school_id:
            return jsonify({"error": "No school associated with this account"}), 400

        school_doc = _db().collection("schools").document(school_id).get()
        if not school_doc.exists:
            return jsonify({"error": "School not found"}), 404

        school_data   = school_doc.to_dict()
        current_tier  = school_data.get("tier", "free")
        tier          = SUBSCRIPTION_TIERS[tier_id]
        context       = _get_school_pricing_context(school_id)

        if billing_cycle not in ("monthly", "annual"):
            billing_cycle = context.get("billing_cycle", "monthly")

        quote = _calculate_price(
            base_price_zar   = tier["price_zar"],
            currency_code    = context["currency"],
            billing_cycle    = billing_cycle,
            institution_type = context["institution_type"],
        )

        payment_id = f"EDUKET_{school_id[:8].upper()}_{uuid.uuid4().hex[:8].upper()}"

        payment_data = {
            "merchant_id":      PAYFAST_MERCHANT_ID,
            "merchant_key":     PAYFAST_MERCHANT_KEY,
            "return_url":       f"{FRONTEND_BASE_URL}/payment/success",
            "cancel_url":       f"{FRONTEND_BASE_URL}/payment/cancel",
            "notify_url":       f"{BACKEND_BASE_URL}/api/payfast/itn",
            "m_payment_id":     payment_id,
            "amount":           f"{quote['amount_zar_equivalent']:.2f}",
            "item_name":        f"Eduket OS {tier['name']} Plan",
            "item_description": f"{tier['name']} Subscription ({billing_cycle})",
            "custom_str1":      school_id,
            "custom_str2":      tier_id,
            "custom_str3":      current_tier,
            "custom_str4":      billing_cycle,
            "custom_str5":      context["currency"],
        }

        payment_data["signature"] = _generate_payfast_signature(payment_data)

        # Write pending transaction BEFORE redirecting to PayFast.
        # The ITN handler verifies the payment amount against this record.
        _db().collection("paymentTransactions").document(payment_id).set({
            "schoolId":         school_id,
            "uid":              uid,
            "tierId":           tier_id,
            "fromTier":         current_tier,
            "billingCycle":     billing_cycle,
            "expectedAmount":   quote["amount_zar_equivalent"],
            "expectedCurrency": "ZAR",
            "displayAmount":    quote["charge_amount"],
            "displayCurrency":  context["currency"],
            "status":           "pending",
            "createdAt":        fs_admin.SERVER_TIMESTAMP,
            "createdByUid":     uid,
        })

        _audit("payment_initiated", uid, school_id, {
            "tierId": tier_id, "amount": quote["amount_zar_equivalent"],
            "currency": "ZAR", "payment_id": payment_id,
        })

        return jsonify({
            "paymentId":   payment_id,
            "paymentData": payment_data,
            "quote": {
                "chargeAmount":        quote["charge_amount"],
                "chargeCurrency":      quote["charge_currency"],
                "amountZarEquivalent": quote["amount_zar_equivalent"],
                "billingCycle":        billing_cycle,
            },
        }), 200

    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Could not initiate payment. Please try again."}), 500


# ══════════════════════════════════════════════════════════════════════════════
# ROUTE: PayFast ITN webhook
# ══════════════════════════════════════════════════════════════════════════════

@billing_bp.route("/api/payfast/itn", methods=["POST"])
def payfast_itn():
    """
    PayFast Instant Transaction Notification (ITN) handler.

    This is the SOLE authority for upgrading a school's tier.
    No client-side code can write to schools/{schoolId}.tier —
    Firestore rules prevent it. Only the Admin SDK (used here) can.

    Security checks in order:
      1. IP allowlist  — warns on non-PayFast IPs (logged, not hard-blocked)
      2. Signature     — MD5 + passphrase must match
      3. Merchant ID   — must match our credentials
      4. Status        — only COMPLETE triggers an upgrade
      5. Transaction   — pending record must exist for this payment_id
      6. Idempotency   — already-complete transactions are silently ignored
      7. Amount        — paid amount must match expected ± 1%
    """
    try:
        itn_data = request.form.to_dict(flat=True)

        # 1. IP allowlist check (warn only — signature is the real gate)
        sender_ip = request.headers.get(
            "X-Forwarded-For", request.remote_addr or ""
        ).split(",")[0].strip()

        if sender_ip and sender_ip not in PAYFAST_IPS:
            logger.warning("[ITN] Request from unrecognised IP: %s", sender_ip)

        # 2. Signature verification
        if not _verify_payfast_signature(itn_data):
            logger.warning("[ITN] Signature mismatch from IP: %s", sender_ip)
            return "", 400

        # 3. Merchant ID check
        if itn_data.get("merchant_id") != PAYFAST_MERCHANT_ID:
            logger.error("[ITN] Wrong merchant ID: %s", itn_data.get("merchant_id"))
            return "", 400

        # 4. Only process COMPLETE payments
        payment_status = itn_data.get("payment_status", "")
        if payment_status != "COMPLETE":
            logger.info("[ITN] Non-COMPLETE status: %s — acknowledged", payment_status)
            return "", 200

        payment_id    = itn_data.get("m_payment_id", "")
        pf_payment_id = itn_data.get("pf_payment_id", "")

        if not payment_id:
            logger.error("[ITN] Missing m_payment_id")
            return "", 400

        db = _db()

        # 5. Look up the pending transaction
        tx_ref  = db.collection("paymentTransactions").document(payment_id)
        tx_snap = tx_ref.get()

        if not tx_snap.exists:
            logger.error("[ITN] Unknown payment_id: %s", payment_id)
            return "", 400

        tx = tx_snap.to_dict()

        # 6. Idempotency — PayFast can resend the same ITN
        if tx.get("status") == "complete":
            logger.warning("[ITN] Duplicate ITN for %s — no-op", payment_id)
            return "", 200

        school_id     = tx.get("schoolId", "")
        tier_id       = tx.get("tierId",   "free")
        from_tier     = tx.get("fromTier", "free")
        billing_cycle = tx.get("billingCycle", "monthly")

        if not school_id or not tier_id:
            logger.error("[ITN] Malformed transaction record: %s", tx)
            return "", 400

        # 7. Amount verification (1% tolerance)
        try:
            paid_amount     = float(itn_data.get("amount_gross", 0))
            expected_amount = float(tx.get("expectedAmount", 0))
        except (TypeError, ValueError):
            logger.error("[ITN] Could not parse amounts for %s", payment_id)
            return "", 400

        tolerance  = max(0.5, expected_amount * 0.01)
        amount_ok  = abs(paid_amount - expected_amount) <= tolerance

        if not amount_ok:
            logger.error(
                "[ITN] Amount mismatch for %s: paid=%.2f expected=%.2f",
                payment_id, paid_amount, expected_amount,
            )
            tx_ref.set({
                "status":      "amount_mismatch",
                "paidAmount":  paid_amount,
                "pfPaymentId": pf_payment_id,
                "flaggedAt":   fs_admin.SERVER_TIMESTAMP,
            }, merge=True)
            return "", 200   # 200 so PayFast does not retry

        # All checks passed — upgrade the school

        # Calculate next billing date
        now          = datetime.now(timezone.utc)
        next_billing = (
            now + timedelta(days=365)
            if billing_cycle == "annual"
            else now + timedelta(days=30)
        )

        # Batch write — school tier + transaction status
        batch = db.batch()

        batch.set(db.collection("schools").document(school_id), {
            "tier":            tier_id,
            "tierUpdatedAt":   fs_admin.SERVER_TIMESTAMP,
            "nextBillingDate": next_billing.isoformat(),
            "pfPaymentId":     pf_payment_id,
            "subscribedAt":    fs_admin.SERVER_TIMESTAMP,
            "billingCycle":    billing_cycle,
        }, merge=True)

        batch.set(db.collection("principals").document(school_id), {
            "tier":            tier_id,
            "tierUpdatedAt":   fs_admin.SERVER_TIMESTAMP,
            "nextBillingDate": next_billing.isoformat(),
        }, merge=True)

        batch.set(tx_ref, {
            "status":      "complete",
            "paidAmount":  paid_amount,
            "pfPaymentId": pf_payment_id,
            "completedAt": fs_admin.SERVER_TIMESTAMP,
            "toTier":      tier_id,
            "fromTier":    from_tier,
            "senderIp":    sender_ip,
        }, merge=True)

        batch.commit()

        # Audit log
        _audit("tier_upgraded", "payfast_itn", school_id, {
            "fromTier":   from_tier,
            "toTier":     tier_id,
            "amount":     paid_amount,
            "payment_id": payment_id,
            "billing":    billing_cycle,
        })

        logger.info(
            "[ITN] ✓ School %s upgraded %s → %s (%.2f ZAR, payment: %s)",
            school_id, from_tier, tier_id, paid_amount, payment_id,
        )
        return "", 200

    except Exception as e:
        traceback.print_exc()
        logger.error("[ITN] Unexpected error: %s", e)
        return "", 500