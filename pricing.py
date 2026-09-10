import math
from datetime import datetime, timedelta, timezone

# ─── Reference Constants & Rates (in ZAR) ────────────────────────────────────
TAX_RATE = 0.15  # 15% VAT
FREE_STUDENT_BASE = 10  # Included free student baseline
FREE_TEACHER_BASE = 2   # Included free teacher baseline
DEFAULT_EXAMS_PER_TEACHER = 50
FREE_TIER_MONTHLY_LIMIT = 100

# ─── Loyalty / Trial-Bypass Codes ─────────────────────────────────────────
# A loyalty code opens a LOYALTY_CYCLE_DAYS window during which the school
# is billed nothing and has no upload/seat limits. It has to be redeemed
# again once the window lapses -- redemption *is* the subscription action
# for these schools, there's no separate payment.
LOYALTY_CYCLE_DAYS = 30

# Parent Fixed Rates
PARENT_RATES = {
    "monthly": 129.0,
    "annual": 1290.0,
    "yearly": 1290.0,
}

RATES = {
    "student_monthly": 32.0,
    "teacher_monthly": 105.0,
    "base_platform_fee_monthly": 350.0,  # Base Monthly Platform Maintenance Fee Floor
    "extra_exam_pack_price": 150.0,      # R150 per 10 additional AI exam extractions
    "extra_exam_pack_size": 10,
}

DISCOUNTS = {
    "monthly": 0.0,
    "quarterly": 0.10,  # 10% discount
    "yearly": 0.20,     # 20% discount
    "annual": 0.20,     # Alias for yearly
}

CYCLE_MONTHS = {
    "monthly": 1,
    "quarterly": 3,
    "yearly": 12,
    "annual": 12,
}


def calculate_parent_subscription_quote(cycle: str = "monthly") -> dict:
    """
    Calculates flat-rate parent portal subscription pricing.
    """
    cycle_key = cycle.lower()
    if cycle_key not in PARENT_RATES:
        raise ValueError(f"Invalid billing cycle for parent subscription: '{cycle}'. Must be one of {list(PARENT_RATES.keys())}")

    months = CYCLE_MONTHS.get(cycle_key, 1)
    total_due = PARENT_RATES[cycle_key]

    # Calculate subtotal before VAT (assuming standard 15% VAT breakdown)
    subtotal = total_due / (1 + TAX_RATE)
    tax_amount = total_due - subtotal

    return {
        "cycle": cycle_key,
        "months": months,
        "is_parent_plan": True,
        "is_free_baseline": False,
        "is_loyalty_plan": False,
        "subtotal_before_tax": round(subtotal, 2),
        "tax_rate_percent": int(TAX_RATE * 100),
        "tax_amount": round(tax_amount, 2),
        "total_due_now": round(total_due, 2),
        "monthly_equivalent": round(total_due / months, 2),
    }


def _build_loyalty_quote(students: int, teachers: int, cycle_key: str, months: int) -> dict:
    """
    Zero-cost, zero-limit quote for a school with an active loyalty
    subscription window. Mirrors the shape of the normal school quote so
    frontend code doesn't need a separate branch to render it -- it just
    checks `is_loyalty_plan` / a null `monthly_upload_limit`.
    """
    return {
        "cycle": cycle_key,
        "months": months,
        "is_parent_plan": False,
        "is_free_baseline": False,
        "is_loyalty_plan": True,
        "total_seats": {"students": students, "teachers": teachers},
        "paid_seats": {"students": 0, "teachers": 0},
        "raw_seat_monthly": 0.0,
        "platform_maintenance_fee_cycle": 0.0,
        "is_maintenance_fee_applied": False,
        "gross_subtotal_before_discount": 0.0,
        "discount_percent": 0,
        "discount_amount": 0.0,
        "subtotal_after_discount": 0.0,
        "addon_exam_packs_cost": 0.0,
        "tax_rate_percent": 0,
        "tax_amount": 0.0,
        "total_due_now": 0.0,
        "monthly_equivalent": 0.0,
        # None == unlimited. Keep this instead of a big int so the frontend
        # can render "Unlimited" instead of silently truncating.
        "monthly_upload_limit": None,
    }


def calculate_subscription_quote(
        students: int = 0,
        teachers: int = 0,
        cycle: str = "monthly",
        additional_exam_packs: int = 0,
        plan_type: str = "school",  # Added plan_type switch
        loyalty_active: bool = False,  # Set by the caller after checking Firestore
) -> dict:
    """
    Calculates subscription billing quote for School, Parent, and
    loyalty-bypassed plans.

    `loyalty_active` should already reflect a verified, still-open loyalty
    cycle -- this function does not talk to Firestore itself. See
    `is_loyalty_subscription_active()` / `redeem_loyalty_code()` below.
    """
    # Route to parent handler if plan_type is parent
    if plan_type.lower() in ["parent", "parent_access"]:
        return calculate_parent_subscription_quote(cycle=cycle)

    cycle_key = cycle.lower()
    if cycle_key not in CYCLE_MONTHS:
        raise ValueError(f"Invalid billing cycle: '{cycle}'. Must be one of {list(CYCLE_MONTHS.keys())}")

    months = CYCLE_MONTHS[cycle_key]

    # Loyalty bypass short-circuits everything below: no seat cost, no
    # maintenance fee, no VAT, no upload cap.
    if loyalty_active:
        return _build_loyalty_quote(students, teachers, cycle_key, months)

    discount_rate = DISCOUNTS[cycle_key]

    # 1. Paid seats above free baseline
    paid_students = max(0, students - FREE_STUDENT_BASE)
    paid_teachers = max(0, teachers - FREE_TEACHER_BASE)
    is_free_baseline = (paid_students <= 0 and paid_teachers <= 0)

    # 2. Monthly un-discounted raw seat costs
    raw_seat_monthly = (paid_students * RATES["student_monthly"]) + (paid_teachers * RATES["teacher_monthly"])
    raw_seat_cycle = raw_seat_monthly * months

    # 3. Dynamic Platform Maintenance Fee
    maintenance_fee_for_cycle = 0.0 if is_free_baseline else (RATES["base_platform_fee_monthly"] * months)

    # 4. Gross cycle subtotal evaluating the maintenance floor
    gross_cycle_subtotal = 0.0 if is_free_baseline else max(raw_seat_cycle, maintenance_fee_for_cycle)
    is_maintenance_fee_applied = not is_free_baseline and (raw_seat_cycle < maintenance_fee_for_cycle)

    # 5. Apply Cycle Discount
    discount_amount = gross_cycle_subtotal * discount_rate
    subtotal_after_discount = gross_cycle_subtotal - discount_amount

    # 6. Optional Add-ons
    addon_cost = additional_exam_packs * RATES["extra_exam_pack_price"]
    taxable_subtotal = subtotal_after_discount + addon_cost

    # 7. Tax (15% VAT) & Period Totals
    tax_amount = 0.0 if is_free_baseline else (taxable_subtotal * TAX_RATE)
    total_due = taxable_subtotal + tax_amount

    # 8. Monthly Exam Upload Limit Calculation
    monthly_upload_limit = max(
        (teachers * DEFAULT_EXAMS_PER_TEACHER),
        FREE_TIER_MONTHLY_LIMIT
    )

    return {
        "cycle": cycle_key,
        "months": months,
        "is_parent_plan": False,
        "is_free_baseline": is_free_baseline,
        "is_loyalty_plan": False,
        "total_seats": {
            "students": students,
            "teachers": teachers
        },
        "paid_seats": {
            "students": paid_students,
            "teachers": paid_teachers
        },
        "raw_seat_monthly": round(raw_seat_monthly, 2),
        "platform_maintenance_fee_cycle": round(maintenance_fee_for_cycle, 2),
        "is_maintenance_fee_applied": is_maintenance_fee_applied,
        "gross_subtotal_before_discount": round(gross_cycle_subtotal, 2),
        "discount_percent": int(discount_rate * 100),
        "discount_amount": round(discount_amount, 2),
        "subtotal_after_discount": round(subtotal_after_discount, 2),
        "addon_exam_packs_cost": round(addon_cost, 2),
        "tax_rate_percent": int(TAX_RATE * 100),
        "tax_amount": round(tax_amount, 2),
        "total_due_now": round(total_due, 2),
        "monthly_equivalent": round(total_due / months, 2),
        "monthly_upload_limit": monthly_upload_limit,
    }


def calculate_prorated_user_addon(
        current_seats: int,
        additional_seats: int,
        seat_type: str,
        cycle: str = "monthly",
        days_remaining: int = 30,
        total_days_in_period: int = 365
) -> dict:
    """
    Calculates prorated mid-cycle additions for new seats.
    (Loyalty-plan schools should never hit this -- they have no seat
    cost to prorate. Gate the call site on `is_loyalty_subscription_active`.)
    """
    seat_key = seat_type.lower()
    if seat_key not in ["student", "teacher"]:
        raise ValueError(f"Invalid seat_type: '{seat_type}'. Must be 'student' or 'teacher'.")

    cycle_key = cycle.lower()
    months = CYCLE_MONTHS.get(cycle_key, 1)
    discount_rate = DISCOUNTS.get(cycle_key, 0.0)

    base_limit = FREE_STUDENT_BASE if seat_key == "student" else FREE_TEACHER_BASE
    rate_per_month = RATES["student_monthly"] if seat_key == "student" else RATES["teacher_monthly"]

    previous_paid = max(0, current_seats - base_limit)
    new_total_seats = current_seats + additional_seats
    new_paid = max(0, new_total_seats - base_limit)

    newly_paid_seats = max(0, new_paid - previous_paid)

    full_cycle_cost = (newly_paid_seats * rate_per_month * months) * (1.0 - discount_rate)

    fraction_remaining = max(0.0, days_remaining / max(1, total_days_in_period))
    prorated_subtotal = full_cycle_cost * fraction_remaining

    tax_amount = prorated_subtotal * TAX_RATE
    total_due = prorated_subtotal + tax_amount

    return {
        "seat_type": seat_key,
        "additional_seats": additional_seats,
        "newly_paid_seats": newly_paid_seats,
        "total_seats_after_update": new_total_seats,
        "days_remaining": days_remaining,
        "prorated_subtotal": round(prorated_subtotal, 2),
        "tax_amount": round(tax_amount, 2),
        "prorated_amount_due": round(total_due, 2),
    }


# ─── Loyalty Code Redemption (Firestore-backed) ───────────────────────────
# These two functions are the only place that talk to the database. Keep
# calculate_subscription_quote() pure -- it just trusts the `loyalty_active`
# bool you pass it after calling is_loyalty_subscription_active().
#
# Firestore shape assumed:
#   loyaltyCodes/{code}            -> { active, expiresAt, maxRedemptions,
#                                        redeemedBySchools: [schoolId, ...] }
#   schools/{schoolId}.loyaltySubscription
#                                   -> { code, active, cycleStart, cycleEnd }

def redeem_loyalty_code(db, school_id: str, code: str) -> dict:
    """
    Validates `code` and (re)activates a LOYALTY_CYCLE_DAYS window for
    `school_id`. Call this from whatever endpoint handles the "enter your
    loyalty code" action -- both on first activation and on any later
    re-activation once a cycle has lapsed. Raises ValueError with a
    user-facing message on any invalid state.
    """
    code_ref = db.collection("loyaltyCodes").document(code)
    code_snap = code_ref.get()

    if not code_snap.exists:
        raise ValueError("Invalid loyalty code.")

    code_data = code_snap.to_dict()

    if not code_data.get("active", False):
        raise ValueError("This loyalty code has been deactivated.")

    expires_at = code_data.get("expiresAt")
    if expires_at is not None and expires_at < datetime.now(timezone.utc):
        raise ValueError("This loyalty code has expired.")

    redeemed_by = code_data.get("redeemedBySchools", [])
    max_redemptions = code_data.get("maxRedemptions")
    if (
        max_redemptions is not None
        and school_id not in redeemed_by
        and len(redeemed_by) >= max_redemptions
    ):
        raise ValueError("This loyalty code has reached its redemption limit.")

    now = datetime.now(timezone.utc)
    cycle_end = now + timedelta(days=LOYALTY_CYCLE_DAYS)

    db.collection("schools").document(school_id).set(
        {
            "loyaltySubscription": {
                "code": code,
                "active": True,
                "cycleStart": now,
                "cycleEnd": cycle_end,
            }
        },
        merge=True,
    )

    if school_id not in redeemed_by:
        # Import kept local so this module doesn't hard-require
        # firebase_admin at import time for callers that only need the
        # pure pricing functions above.
        from firebase_admin import firestore as fb_firestore
        code_ref.set(
            {"redeemedBySchools": fb_firestore.ArrayUnion([school_id])},
            merge=True,
        )

    return {
        "school_id": school_id,
        "code": code,
        "cycle_start": now.isoformat(),
        "cycle_end": cycle_end.isoformat(),
        "status": "active",
    }


def is_loyalty_subscription_active(db, school_id: str) -> bool:
    """
    True only while the school's most recent redemption window hasn't
    lapsed yet. Once cycleEnd passes without re-redemption this quietly
    goes False and normal tier limits/billing resume -- no separate
    "deactivate" step needed.
    """
    school_snap = db.collection("schools").document(school_id).get()
    if not school_snap.exists:
        return False

    loyalty = school_snap.to_dict().get("loyaltySubscription")
    if not loyalty or not loyalty.get("active"):
        return False

    cycle_end = loyalty.get("cycleEnd")
    if not isinstance(cycle_end, datetime):
        return False

    return cycle_end > datetime.now(timezone.utc)