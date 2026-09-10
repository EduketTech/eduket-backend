"""
tier_limits.py — Dynamic Seat-Based Quotas & Usage Tracking
═══════════════════════════════════════════════════════════════════════════════
Replaces hardcoded tiers with dynamic seat-based calculations (Students + Teachers).
Uses server-side Firestore aggregation (.count()) for high-performance usage reads.
"""

import logging
import os
import threading
from datetime import datetime, timezone

from firebase_admin import firestore
from google.cloud.firestore_v1.base_query import FieldFilter

# Loyalty bypass -- see pricing.py. This is the actual gatekeeper behind
# /api/register-user and /check-tier-limit in app.py, so a school with an
# active loyalty cycle needs to be exempted HERE, not just in app.py's own
# check_school_exam_quota (which only governs exam uploads, not seats).
from pricing import is_loyalty_subscription_active

log = logging.getLogger(__name__)

FIRESTORE_TIMEOUT = 8.0

# ── Baseline Allocations & Quotas ────────────────────────────────────────────
FREE_STUDENT_BASE = 10  # Included free student baseline
FREE_TEACHER_BASE = 2   # Included free teacher baseline

DEFAULT_EXAMS_PER_TEACHER = 50  # Monthly exam quota generated per teacher seat
FREE_TIER_MONTHLY_LIMIT = 100   # Free/trial tier baseline monthly upload quota

# Exams are scoped to the current UTC calendar month; headcounts are standing total seats
MONTHLY_SCOPED = {"exams", "exam"}

# ── Lazy Per-Process Firestore Client ─────────────────────────────────────────
_db = None
_db_lock = threading.Lock()


def get_db():
    global _db
    if _db is None:
        with _db_lock:
            if _db is None:
                _db = firestore.client()
                log.info("Firestore client created in pid %s", os.getpid())
    return _db


# ── Internal Helpers ──────────────────────────────────────────────────────────

def _count(query) -> int:
    """
    Server-side count aggregation query. Bills 1 read per 1,000 documents
    instead of pulling full document payloads into memory.
    """
    try:
        result = query.count().get(timeout=FIRESTORE_TIMEOUT)
        return int(result[0][0].value)
    except AttributeError:
        # Fallback for older google-cloud-firestore SDKs lacking aggregation
        log.warning("Aggregation query unavailable, falling back to stream()")
        return sum(1 for _ in query.stream(timeout=FIRESTORE_TIMEOUT))


def _get_month_bounds():
    """
    Returns (iso_string, datetime_obj) for the 1st day of current UTC month.
    Supports querying whether uploadedAt is stored as string or Timestamp.
    """
    now = datetime.now(timezone.utc)
    start_dt = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
    return start_dt.isoformat(), start_dt


def _is_loyalty_active(school_id: str) -> bool:
    """Shared loyalty check for every branch of check_school_limit() below.
    Fails closed (returns False, i.e. normal limits apply) on any error
    rather than accidentally granting unlimited access on a Firestore
    hiccup."""
    if not school_id:
        return False
    try:
        return is_loyalty_subscription_active(get_db(), school_id)
    except Exception as e:
        log.error("[Loyalty] Status check failed for school %s: %s", school_id, e)
        return False


# ── Dynamic Limit Calculation Engine ─────────────────────────────────────────

def get_school_exam_limit(school_id: str) -> int:
    """
    Calculates monthly exam upload limit based on teacher headcount / seat allocations
    or returns custom/overridden limit if set on the school subscription.
    """
    if not school_id:
        return FREE_TIER_MONTHLY_LIMIT

    try:
        db = get_db()
        sub_doc = db.collection("subscriptions").document(school_id).get(timeout=FIRESTORE_TIMEOUT)

        if sub_doc.exists:
            sub_data = sub_doc.to_dict() or {}

            # 1. Custom explicit limit override takes precedence if set
            if "customExamLimit" in sub_data and sub_data["customExamLimit"] is not None:
                return int(sub_data["customExamLimit"])

            # 2. Seat-based dynamic calculation
            seats = sub_data.get("seats", {})
            teachers = int(seats.get("teachers", FREE_TEACHER_BASE))

            calculated_limit = teachers * DEFAULT_EXAMS_PER_TEACHER
            return max(calculated_limit, FREE_TIER_MONTHLY_LIMIT)

        return FREE_TIER_MONTHLY_LIMIT

    except Exception as e:
        log.error("[Quota Calculation] Error calculating exam limit for school %s: %s", school_id, e)
        return FREE_TIER_MONTHLY_LIMIT


def count_school_usage(school_id: str, resource: str) -> int:
    """
    Single source of truth for counting resource usage across all endpoints.

    Supported resources:
      - 'exams' or 'exam' (Counts uploads in the current UTC calendar month)
      - 'teachers' or 'teacher' (Counts active registered teacher profiles)
      - 'students' or 'student' (Counts active registered student profiles)
    """
    if not school_id:
        return 0

    db = get_db()
    res = resource.lower().rstrip("s")

    if res == "exam":
        iso_start, dt_start = _get_month_bounds()
        # Filter for current month exams
        q = (db.collection("exams")
             .where(filter=FieldFilter("schoolId", "==", school_id))
             .where(filter=FieldFilter("uploadedAt", ">=", iso_start)))
        return _count(q)

    # For 'teacher' or 'student' headcounts
    q = (db.collection("users")
         .where(filter=FieldFilter("schoolId", "==", school_id))
         .where(filter=FieldFilter("role", "==", res)))
    return _count(q)


def check_school_exam_quota(school_id: str) -> tuple[bool, int, int]:
    """
    Evaluates current month exam upload quota.
    Returns: (can_upload: bool, used: int, limit: int)

    A school with an active loyalty cycle bypasses the limit entirely --
    limit is reported as -1 to signal "unlimited" to callers, matching the
    convention used in app.py's own check_school_exam_quota.
    """
    if _is_loyalty_active(school_id):
        used = count_school_usage(school_id, "exams")
        return True, used, -1

    limit = get_school_exam_limit(school_id)
    used = count_school_usage(school_id, "exams")
    return (used < limit), used, limit


# ── Master Gatekeeper Evaluation ──────────────────────────────────────────────

def check_school_limit(school_id: str, limit_type: str) -> tuple[bool, str]:
    """
    Evaluates capacity for a requested resource/seat or exam upload.
    Used for pre-checks and authoritative write guardrails.

    Loyalty bypass is checked ONCE, up front, covering all three resource
    types (teacher, student, exam) -- this is the function actually behind
    /api/register-user and /check-tier-limit in app.py, so this is the
    real enforcement point for seat limits, not just exam uploads.
    """
    if not school_id:
        return False, "No school ID associated with request."

    if _is_loyalty_active(school_id):
        return True, "Allowed (loyalty access active)"

    db = get_db()
    res = limit_type.lower().rstrip("s")

    # 1. Fetch subscription details
    try:
        sub_doc = db.collection("subscriptions").document(school_id).get(timeout=FIRESTORE_TIMEOUT)
        sub_data = sub_doc.to_dict() if sub_doc.exists else {}
    except Exception as e:
        log.error("[Limit Check] Subscription lookup failed for school %s: %s", school_id, e)
        return False, "Unable to verify school subscription status."

    status = sub_data.get("status", "free")
    seats = sub_data.get("seats", {})

    # Use purchased seat numbers if present; fall back to free baseline allocations
    purchased_students = int(seats.get("students", FREE_STUDENT_BASE))
    purchased_teachers = int(seats.get("teachers", FREE_TEACHER_BASE))

    # --------------------------------------------------------------------------
    # CHECK 1: Teacher Registration Seats
    # --------------------------------------------------------------------------
    if res == "teacher":
        teacher_count = count_school_usage(school_id, "teachers")

        if teacher_count >= purchased_teachers:
            return False, f"Teacher seat limit reached ({teacher_count}/{purchased_teachers}). Please upgrade your seat allocation."

        return True, "Allowed"

    # --------------------------------------------------------------------------
    # CHECK 2: Student Registration Seats
    # --------------------------------------------------------------------------
    elif res == "student":
        student_count = count_school_usage(school_id, "students")

        if student_count >= purchased_students:
            return False, f"Student seat limit reached ({student_count}/{purchased_students}). Please upgrade your seat allocation."

        return True, "Allowed"

    # --------------------------------------------------------------------------
    # CHECK 3: Exam Monthly Generation/Upload Quota
    # --------------------------------------------------------------------------
    elif res == "exam":
        can_upload, used, limit = check_school_exam_quota(school_id)
        if not can_upload:
            return False, f"Monthly exam quota reached ({used}/{limit}). Upgrade seats to increase quota."
        return True, "Allowed"

    return True, "Allowed"