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
                        # Return timezone-aware epoch so comparison works
                        return datetime(1970, 1, 1, tzinfo=timezone.utc)
                    # Firestore DatetimeWithNanoseconds is timezone-aware
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

            # Safely format timestamp
            try:
                ts_formatted = ts.isoformat() if hasattr(ts, "isoformat") else str(ts or "")
            except Exception:
                ts_formatted = ""

            events.append({
                "id":          doc.id,
                "type":        d.get("type",        ""),
                "actorUid":    d.get("actorUid",    ""),
                "actorName":   d.get("actorName",   ""),
                "actorEmail":  d.get("actorEmail",  ""),
                "actorRole":   d.get("actorRole",   ""),
                "description": d.get("description", ""),
                "grade":       d.get("grade",       ""),     # student grade
                "subjects":    d.get("subjects",    []),     # student/teacher subjects
                "timestamp":   ts_formatted,
                "read":        d.get("read",        False),
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

        # Firestore batch limit is 500 writes — chunk for large lists
        CHUNK_SIZE = 500
        total_updated = 0

        for i in range(0, len(event_ids), CHUNK_SIZE):
            chunk = event_ids[i:i + CHUNK_SIZE]
            batch = db.batch()
            for eid in chunk:
                if eid and isinstance(eid, str):
                    doc_ref = db.collection("schoolActivity").document(eid)
                    batch.update(doc_ref, {"read": True})
                    total_updated += 1
            batch.commit()

        return jsonify({"success": True, "updated": total_updated})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500