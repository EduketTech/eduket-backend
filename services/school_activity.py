"""
services/school_activity.py
Handles fetching and updating principal activity audit logs.
"""

import traceback
from datetime import datetime
from flask import request, jsonify
from google.cloud.firestore_v1.base_query import FieldFilter

def get_school_activity_handler(db):
    if request.method == "OPTIONS":
        return "", 200

    try:
        school_id = request.args.get("schoolId", "").strip()
        if not school_id:
            return jsonify({"error": "schoolId required"}), 400

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
            if "requires an index" in str(index_err) or "FAILED_PRECONDITION" in str(index_err):
                raw_docs = list(
                    db.collection("schoolActivity")
                      .where(filter=FieldFilter("schoolId", "==", school_id))
                      .limit(100)
                      .stream()
                )
                raw_docs.sort(
                    key=lambda doc: doc.to_dict().get("timestamp") or datetime.min,
                    reverse=True
                )
                raw_docs = raw_docs[:50]
            else:
                raise index_err

        events = []
        for doc in raw_docs:
            d = doc.to_dict()
            ts = d.get("timestamp")
            ts_formatted = ts.isoformat() if hasattr(ts, "isoformat") else str(ts or "")

            events.append({
                "id":          doc.id,
                "type":        d.get("type",        ""),
                "actorName":   d.get("actorName",   ""),
                "actorEmail":  d.get("actorEmail",  ""),
                "actorRole":   d.get("actorRole",   ""),
                "description": d.get("description", ""),
                "timestamp":   ts_formatted,
                "read":        d.get("read",        False),
            })

        return jsonify({"events": events, "success": True})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def mark_activity_read_handler(db):
    if request.method == "OPTIONS":
        return "", 200

    try:
        data      = request.get_json() or {}
        event_ids = data.get("eventIds", [])

        if not event_ids or not isinstance(event_ids, list):
            return jsonify({"success": True, "updated": 0})

        CHUNK_SIZE = 500
        for i in range(0, len(event_ids), CHUNK_SIZE):
            chunk = event_ids[i:i + CHUNK_SIZE]
            batch = db.batch()
            for eid in chunk:
                if eid:
                    doc_ref = db.collection("schoolActivity").document(eid)
                    batch.update(doc_ref, {"read": True})
            batch.commit()

        return jsonify({"success": True, "updated": len(event_ids)})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500