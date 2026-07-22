# gunicorn.conf.py
import os

workers      = 1
worker_class = "gevent"
bind         = f"0.0.0.0:{os.environ.get('PORT', '10000')}"
timeout      = 300
keepalive    = 5
loglevel     = "info"

def post_fork(server, worker):
    """
    Reinitialize Firebase after gunicorn forks.
    Does NOT import app — that would re-run module-level startup code.
    Instead reinitializes Firebase directly using the same env vars.
    """
    import json
    import firebase_admin
    from firebase_admin import credentials, firestore as fs_admin, storage

    # Wipe inherited Firebase apps from parent process
    for existing_app in list(firebase_admin._apps.values()):
        try:
            firebase_admin.delete_app(existing_app)
        except Exception:
            pass

    # Reinitialize cleanly
    try:
        raw = (
            os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON") or
            os.environ.get("FIREBASE_SERVICE_ACCOUNT") or
            ""
        ).strip()

        if raw:
            sa_dict = json.loads(raw)
            if "private_key" in sa_dict:
                sa_dict["private_key"] = sa_dict["private_key"].replace("\\n", "\n")
            cred = credentials.Certificate(sa_dict)
        else:
            cred = credentials.Certificate("serviceAccountKey.json")

        firebase_admin.initialize_app(cred, {
            "storageBucket": os.environ.get(
                "FIREBASE_STORAGE_BUCKET",
                "eduket.firebasestorage.app"
            )
        })

        # Update app.py module-level globals
        import app as flask_app
        flask_app.db     = fs_admin.client()
        flask_app.bucket = storage.bucket()

        print(f"[post_fork] ✅ Firebase ready in worker pid={worker.pid}", flush=True)

    except Exception as e:
        print(f"[post_fork] ❌ Firebase init failed: {e}", flush=True)