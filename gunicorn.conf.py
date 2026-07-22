# gunicorn.conf.py
import os
import gevent.monkey
gevent.monkey.patch_all()

workers     = 1
worker_class = "gevent"          # ← was "sync" — match Dockerfile
bind        = f"0.0.0.0:{os.environ.get('PORT', '10000')}"
timeout     = 300                # ← was 120 — increase for extraction
keepalive   = 5
loglevel    = "info"

# gunicorn.conf.py — update post_fork to not re-run startup tasks

def post_fork(server, worker):
    import firebase_admin
    import app as flask_app

    # Wipe inherited Firebase apps
    for existing_app in list(firebase_admin._apps.values()):
        firebase_admin.delete_app(existing_app)

    flask_app.db     = None
    flask_app.bucket = None

    # Reinitialize Firebase only — do NOT re-run listener or sweep
    flask_app._init_firebase()
    print(f"[post_fork] ✅ Firebase reinitialized in worker pid={worker.pid}",
          flush=True)