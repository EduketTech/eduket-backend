# gunicorn.conf.py
import os

workers     = 1
worker_class = "gevent"          # ← was "sync" — match Dockerfile
bind        = f"0.0.0.0:{os.environ.get('PORT', '10000')}"
timeout     = 300                # ← was 120 — increase for extraction
keepalive   = 5
loglevel    = "info"

def post_fork(server, worker):
    """
    Reinitialize Firebase in each worker after gunicorn forks.
    Fixes gRPC incompatibility with gunicorn's fork-based workers.
    """
    import firebase_admin
    import app as flask_app

    # Wipe all existing Firebase apps inherited from parent process
    for existing_app in list(firebase_admin._apps.values()):
        firebase_admin.delete_app(existing_app)

    # Reset module-level db and bucket
    flask_app.db     = None
    flask_app.bucket = None

    # Reinitialize cleanly in this worker
    flask_app._init_firebase()
    print(f"[post_fork] ✅ Firebase reinitialized in worker pid={worker.pid}",
          flush=True)