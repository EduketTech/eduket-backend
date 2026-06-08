# gunicorn.conf.py
import os
import sys

workers = 1
worker_class = "sync"
bind = "0.0.0.0:10000"
timeout = 120
keepalive = 5

def post_fork(server, worker):
    """
    Reinitialize Firebase in each worker after gunicorn forks.
    Fixes gRPC incompatibility with gunicorn's fork-based workers.
    """
    import firebase_admin
    import app as flask_app

    # Wipe all existing Firebase apps inherited from the parent process
    for existing_app in list(firebase_admin._apps.values()):
        firebase_admin.delete_app(existing_app)

    # Reset the module-level db and bucket to None
    flask_app.db = None
    flask_app.bucket = None

    # Reinitialize cleanly in this worker process
    flask_app._init_firebase()

    print(f"[post_fork] ✅ Firebase reinitialized in worker pid={worker.pid}", flush=True)