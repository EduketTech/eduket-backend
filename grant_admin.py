"""
grant_admin.py

Adds a document to the `admins` collection for your own email -- this is
what app.py's require_admin (and now loyalty_routes.py's
_verify_admin_token) actually checks. No custom claim needed; your
codebase already uses this pattern for /admin/extraction-status,
/admin/trigger-extract and /admin/cleanup-sessions, so granting yourself
access here also unlocks the loyalty admin routes for the same account.

Usage:
    python grant_admin.py you@example.com
    python grant_admin.py you@example.com --revoke
"""

import argparse
import json
import os
import sys

import firebase_admin
from firebase_admin import credentials, firestore


def _init_firebase():
    if not firebase_admin._apps:
        cred_json = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON")
        if not cred_json:
            print("FIREBASE_SERVICE_ACCOUNT_JSON is not set in this environment.", file=sys.stderr)
            sys.exit(1)
        cred = credentials.Certificate(json.loads(cred_json))
        firebase_admin.initialize_app(cred)
    return firestore.client()


def main():
    parser = argparse.ArgumentParser(description="Grant or revoke admin access for an email.")
    parser.add_argument("email", help="Email address to grant/revoke admin access for.")
    parser.add_argument("--revoke", action="store_true", help="Remove admin access instead of granting it.")
    args = parser.parse_args()

    db = _init_firebase()
    admin_ref = db.collection("admins").document(args.email)

    if args.revoke:
        admin_ref.delete()
        print(f"Revoked admin access for {args.email}.")
    else:
        admin_ref.set({"grantedAt": firestore.SERVER_TIMESTAMP})
        print(f"Granted admin access to {args.email}.")

    print("Takes effect immediately -- no token refresh needed, since "
          "require_admin looks this up on every request rather than "
          "reading it off the token.")


if __name__ == "__main__":
    main()