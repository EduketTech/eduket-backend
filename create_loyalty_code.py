"""
create_loyalty_code.py

Quick CLI for minting a loyalty code without going through the admin API.
Run it wherever your service account credentials are already available
(same env this app deploys with, e.g. locally with FIREBASE_SERVICE_ACCOUNT_JSON
set, or via `render run` / a one-off shell on Render).

Usage:
    python create_loyalty_code.py PILOT2026 --max-redemptions 5 --description "Q4 pilot schools"
    python create_loyalty_code.py --max-redemptions 1          # auto-generates a code
    python create_loyalty_code.py PILOT2026 --expires-days 60  # expires 60 days from now
"""

import argparse
import json
import os
import secrets
import sys
from datetime import datetime, timedelta, timezone

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


def _generate_code(length: int = 10) -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # no O/0, I/1
    return "".join(secrets.choice(alphabet) for _ in range(length))


def main():
    parser = argparse.ArgumentParser(description="Mint a new Eduket OS loyalty code.")
    parser.add_argument("code", nargs="?", help="Code to create (omit to auto-generate).")
    parser.add_argument("--max-redemptions", type=int, default=None,
                         help="Max number of distinct schools that can redeem this code. Omit for unlimited.")
    parser.add_argument("--expires-days", type=int, default=None,
                         help="Days from now until the code itself stops being redeemable. Omit for no expiry.")
    parser.add_argument("--description", default="", help="Internal note about what this code is for.")
    args = parser.parse_args()

    db = _init_firebase()

    code = (args.code or _generate_code()).strip().upper()
    code_ref = db.collection("loyaltyCodes").document(code)

    if code_ref.get().exists:
        print(f"Code '{code}' already exists.", file=sys.stderr)
        sys.exit(1)

    expires_at = None
    if args.expires_days is not None:
        expires_at = datetime.now(timezone.utc) + timedelta(days=args.expires_days)

    code_ref.set({
        "active": True,
        "maxRedemptions": args.max_redemptions,
        "expiresAt": expires_at,
        "description": args.description,
        "redeemedBySchools": [],
        "createdAt": datetime.now(timezone.utc),
    })

    print(f"Created loyalty code: {code}")
    if args.max_redemptions is not None:
        print(f"  Max redemptions: {args.max_redemptions} schools")
    if expires_at:
        print(f"  Expires: {expires_at.isoformat()}")
    print("  Give this code to the school -- they redeem it from their billing page.")


if __name__ == "__main__":
    main()