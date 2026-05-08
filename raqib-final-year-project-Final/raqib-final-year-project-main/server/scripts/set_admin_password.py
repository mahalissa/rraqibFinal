"""
RAQIB - Admin password setup
=============================
Generates a PBKDF2-HMAC-SHA256 hash for the admin password and writes it
to the project's ``.env`` file (creating the file if missing). Existing
ADMIN_USERNAME / ADMIN_PASSWORD_HASH lines are replaced; everything else
in the file is preserved.

Usage:
    python server/scripts/set_admin_password.py             # interactive
    python server/scripts/set_admin_password.py --user me --password s3cret
"""
from __future__ import annotations

import argparse
import getpass
import secrets
import sys
from pathlib import Path

# Allow importing from the parent ``server`` package when run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from auth import hash_password  # noqa: E402

ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"


def _set_env_line(lines: list[str], key: str, value: str) -> list[str]:
    out: list[str] = []
    found = False
    for line in lines:
        if line.startswith(f"{key}="):
            out.append(f"{key}={value}")
            found = True
        else:
            out.append(line)
    if not found:
        out.append(f"{key}={value}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default=None)
    ap.add_argument("--password", default=None)
    ap.add_argument("--no-secret", action="store_true",
                    help="Don't generate a SESSION_SECRET if missing.")
    args = ap.parse_args()

    user = args.user or input("Admin username [admin]: ").strip() or "admin"
    pw = args.password
    if pw is None:
        pw = getpass.getpass("Admin password: ")
        pw2 = getpass.getpass("Confirm:        ")
        if pw != pw2:
            print("Passwords don't match.", file=sys.stderr)
            return 1
    if not pw:
        print("Password may not be empty.", file=sys.stderr)
        return 1

    pw_hash = hash_password(pw)

    if ENV_PATH.exists():
        lines = ENV_PATH.read_text().splitlines()
    else:
        lines = []
    lines = _set_env_line(lines, "ADMIN_USERNAME", user)
    lines = _set_env_line(lines, "ADMIN_PASSWORD_HASH", pw_hash)
    if not args.no_secret and not any(l.startswith("SESSION_SECRET=") for l in lines):
        lines = _set_env_line(lines, "SESSION_SECRET", secrets.token_urlsafe(32))

    ENV_PATH.write_text("\n".join(lines).rstrip() + "\n")
    print(f"Wrote {ENV_PATH}")
    print(f"  ADMIN_USERNAME={user}")
    print(f"  ADMIN_PASSWORD_HASH=<{len(pw_hash)} chars, hidden>")
    print("\nRestart the server (or `docker compose up`) to apply.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
