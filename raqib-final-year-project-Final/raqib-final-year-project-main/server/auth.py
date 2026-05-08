"""
RAQIB - Admin authentication
=============================
Simple session-cookie auth. PBKDF2-HMAC-SHA256 password hashing (200k
iterations, stdlib-only — no bcrypt / passlib dependency), signed
session tokens via HMAC-SHA256.

The whole module is configured by environment variables:

    ADMIN_USERNAME       — e.g. "admin"
    ADMIN_PASSWORD_HASH  — output of ``hash_password(plaintext)``
    SESSION_SECRET       — long random string used to sign session
                           cookies; if absent, falls back to PUSHER_SECRET
                           or a process-life random secret (which means
                           sessions are invalidated on restart).
    SESSION_TTL_S        — token lifetime in seconds, default 86400 (24h)

If ``ADMIN_PASSWORD_HASH`` is unset the auth check is *disabled* — the
server logs a loud warning at startup and treats every request as an
authenticated admin. This keeps local dev frictionless; production
deployments are expected to set the hash via the helper script
``server/scripts/set_admin_password.py``.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import time

logger = logging.getLogger(__name__)

COOKIE_NAME = "raqib_session"
_DEFAULT_TTL_S = 24 * 60 * 60
_PBKDF2_ITERATIONS = 200_000


# ── Password hashing ────────────────────────────────────────────────────
def hash_password(plain: str) -> str:
    """Hash a plaintext password into a self-describing string.

    Format: ``pbkdf2_sha256$<iterations>$<salt_b64>$<hash_b64>``
    """
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", plain.encode("utf-8"), salt, _PBKDF2_ITERATIONS,
    )
    return (
        f"pbkdf2_sha256${_PBKDF2_ITERATIONS}$"
        f"{base64.urlsafe_b64encode(salt).decode()}${base64.urlsafe_b64encode(digest).decode()}"
    )


def verify_password(plain: str, stored: str) -> bool:
    try:
        algo, iters_s, salt_b64, hash_b64 = stored.split("$")
    except ValueError:
        return False
    if algo != "pbkdf2_sha256":
        return False
    try:
        iters = int(iters_s)
        salt = base64.urlsafe_b64decode(salt_b64)
        expected = base64.urlsafe_b64decode(hash_b64)
    except Exception:
        return False
    actual = hashlib.pbkdf2_hmac("sha256", plain.encode("utf-8"), salt, iters)
    return hmac.compare_digest(expected, actual)


# ── Session secret (lazily resolved so .env load order doesn't matter) ──
_RUNTIME_SECRET: bytes | None = None


def _session_secret() -> bytes:
    global _RUNTIME_SECRET
    explicit = os.environ.get("SESSION_SECRET")
    if explicit:
        return explicit.encode("utf-8")
    fallback = os.environ.get("PUSHER_SECRET")
    if fallback:
        return ("session-derived-from-pusher-" + fallback).encode("utf-8")
    if _RUNTIME_SECRET is None:
        _RUNTIME_SECRET = secrets.token_bytes(32)
        logger.warning(
            "[auth] SESSION_SECRET not set — using a process-lifetime random "
            "secret, so existing sessions will not survive a server restart."
        )
    return _RUNTIME_SECRET


def _ttl_seconds() -> int:
    raw = os.environ.get("SESSION_TTL_S")
    try:
        return int(raw) if raw else _DEFAULT_TTL_S
    except ValueError:
        return _DEFAULT_TTL_S


# ── Token issuance + verification ───────────────────────────────────────
def make_token(username: str) -> str:
    payload = json.dumps(
        {"u": username, "exp": int(time.time()) + _ttl_seconds()},
        separators=(",", ":"),
    ).encode("utf-8")
    payload_b64 = base64.urlsafe_b64encode(payload).rstrip(b"=").decode()
    sig = hmac.new(_session_secret(), payload_b64.encode(), hashlib.sha256).digest()
    sig_b64 = base64.urlsafe_b64encode(sig).rstrip(b"=").decode()
    return f"{payload_b64}.{sig_b64}"


def _b64decode_padded(s: str) -> bytes:
    padding = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + padding)


def verify_token(token: str | None) -> str | None:
    if not token or "." not in token:
        return None
    payload_b64, sig_b64 = token.rsplit(".", 1)
    try:
        sig = _b64decode_padded(sig_b64)
        expected = hmac.new(
            _session_secret(), payload_b64.encode(), hashlib.sha256,
        ).digest()
        if not hmac.compare_digest(sig, expected):
            return None
        payload = json.loads(_b64decode_padded(payload_b64).decode("utf-8"))
    except Exception:
        return None
    if int(payload.get("exp", 0)) < int(time.time()):
        return None
    return payload.get("u") or None


# ── High-level auth state ───────────────────────────────────────────────
def auth_enabled() -> bool:
    """Auth gate is on iff an ADMIN_PASSWORD_HASH has been configured."""
    return bool(os.environ.get("ADMIN_PASSWORD_HASH"))


def check_credentials(username: str, password: str) -> bool:
    expected_user = os.environ.get("ADMIN_USERNAME", "admin")
    expected_hash = os.environ.get("ADMIN_PASSWORD_HASH", "")
    if not expected_hash:
        return False
    if not hmac.compare_digest(username, expected_user):
        return False
    return verify_password(password, expected_hash)


def request_is_authed(cookie_value: str | None) -> bool:
    """Returns True if either auth is disabled, or the cookie is valid."""
    if not auth_enabled():
        return True
    return verify_token(cookie_value) is not None
