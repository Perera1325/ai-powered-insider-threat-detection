"""Login and role-based access for the dashboard. Pure functions (no Streamlit) so they can be unit-tested.

Passwords are stored as PBKDF2-HMAC-SHA256 hashes. Accounts come from Streamlit secrets ([users.<name>] with
`password_hash` or `password`, and `role`) or from ITD_USERS_JSON; if none are configured, three demo accounts
are used (shown on the login page so a demo can be given). For a real deployment, set your own accounts.
Set ITD_AUTH=off to disable the login (local development and automated tests only).
"""
import hashlib
import hmac
import os

ROLES = ["viewer", "analyst", "admin"]
# action -> roles allowed
PERMISSIONS = {
    "view": {"viewer", "analyst", "admin"},
    "case_write": {"analyst", "admin"},     # save case status / notes
    "export": {"analyst", "admin"},         # CSV / PDF / case log downloads
    "simulate": {"analyst", "admin"},       # start stream, inject attack, attack simulation
    "admin": {"admin"},                     # audit log, restore case log, test notification
}
DEMO_PASSWORDS = {"admin": "admin123", "analyst": "analyst123", "viewer": "viewer123"}
ITERATIONS = 120_000
MAX_FAILED = 5


def hash_password(password, salt=None, iterations=ITERATIONS):
    salt = salt or os.urandom(16).hex()
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), iterations)
    return f"pbkdf2_sha256${iterations}${salt}${dk.hex()}"


def verify_password(password, stored):
    try:
        algo, iters, salt, digest = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), int(iters))
        return hmac.compare_digest(dk.hex(), digest)
    except (ValueError, AttributeError):
        return False


def demo_users():
    return {u: {"password_hash": hash_password(p), "role": u} for u, p in DEMO_PASSWORDS.items()}


def normalise_users(raw):
    """raw: {name: {password|password_hash, role}} -> {name: {password_hash, role}}; invalid entries skipped."""
    users = {}
    for name, cfg in (raw or {}).items():
        role = str(cfg.get("role", "viewer")).lower()
        if role not in ROLES:
            continue
        if cfg.get("password_hash"):
            users[name] = {"password_hash": cfg["password_hash"], "role": role}
        elif cfg.get("password"):
            users[name] = {"password_hash": hash_password(str(cfg["password"])), "role": role}
    return users


def authenticate(username, password, users):
    """Returns the role on success, else None. Runs a hash either way to keep timing similar."""
    cfg = users.get(username)
    stored = cfg["password_hash"] if cfg else hash_password("x", salt="00" * 16, iterations=1000)
    ok = verify_password(password or "", stored)
    return cfg["role"] if (cfg and ok) else None


def auth_enabled():
    return os.environ.get("ITD_AUTH", "on").lower() not in ("off", "0", "false", "no")


def demo_logins_visible():
    return os.environ.get("ITD_SHOW_DEMO_LOGINS", "on").lower() not in ("off", "0", "false", "no")


def can(role, action):
    return role in PERMISSIONS.get(action, set())
