"""Username / password login. No extra packages needed.

Passwords are never stored. Only a salted PBKDF2-SHA256 hash is kept in the users table.
After 5 wrong passwords for one username, that username is locked for 5 minutes
(the counter lives in the running app, so it resets if the app is restarted).
"""
import datetime
import hashlib
import hmac
import re
import secrets
import time

from sqlalchemy import delete, func, insert, select, update

from src import db

ITERATIONS = 200_000
MAX_FAILS, LOCK_SECONDS = 5, 300
MIN_PASSWORD = 8
_FAILS = {}                                   # username -> (failed count, time of last failure)


def _hash(password: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, ITERATIONS)
    return f"pbkdf2_sha256${ITERATIONS}${salt.hex()}${dk.hex()}"


def _check(password: str, stored: str) -> bool:
    try:
        _, iters, salt, dk = stored.split("$")
        new = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"),
                                  bytes.fromhex(salt), int(iters))
        return hmac.compare_digest(new.hex(), dk)
    except Exception:
        return False


_DUMMY = _hash("not-a-real-password")         # used so unknown usernames take equally long


def _clean_username(username: str) -> str:
    u = (username or "").strip().lower()
    if not re.fullmatch(r"[a-z0-9_.-]{3,64}", u):
        raise ValueError("Username must be 3 to 64 characters: letters, digits, dot, dash or underscore.")
    return u


def _check_password_rules(password: str):
    if len(password or "") < MIN_PASSWORD:
        raise ValueError(f"Password must be at least {MIN_PASSWORD} characters.")


# ------------------------------------------------------------------ administration
def create_user(username: str, password: str):
    u = _clean_username(username)
    _check_password_rules(password)
    with db.engine.begin() as con:
        if con.execute(select(db.users.c.id).where(db.users.c.username == u)).first():
            raise ValueError(f"User '{u}' already exists.")
        con.execute(insert(db.users).values(
            username=u, password_hash=_hash(password),
            created_at=datetime.datetime.now(), active=1))


def set_password(username: str, password: str):
    u = _clean_username(username)
    _check_password_rules(password)
    with db.engine.begin() as con:
        res = con.execute(update(db.users).where(db.users.c.username == u)
                          .values(password_hash=_hash(password)))
        if res.rowcount == 0:
            raise ValueError(f"User '{u}' not found.")
    _FAILS.pop(u, None)


def delete_user(username: str):
    u = _clean_username(username)
    with db.engine.begin() as con:
        res = con.execute(delete(db.users).where(db.users.c.username == u))
        if res.rowcount == 0:
            raise ValueError(f"User '{u}' not found.")


def list_users():
    with db.engine.connect() as con:
        rows = con.execute(select(db.users.c.username, db.users.c.created_at)
                           .order_by(db.users.c.username)).all()
    return [(r[0], r[1]) for r in rows]


def user_count() -> int:
    with db.engine.connect() as con:
        return int(con.execute(select(func.count()).select_from(db.users)).scalar())


# ------------------------------------------------------------------ login
def login(username: str, password: str):
    """Returns (ok, message)."""
    u = (username or "").strip().lower()
    now = time.time()
    count, last = _FAILS.get(u, (0, 0.0))
    if count >= MAX_FAILS:
        if now - last < LOCK_SECONDS:
            return False, "Too many failed attempts. Try again in a few minutes."
        count = 0                                          # lock has expired

    with db.engine.connect() as con:
        row = con.execute(select(db.users.c.password_hash)
                          .where(db.users.c.username == u, db.users.c.active == 1)).first()
    ok = _check(password or "", row[0] if row else _DUMMY) and row is not None
    if ok:
        _FAILS.pop(u, None)
        return True, ""
    _FAILS[u] = (count + 1, now)
    return False, "Wrong username or password."
