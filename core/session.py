"""
core/session.py
==================
Session ID management (cookie-based) and simple in-memory rate limiting.

NOTE: in-memory state means this resets on server restart and does not
share state across multiple server processes/workers. Acceptable for a
portfolio-scale, single-instance deployment; a real production system
at scale would use Redis instead. Documented as a known limitation.
"""

import time
import uuid
from dataclasses import dataclass, field

SESSION_COOKIE_NAME = "session_id"
MAX_REQUESTS_PER_MINUTE = 10
MAX_QUERIES_PER_SESSION = 30


@dataclass
class SessionUsage:
    query_count: int = 0
    request_timestamps: list = field(default_factory=list)
    total_bytes_uploaded: int = 0

MAX_SESSION_UPLOAD_MB = 50
_usage_store: dict[str, SessionUsage] = {}


def new_session_id() -> str:
    return uuid.uuid4().hex  # 32 lowercase alnum chars — safe for Chroma collection names AND Pinecone namespaces


def get_usage(session_id: str) -> SessionUsage:
    if session_id not in _usage_store:
        _usage_store[session_id] = SessionUsage()
    return _usage_store[session_id]


def check_rate_limit(session_id: str) -> tuple[bool, str]:
    """Returns (allowed, reason_if_not)."""
    usage = get_usage(session_id)
    now = time.time()

    usage.request_timestamps = [t for t in usage.request_timestamps if now - t < 60]
    if len(usage.request_timestamps) >= MAX_REQUESTS_PER_MINUTE:
        return False, f"Rate limit exceeded: max {MAX_REQUESTS_PER_MINUTE} requests/minute"

    if usage.query_count >= MAX_QUERIES_PER_SESSION:
        return False, f"Session query limit reached ({MAX_QUERIES_PER_SESSION} max) — clear session to continue"

    usage.request_timestamps.append(now)
    return True, ""


def record_query(session_id: str):
    get_usage(session_id).query_count += 1


def clear_session_usage(session_id: str):
    _usage_store.pop(session_id, None)

def check_upload_limit(session_id: str, new_file_size_bytes: int) -> tuple[bool, str]:
    """Checks whether adding this file would exceed the session's cumulative upload cap."""
    usage = get_usage(session_id)
    projected_mb = (usage.total_bytes_uploaded + new_file_size_bytes) / (1024 * 1024)
    if projected_mb > MAX_SESSION_UPLOAD_MB:
        used_mb = usage.total_bytes_uploaded / (1024 * 1024)
        return False, f"Session upload limit reached: {used_mb:.1f}MB used, {MAX_SESSION_UPLOAD_MB}MB max. Clear session to continue."
    return True, ""

def record_upload(session_id: str, file_size_bytes: int):
    get_usage(session_id).total_bytes_uploaded += file_size_bytes