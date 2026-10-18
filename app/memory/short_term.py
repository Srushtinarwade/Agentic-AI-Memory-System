"""Small, process-local conversation buffer (no Redis service needed).

Conversation turns expire after 24 hours and are lost when the API stops. Durable
facts live in Neo4j. Run the app with ONE uvicorn worker on a laptop.
"""
from __future__ import annotations

import threading
import time
from collections import deque

MAX_MESSAGES = 20
SESSION_TTL = 24 * 60 * 60
_lock = threading.RLock()
_sessions: dict[tuple[str, str], tuple[float, deque[dict[str, str]]]] = {}


def _session_key(user_id: str, session_id: str) -> tuple[str, str]:
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("Valid user_id is required.")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("Valid session_id is required.")
    return user_id.strip(), session_id.strip()


def add_message(user_id: str, session_id: str, role: str, content: str) -> dict:
    key = _session_key(user_id, session_id)
    if role not in {"user", "assistant"}:
        raise ValueError("Invalid message role.")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Message content cannot be empty.")
    message = {"role": role, "content": content.strip()}
    with _lock:
        expiry, turns = _sessions.get(key, (0, deque(maxlen=MAX_MESSAGES)))
        if expiry <= time.monotonic():
            turns = deque(maxlen=MAX_MESSAGES)
        turns.append(message)
        _sessions[key] = (time.monotonic() + SESSION_TTL, turns)
    return message.copy()


def get_recent_messages(user_id: str, session_id: str, limit: int = 10) -> list[dict]:
    key = _session_key(user_id, session_id)
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= MAX_MESSAGES:
        raise ValueError(f"limit must be between 1 and {MAX_MESSAGES}.")
    with _lock:
        expiry, turns = _sessions.get(key, (0, ()))
        if expiry <= time.monotonic():
            _sessions.pop(key, None)
            return []
        return [dict(turn) for turn in list(turns)[-limit:]]


def clear_session(user_id: str, session_id: str) -> bool:
    with _lock:
        return _sessions.pop(_session_key(user_id, session_id), None) is not None


def shutdown() -> None:
    """Clear ephemeral conversation data during FastAPI graceful shutdown."""
    with _lock:
        _sessions.clear()
