import json

from app.memory.redis_client import redis_client

MAX_MESSAGES = 20
SESSION_TTL = 86400  # 24 hours


def _session_key(user_id: str, session_id: str) -> str:
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("Valid user_id is required.")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("Valid session_id is required.")

    return f"memory:user:{user_id.strip()}:session:{session_id.strip()}:messages"


def add_message(
    user_id: str, session_id: str, role: str, content: str
) -> dict:
    key = _session_key(user_id, session_id)

    if role not in ("user", "assistant"):
        raise ValueError("Invalid message role.")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Message content cannot be empty.")

    message = {"role": role, "content": content.strip()}

    with redis_client.pipeline(transaction=True) as pipe:
        pipe.rpush(key, json.dumps(message))
        pipe.ltrim(key, -MAX_MESSAGES, -1)
        pipe.expire(key, SESSION_TTL)
        pipe.execute()

    return message


def get_recent_messages(
    user_id: str, session_id: str, limit: int = 10
) -> list[dict]:
    if not isinstance(limit, int) or isinstance(limit, bool):
        raise ValueError("limit must be an integer.")
    if not 1 <= limit <= MAX_MESSAGES:
        raise ValueError(f"limit must be between 1 and {MAX_MESSAGES}.")

    messages = redis_client.lrange(
        _session_key(user_id, session_id), -limit, -1
    )
    return [json.loads(message) for message in messages]


def clear_session(user_id: str, session_id: str) -> bool:
    return bool(redis_client.delete(_session_key(user_id, session_id)))