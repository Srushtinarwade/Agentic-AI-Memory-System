import json
import uuid
from datetime import datetime, timezone

from redis.exceptions import WatchError

from app.memory.redis_client import redis_client

MEMORY_TYPES = {"semantic", "episodic"}


def _text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Valid {name} is required.")
    return value.strip()


def _key(user_id: str, memory_id: str) -> str:
    user_id = _text(user_id, "user_id")
    memory_id = _text(memory_id, "memory_id")
    return f"memory:user:{user_id}:long_term:{memory_id}"


def _index(user_id: str) -> str:
    return f"memory:user:{_text(user_id, 'user_id')}:long_term:index"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def add_memory(
    user_id: str,
    content: str,
    memory_type: str,
    metadata: dict | None = None,
) -> dict:
    user_id = _text(user_id, "user_id")
    content = _text(content, "content")
    memory_type = _text(memory_type, "memory_type")

    if memory_type not in MEMORY_TYPES:
        raise ValueError("memory_type must be semantic or episodic.")
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError("metadata must be a dictionary.")

    now = _now()
    memory = {
        "id": str(uuid.uuid4()),
        "user_id": user_id,
        "type": memory_type,
        "content": content,
        "metadata": metadata or {},
        "created_at": now,
        "updated_at": now,
    }

    with redis_client.pipeline(transaction=True) as pipe:
        pipe.set(_key(user_id, memory["id"]),
                 json.dumps(memory, ensure_ascii=False))
        pipe.sadd(_index(user_id), memory["id"])
        pipe.execute()

    return memory


def get_memory(user_id: str, memory_id: str) -> dict | None:
    value = redis_client.get(_key(user_id, memory_id))
    return json.loads(value) if value else None


def list_memories(
    user_id: str, memory_type: str | None = None
) -> list[dict]:
    user_id = _text(user_id, "user_id")

    if memory_type is not None and memory_type not in MEMORY_TYPES:
        raise ValueError("Invalid memory_type.")

    ids = redis_client.smembers(_index(user_id))
    if not ids:
        return []

    values = redis_client.mget([_key(user_id, mid) for mid in ids])
    memories = [json.loads(v) for v in values if v]

    if memory_type:
        memories = [m for m in memories if m["type"] == memory_type]

    return sorted(memories, key=lambda m: m["created_at"], reverse=True)


def update_memory(
    user_id: str,
    memory_id: str,
    content: str | None = None,
    metadata: dict | None = None,
) -> dict | None:
    key = _key(user_id, memory_id)

    if content is not None:
        content = _text(content, "content")
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError("metadata must be a dictionary.")

    while True:
        with redis_client.pipeline() as pipe:
            try:
                pipe.watch(key)
                value = pipe.get(key)

                if value is None:
                    return None

                memory = json.loads(value)
                if content is not None:
                    memory["content"] = content
                if metadata is not None:
                    memory["metadata"] = metadata

                memory["updated_at"] = _now()
                pipe.multi()
                pipe.set(key, json.dumps(memory, ensure_ascii=False))
                pipe.execute()
                return memory

            except WatchError:
                continue


def forget_memory(user_id: str, memory_id: str) -> bool:
    key = _key(user_id, memory_id)

    with redis_client.pipeline(transaction=True) as pipe:
        pipe.delete(key)
        pipe.srem(_index(user_id), memory_id)
        result = pipe.execute()

    return bool(result[0])