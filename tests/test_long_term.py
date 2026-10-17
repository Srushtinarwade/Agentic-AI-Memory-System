import uuid

from app.memory.long_term import (
    add_memory,
    get_memory,
    list_memories,
    update_memory,
    forget_memory,
)


def new_user():
    return f"test_user_{uuid.uuid4().hex[:8]}"


def test_add_and_retrieve_memory():
    user_id = new_user()
    memory = add_memory(user_id, "I like Python", "semantic")

    assert memory is not None
    memory_id = memory["id"]

    retrieved = get_memory(user_id, memory_id)
    assert retrieved is not None
    assert retrieved["content"] == "I like Python"

    forget_memory(user_id, memory_id)


def test_update_memory():
    user_id = new_user()
    memory = add_memory(user_id, "I like Python", "semantic")
    memory_id = memory["id"]

    try:
        updated = update_memory(
            user_id, memory_id, "I like Python and AI"
        )

        retrieved = get_memory(user_id, memory_id)
        assert retrieved["content"] == "I like Python and AI"
    finally:
        forget_memory(user_id, memory_id)


def test_list_memories():
    user_id = new_user()
    memory = add_memory(user_id, "I enjoy machine learning", "semantic")

    try:
        memories = list_memories(user_id)

        assert any(item["id"] == memory["id"] for item in memories)
    finally:
        forget_memory(user_id, memory["id"])


def test_users_cannot_access_each_others_memories():
    user1 = new_user()
    user2 = new_user()

    memory = add_memory(user1, "I like cricket", "semantic")
    memory_id = memory["id"]

    try:
        assert get_memory(user1, memory_id) is not None
        assert get_memory(user2, memory_id) is None
    finally:
        forget_memory(user1, memory_id)


def test_delete_memory():
    user_id = new_user()
    memory = add_memory(user_id, "Temporary memory", "semantic")
    memory_id = memory["id"]

    forget_memory(user_id, memory_id)

    assert get_memory(user_id, memory_id) is None
    