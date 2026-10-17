import uuid

import pytest

from app.memory.short_term import (
    add_message,
    get_recent_messages,
    clear_session,
)


@pytest.fixture
def session_ids():
    """Create separate IDs for each test."""
    user_id = f"pytest_user_{uuid.uuid4().hex}"
    session_a = "session_a"
    session_b = "session_b"

    yield user_id, session_a, session_b

    # Clean up test data after each test.
    clear_session(user_id, session_a)
    clear_session(user_id, session_b)


def test_add_and_retrieve_messages(session_ids):
    user_id, session_id, _ = session_ids

    add_message(user_id, session_id, "user", "Hello")
    add_message(user_id, session_id, "assistant", "Hi!")

    messages = get_recent_messages(user_id, session_id)

    assert messages == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"},
    ]


def test_sessions_are_separate(session_ids):
    user_id, session_a, session_b = session_ids

    add_message(user_id, session_a, "user", "Message for A")
    add_message(user_id, session_b, "user", "Message for B")

    messages_a = get_recent_messages(user_id, session_a)
    messages_b = get_recent_messages(user_id, session_b)

    assert messages_a[0]["content"] == "Message for A"
    assert messages_b[0]["content"] == "Message for B"


def test_only_latest_20_messages_are_kept(session_ids):
    user_id, session_id, _ = session_ids

    for i in range(25):
        add_message(user_id, session_id, "user", f"Message {i}")

    messages = get_recent_messages(
        user_id,
        session_id,
        limit=20,
    )

    assert len(messages) == 20
    assert messages[0]["content"] == "Message 5"
    assert messages[-1]["content"] == "Message 24"


def test_clear_session_deletes_only_that_session(session_ids):
    user_id, session_a, session_b = session_ids

    add_message(user_id, session_a, "user", "Delete me")
    add_message(user_id, session_b, "user", "Keep me")

    deleted = clear_session(user_id, session_a)

    assert deleted is True
    assert get_recent_messages(user_id, session_a) == []
    assert get_recent_messages(user_id, session_b) == [
        {"role": "user", "content": "Keep me"}
    ]


def test_empty_message_is_rejected(session_ids):
    user_id, session_id, _ = session_ids

    with pytest.raises(ValueError):
        add_message(user_id, session_id, "user", "   ")