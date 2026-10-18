"""No external services needed to test deterministic tenant-scoped thread IDs."""
import importlib
import sys

import pytest
from dependency_stubs import install_agent_stubs


@pytest.fixture
def chat_module(monkeypatch):
    install_agent_stubs(monkeypatch)
    monkeypatch.delitem(sys.modules, "app.agent", raising=False)
    monkeypatch.delitem(sys.modules, "app.router.chat", raising=False)
    module = importlib.import_module("app.router.chat")
    yield module
    sys.modules.pop("app.router.chat", None)
    sys.modules.pop("app.agent", None)


def test_thread_keys_stable_isolated_and_short(chat_module):
    a = chat_module.thread_id_for("alice", "same")
    assert a == chat_module.thread_id_for("alice", "same")
    assert a != chat_module.thread_id_for("bob", "same")
    assert a != chat_module.thread_id_for("alice", "different")
    assert a != chat_module.thread_id_for("alice:a", "b")
    assert len(a) < 255
    assert "alice" not in a
