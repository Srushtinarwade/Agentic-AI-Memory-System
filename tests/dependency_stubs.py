"""Minimal import-only stubs for offline unit tests, never used by application."""
import importlib.util
import sys
import types
from typing import Generic, TypeVar


def missing(name):
    try:
        return importlib.util.find_spec(name) is None
    except (ModuleNotFoundError, ValueError, ImportError):
        return True


def install_agent_stubs(monkeypatch):
    if missing("langchain_core.messages"):
        core = types.ModuleType("langchain_core")
        core.__path__ = []
        msg = types.ModuleType("langchain_core.messages")

        class Message:
            def __init__(self, content):
                self.content = content

        class HumanMessage(Message):
            pass

        class AIMessage(Message):
            pass

        msg.HumanMessage = HumanMessage
        msg.AIMessage = AIMessage
        msg.AnyMessage = Message
        monkeypatch.setitem(sys.modules, "langchain_core", core)
        monkeypatch.setitem(sys.modules, "langchain_core.messages", msg)

    if missing("langgraph.graph"):
        pkg = types.ModuleType("langgraph")
        pkg.__path__ = []
        graph = types.ModuleType("langgraph.graph")
        message = types.ModuleType("langgraph.graph.message")
        runtime = types.ModuleType("langgraph.runtime")

        class StateGraph:
            def __init__(self, schema, context_schema=None):
                self.nodes = {}

            def add_node(self, name, callback):
                self.nodes[name] = callback

            def add_edge(self, *args):
                pass

            def add_conditional_edges(self, *args):
                pass

            def compile(self, checkpointer=None):
                return self

        class Runtime(Generic[TypeVar("ContextT")]):
            pass

        graph.StateGraph = StateGraph
        graph.START = "start"
        graph.END = "end"
        message.add_messages = lambda old, new: (old or []) + new
        runtime.Runtime = Runtime
        monkeypatch.setitem(sys.modules, "langgraph", pkg)
        monkeypatch.setitem(sys.modules, "langgraph.graph", graph)
        monkeypatch.setitem(sys.modules, "langgraph.graph.message", message)
        monkeypatch.setitem(sys.modules, "langgraph.runtime", runtime)


def install_db_stubs(monkeypatch):
    if missing("psycopg.rows"):
        pg = types.ModuleType("psycopg")
        pg.__path__ = []
        rows = types.ModuleType("psycopg.rows")
        rows.dict_row = object()
        monkeypatch.setitem(sys.modules, "psycopg", pg)
        monkeypatch.setitem(sys.modules, "psycopg.rows", rows)
    if missing("psycopg_pool"):
        pool = types.ModuleType("psycopg_pool")
        pool.AsyncConnectionPool = object
        monkeypatch.setitem(sys.modules, "psycopg_pool", pool)
    if missing("langgraph.checkpoint.postgres.aio"):
        checkpoint = types.ModuleType("langgraph.checkpoint")
        checkpoint.__path__ = []
        postgres = types.ModuleType("langgraph.checkpoint.postgres")
        postgres.__path__ = []
        aio = types.ModuleType("langgraph.checkpoint.postgres.aio")
        aio.AsyncPostgresSaver = object
        monkeypatch.setitem(sys.modules, "langgraph.checkpoint", checkpoint)
        monkeypatch.setitem(sys.modules, "langgraph.checkpoint.postgres", postgres)
        monkeypatch.setitem(sys.modules, "langgraph.checkpoint.postgres.aio", aio)
    if missing("google.genai"):
        google = types.ModuleType("google")
        google.__path__ = []
        genai = types.ModuleType("google.genai")
        genai.types = types.ModuleType("google.genai.types")
        google.genai = genai
        monkeypatch.setitem(sys.modules, "google", google)
        monkeypatch.setitem(sys.modules, "google.genai", genai)
