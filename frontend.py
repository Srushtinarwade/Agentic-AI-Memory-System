"""Streamlit demo: chat, active memories, and the user's evolving graph."""
from __future__ import annotations

import json
import os
import uuid

from dotenv import load_dotenv
import requests
import streamlit as st

load_dotenv()
API = os.getenv("MEMORY_API_URL", "http://127.0.0.1:8000").rstrip("/")
API_TOKEN = os.getenv("AUTH_TOKEN", "")
HEADERS = {"X-API-Key": API_TOKEN} if API_TOKEN else {}

st.set_page_config(page_title="Graph Memory AI", page_icon="🧠", layout="wide")
st.title("🧠 Agentic AI — Graph Memory")
st.caption("Gemini chat · Neo4j long-term memory · local CPU embeddings")

if "user" not in st.session_state:
    st.session_state.user = "user_101"
if "session" not in st.session_state:
    st.session_state.session = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("Settings")
    st.session_state.user = st.text_input("User ID", st.session_state.user)
    if st.button("＋ New conversation"):
        st.session_state.session = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()
    st.link_button("API documentation", f"{API}/docs")
    st.link_button("Neo4j Browser", "http://127.0.0.1:7474")
    st.caption("Demo only. User IDs are not authentication; keep the API on localhost.")

chat, memory = st.columns([3, 2], gap="large")
with chat:
    st.subheader("💬 Chat")
    for item in st.session_state.messages:
        with st.chat_message(item["role"]):
            st.write(item["content"])

    prompt = st.chat_input("Tell the AI about a project, preference, or goal...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        try:
            response = requests.post(
                f"{API}/chat",
                headers=HEADERS,
                json={
                    "user_id": st.session_state.user,
                    "session_id": st.session_state.session,
                    "message": prompt,
                },
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
            st.session_state.messages.append({"role": "assistant", "content": data.get("answer", "")})
            st.rerun()
        except requests.RequestException as exc:
            st.error(f"Chat failed: {exc}")

with memory:
    st.subheader("🗃️ Active long-term memories")
    include_history = st.checkbox("Include superseded history", value=False)
    try:
        result = requests.get(
            f"{API}/users/{st.session_state.user}/memories",
            headers=HEADERS, params={"limit": 100, "include_history": include_history}, timeout=20,
        )
        result.raise_for_status()
        facts = result.json().get("memories", [])
        st.metric("Saved facts", len(facts))
        for fact in facts:
            st.markdown(f"**{fact['statement']}**")
            st.caption(
                f"{fact['predicate']} · {fact['status']} · seen {fact['seen_count']} time(s)"
            )
            if st.button("Forget this fact", key=f"delete-{fact['fact_id']}"):
                deleted = requests.delete(
                    f"{API}/users/{st.session_state.user}/memories/{fact['fact_id']}",
                    headers=HEADERS, timeout=20,
                )
                deleted.raise_for_status()
                st.rerun()
            st.divider()
    except requests.RequestException as exc:
        st.warning(f"Memory API unavailable: {exc}")

st.divider()
st.subheader("🕸️ Your evolving memory graph")
st.caption("Blue root = user. Dotted edges = membership. Solid edges = learned facts.")
try:
    response = requests.get(
        f"{API}/users/{st.session_state.user}/graph",
        headers=HEADERS, params={"include_history": include_history, "limit": 150}, timeout=20,
    )
    response.raise_for_status()
    graph = response.json()
    if graph.get("nodes"):
        # DOT quotes/escapes all user-derived labels rather than interpolating raw text.
        lines = ['digraph Memory {', '  graph [layout=neato, overlap=false];',
                 '  node [shape=ellipse, style=filled, fillcolor="#eef2ff"];']
        for node in graph["nodes"]:
            color = "#93c5fd" if node["type"] == "User" else "#dcfce7"
            lines.append(f'  {json.dumps(node["id"])} [label={json.dumps(node["name"])}, fillcolor="{color}"];')
        for edge in graph["edges"]:
            dashed = 'dotted' if edge["predicate"] == "HAS_ENTITY" else 'solid'
            label = '' if dashed == 'dotted' else edge["predicate"]
            lines.append(
                f'  {json.dumps(edge["source"])} -> {json.dumps(edge["target"])} '
                f'[label={json.dumps(label)}, style="{dashed}"];'
            )
        lines.append('}')
        st.graphviz_chart("\n".join(lines), use_container_width=True)
    else:
        st.info("Your graph is empty. Tell the assistant about your work or preferences to create its first nodes.")
except requests.RequestException as exc:
    st.warning(f"Graph API unavailable: {exc}")
