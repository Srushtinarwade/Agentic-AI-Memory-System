
import uuid
import requests
import streamlit as st

API = "http://127.0.0.1:8000"

st.set_page_config(page_title="Memory AI", page_icon="🧠", layout="wide")
st.title("🧠 Memory AI")
st.caption("An AI assistant that remembers you.")

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

chat, memory = st.columns([2, 1])

with chat:
    st.subheader("💬 Chat")
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    prompt = st.chat_input("Message your AI...")

    if prompt:
        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )
        try:
            r = requests.post(
                f"{API}/chat",
                json={
                    "user_id": st.session_state.user,
                    "session_id": st.session_state.session,
                    "message": prompt,
                },
                timeout=120,
            )
            r.raise_for_status()
            answer = r.json().get("answer", "No answer returned.")
            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )
            st.rerun()
        except requests.RequestException as e:
            st.error(f"Chat request failed: {e}")

with memory:
    st.subheader("🗃️ Saved Memories")
    try:
        r = requests.get(
            f"{API}/memories/",
            params={"user_id": st.session_state.user},
            timeout=15,
        )
        r.raise_for_status()
        memories = r.json()
        st.metric("Total memories", len(memories))

        for m in memories:
            st.caption(m.get("type", m.get("memory_type", "memory")).title())
            st.write(m.get("content", ""))
            st.divider()
    except requests.RequestException as e:
        st.error(f"Cannot load memories: {e}")
