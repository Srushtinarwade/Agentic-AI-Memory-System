import ast
import json
import os
import re
from functools import lru_cache
from typing import TypedDict

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END

from app.memory.short_term import add_message, get_recent_messages
from app.memory.long_term import (
    add_memory,
    list_memories,
    forget_memory,
    update_memory,
)

load_dotenv()


class AgentState(TypedDict, total=False):
    user_id: str
    session_id: str
    message: str
    history: list[dict]
    memories: list[dict]
    answer: str
    forget_request: bool


@lru_cache
def get_llm():
    """Create and cache the Gemini chat model."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing from .env")
    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        api_key=api_key,
    )


def parse_memory_output(raw: str) -> list[dict]:
    """Parse a JSON array of extracted memories, tolerating common LLM formatting errors."""
    if not isinstance(raw, str):
        raise ValueError("Gemini memory output was not text.")

    raw = re.sub(
        r"^\s*```(?:json)?\s*|\s*```\s*$", "", raw.strip(), flags=re.IGNORECASE
    ).strip()

    try:
        candidates = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", raw)
        if not match:
            raise ValueError(f"No JSON array found in Gemini output: {raw[:300]!r}")
        candidate_text = match.group(0)
        try:
            candidates = json.loads(candidate_text)
        except json.JSONDecodeError:
            repaired = re.sub(
                r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:', r'\1"\2":', candidate_text
            )
            try:
                candidates = json.loads(repaired)
            except json.JSONDecodeError:
                try:
                    candidates = ast.literal_eval(repaired)
                except (ValueError, SyntaxError) as exc:
                    raise ValueError(
                        "Could not parse Gemini memory output. "
                        f"Output begins: {raw[:300]!r}"
                    ) from exc

    if isinstance(candidates, dict):
        candidates = candidates.get("memories", candidates.get("items", []))
    if not isinstance(candidates, list):
        raise ValueError("Memory extraction must return a JSON array.")

    cleaned = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        content = item.get("content")
        if not isinstance(kind, str) or not isinstance(content, str):
            continue
        kind, content = kind.strip().lower(), content.strip()
        if kind not in {"semantic", "episodic"} or not content:
            continue
        operation = str(item.get("operation", "add")).strip().lower()
        if operation not in {"add", "update"}:
            operation = "add"
        previous_content = item.get("previous_content")
        if not isinstance(previous_content, str):
            previous_content = ""
        cleaned.append({
            "type": kind,
            "content": content,
            "operation": operation,
            "previous_content": previous_content.strip(),
        })
    return cleaned


# ---------------------------- Forget handling ----------------------------
FORGET_PATTERN = re.compile(
    r"\b(?:forget|delete|remove|erase)\b"
    r"|\b(?:don't remember|do not remember|stop remembering)\b",
    re.IGNORECASE,
)

FORGET_STOPWORDS = {
    "please", "hey", "my", "the", "a", "an", "about", "from", "in", "on",
    "of", "to", "for", "me", "you", "your", "i", "want", "would", "like",
    "all", "information", "memory", "memories", "fact", "facts", "details",
    "that", "this", "it", "related", "regarding", "personal", "interest",
    "interests", "thing", "things", "data", "saved",
    # Negation and filler words should not count as memory subjects.
    "ok", "okay", "dont", "doesnt", "didnt", "anymore", "longer",
    "no", "not", "stop", "stopped", "remembering", "any",
}


def _forget_terms(message: str) -> set[str]:
    """Extract the specific subject from a forget/delete request."""
    match = re.search(r"\b(?:forget|delete|remove|erase)\b", message, re.IGNORECASE)
    if match:
        target = message[match.end():]
    else:
        match = re.search(
            r"\b(?:don't remember|do not remember|stop remembering)\b",
            message, re.IGNORECASE,
        )
        target = message[match.end():] if match else ""

    words = re.findall(r"[a-z]+", target.lower())
    normalized = []
    for word in words:
        if word == "interested":
            word = "interest"
        elif word.endswith("s") and len(word) > 4:
            word = word[:-1]
        if word not in FORGET_STOPWORDS:
            normalized.append(word)
    return set(normalized)


def handle_forget(state: AgentState) -> dict:
    """Handle explicit user requests to remove long-term memories."""
    message = state["message"]
    if not FORGET_PATTERN.search(message):
        return {"forget_request": False}

    terms = _forget_terms(message)
    if not terms:
        return {
            "forget_request": True,
            "answer": (
                "Please specify which saved information to forget, for example: "
                "'Forget my stock market interest.' Nothing has been deleted."
            ),
        }

    memories = list_memories(state["user_id"])
    matches = []
    # If the request identifies one subject (for example, "cricket"),
    # one matching word is enough. Longer requests require multiple matches.
    required_overlap = (
        1 if len(terms) == 1
        else max(2, (len(terms) * 3 + 4) // 5)
    )
    for memory in memories:
        content_words = set(re.findall(r"[a-z]+", memory.get("content", "").lower()))
        if "interested" in content_words:
            content_words.add("interest")
        content_words = {
            word[:-1] if word.endswith("s") and len(word) > 4 else word
            for word in content_words
        }
        if len(terms & content_words) >= required_overlap:
            matches.append(memory)

    deleted = sum(
        1 for memory in matches if forget_memory(state["user_id"], memory["id"])
    )
    if deleted:
        answer = f"Done — I deleted {deleted} matching saved memory record(s)."
    else:
        answer = (
            "I couldn't find a saved memory that clearly matches that request, "
            "so nothing was deleted. Please describe the memory more specifically."
        )
    return {"forget_request": True, "answer": answer}


def route_after_forget(state: AgentState) -> str:
    """Skip retrieval and LLM generation for an explicit forget request."""
    return "save" if state.get("forget_request") else "retrieve"


# ---------------------------- Memory retrieval ---------------------------
STOPWORDS = {
    "the", "and", "for", "you", "your", "are", "what", "how", "can", "tell",
    "about", "with", "that", "this", "have", "has", "was", "were", "from",
    "remember", "please", "user", "does", "did", "not", "my", "me", "all",
    "who", "when", "where", "which", "would", "could", "should", "into",
}

PREFERENCE_QUERY_TERMS = {
    "interest", "interests", "interested", "like", "likes", "liked", "enjoy",
    "enjoys", "hobby", "hobbies", "prefer", "prefers", "preference", "passion",
    "passionate", "favorite", "favourite", "love", "loves",
}
PREFERENCE_MEMORY_TERMS = {
    "interest", "interested", "like", "likes", "enjoy", "enjoys", "hobby",
    "hobbies", "prefer", "prefers", "preference", "passion", "passionate",
    "favorite", "favourite", "love", "loves", "want", "wants",
}

# Small synonym groups improve retrieval without adding an embedding service.
SYNONYM_GROUPS = [
    {"stock", "stocks", "market", "markets", "invest", "investing", "investment", "trading", "finance"},
    {"program", "programming", "coding", "code", "developer", "development", "software"},
    {"run", "running", "runner", "jog", "jogging", "fitness", "exercise", "workout"},
    {"movie", "movies", "film", "films", "cinema"},
    {"travel", "travelling", "traveling", "trip", "trips", "journey", "vacation"},
    {"food", "cooking", "cook", "recipe", "recipes", "cuisine"},
]


def _tokens(text: str) -> set[str]:
    words = set(re.findall(r"\b[a-z]{3,}\b", text.lower()))
    normalized = set()
    for word in words:
        if word.endswith("ies") and len(word) > 4:
            word = word[:-3] + "y"
        elif word.endswith("ing") and len(word) > 5:
            word = word[:-3]
        elif word.endswith("s") and len(word) > 4:
            word = word[:-1]
        normalized.add(word)
    return normalized - STOPWORDS


def _expanded_tokens(words: set[str]) -> set[str]:
    """Add a small number of known synonyms to each token set."""
    result = set(words)
    for group in SYNONYM_GROUPS:
        if result & group:
            result.update(group)
    return result


def retrieve(state: AgentState) -> dict:
    """Retrieve recent session history and the five most relevant user memories."""
    session_id = state["session_id"]
    query = state["message"].lower()
    history = get_recent_messages(state["user_id"], session_id, limit=10)
    memories = list_memories(state["user_id"])

    query_words = _expanded_tokens(_tokens(query))
    is_preference_question = bool(PREFERENCE_QUERY_TERMS & set(re.findall(r"[a-z]+", query)))
    is_general_recall = any(phrase in query for phrase in (
        "what do you remember", "what do you know about me", "tell me about me",
        "what have i told you", "summarize what you know",
    ))

    ranked = []
    for index, memory in enumerate(memories):
        content = memory.get("content", "")
        content_words = _expanded_tokens(_tokens(content))
        overlap = len(query_words & content_words)
        score = overlap

        if is_preference_question and (PREFERENCE_MEMORY_TERMS & set(re.findall(r"[a-z]+", content.lower()))):
            score += 3
        if is_general_recall:
            score += 1
        # Favor memories with higher word overlap; index preserves ordering for ties.
        if score > 0:
            ranked.append((score, -index, memory))

    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected = [item[2] for item in ranked[:5]]

    # General recall should return recent memories even if the query has no topical keywords.
    if is_general_recall and not selected:
        selected = memories[:5]

    return {"history": history, "memories": selected}


# ------------------------------- Chat ------------------------------------
def respond(state: AgentState) -> dict:
    """Generate an answer using Gemini and relevant stored memories."""
    context = "\n".join(
        f"- [{m.get('type', 'semantic')}] {m['content']}"
        for m in state.get("memories", [])
    ) or "No relevant stored memories."

    messages = [SystemMessage(content=(
        "You are a helpful AI assistant with persistent memory. Use the relevant "
        "stored memories when useful, but do not invent facts about the user. "
        "If no memory is relevant, answer normally. If asked what you remember, "
        "summarize only the provided memories.\nRelevant memories:\n" + context
    ))]

    for item in state.get("history", []):
        if item.get("role") == "user":
            messages.append(HumanMessage(content=item["content"]))
        else:
            messages.append(AIMessage(content=item["content"]))
    messages.append(HumanMessage(content=state["message"]))

    result = get_llm().invoke(messages)
    content = result.content
    if isinstance(content, str):
        answer = content
    elif isinstance(content, list):
        answer = "\n".join(
            block["text"] for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        )
    else:
        answer = str(content)
    return {"answer": answer.strip()}


def save_conversation(state: AgentState) -> dict:
    """Save the user's message and assistant's response to session history."""
    add_message(state["user_id"], state["session_id"], "user", state["message"])
    add_message(state["user_id"], state["session_id"], "assistant", state["answer"])
    return {}


# ------------------------- Long-term extraction -------------------------
UPDATE_CUES = re.compile(
    r"\b(actually|instead|correction|i changed|change that|no longer|not anymore|"
    r"used to|now i|from now on|i prefer|i don't like|i do not like|"
    r"i no longer|my preference is|update my|that's wrong|that is wrong)\b",
    re.IGNORECASE,
)


def extract_memories(state: AgentState) -> dict:
    """Extract facts; update a prior memory only when Gemini identifies it explicitly."""
    if state.get("forget_request"):
        return {}

    user_id = state["user_id"]
    existing_memories = list_memories(user_id)
    existing_for_prompt = [
        {"type": m.get("type", "semantic"), "content": m.get("content", "")}
        for m in existing_memories[:50]
    ]

    prompt = f"""
You maintain a user's long-term memory. Analyze only the latest user message.
Return ONLY a valid JSON array, no markdown and no explanation.
Each item must include:
- "type": "semantic" or "episodic"
- "content": concise memory text written in third person (e.g. "The user prefers ...")
- "operation": "add" or "update"
- "previous_content": exact full content of an existing memory to replace if operation is update; otherwise an empty string

Rules:
1. Extract only useful information explicitly stated by the user. Do not infer from a question.
2. Semantic = stable facts/preferences; episodic = important events.
3. For a clear correction/change of a previously saved fact or preference, return operation "update" and copy the old memory's content exactly into previous_content. Return the corrected replacement as content.
4. Only use operation "update" when one existing memory is clearly being corrected or replaced. Never update unrelated memories.
5. If the message does not change a saved memory, operation is "add".
6. Do not return exact duplicates. Return [] if nothing useful is worth remembering.
7. If a user says they no longer like/prefer something, update the old preference when identifiable; do not save a contradictory preference as a separate new memory.

Existing memories:
{json.dumps(existing_for_prompt, ensure_ascii=False)}

Latest user message:
{json.dumps(state['message'], ensure_ascii=False)}
"""

    try:
        result = get_llm().invoke([HumanMessage(content=prompt)])
        raw = result.content
        if isinstance(raw, list):
            raw = "\n".join(
                block["text"] for block in raw
                if isinstance(block, dict) and isinstance(block.get("text"), str)
            )
        if not isinstance(raw, str):
            raise ValueError(f"Unexpected Gemini response type: {type(raw).__name__}")

        candidates = parse_memory_output(raw)
        existing = list_memories(user_id)
        saved_count = 0
        updated_count = 0

        for item in candidates:
            kind = item["type"]
            content = item["content"].strip()
            operation = item.get("operation", "add")
            previous_content = item.get("previous_content", "")
            if not content:
                continue

            # Perform an update only if Gemini names an exact existing record.
            if operation == "update" and previous_content:
                target = next((
                    m for m in existing
                    if m.get("content", "").strip().casefold() == previous_content.casefold()
                ), None)
                if target:
                    update_memory(user_id, target["id"], content=content)
                    updated_count += 1
                    existing = list_memories(user_id)
                    continue

            normalized = content.casefold()
            if any(m.get("content", "").strip().casefold() == normalized for m in existing):
                continue

            # Conservative fallback: never guess which old memory to overwrite.
            if operation == "update":
                print("[MEMORY UPDATE] Update target was not an exact existing memory; skipped.")
                continue

            add_memory(user_id, content, kind)
            saved_count += 1
            existing = list_memories(user_id)

        print(
            f"[MEMORY EXTRACTION] Candidates: {len(candidates)}, "
            f"saved: {saved_count}, updated: {updated_count}"
        )
    except Exception as exc:
        # An extraction error must not break an otherwise successful chat request.
        print(f"[MEMORY EXTRACTION ERROR] {type(exc).__name__}: {exc}")
        if "raw" in locals() and isinstance(raw, str):
            print(f"[MEMORY EXTRACTION RAW] {raw[:500]!r}")
    return {}


# ----------------------------- LangGraph --------------------------------
builder = StateGraph(AgentState)
builder.add_node("handle_forget", handle_forget)
builder.add_node("retrieve", retrieve)
builder.add_node("respond", respond)
builder.add_node("save", save_conversation)
builder.add_node("extract", extract_memories)
builder.add_edge(START, "handle_forget")
builder.add_conditional_edges(
    "handle_forget", route_after_forget, {"save": "save", "retrieve": "retrieve"}
)
builder.add_edge("retrieve", "respond")
builder.add_edge("respond", "save")
builder.add_edge("save", "extract")
builder.add_edge("extract", END)

agent = builder.compile()
