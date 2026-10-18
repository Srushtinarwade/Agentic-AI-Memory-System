"""Optional smoke test using only Python's standard library. Requires a running API."""
import json
import os
import urllib.request
import urllib.error

ROOT = os.getenv("MEMORY_API_URL", "http://127.0.0.1:8000")
USER = "demo_user_123"


def call(path: str, payload: dict):
    raw = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    token = os.getenv("MEMORY_API_TOKEN")
    if token:
        headers["X-API-Key"] = token
    request = urllib.request.Request(ROOT + path, data=raw, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def main():
    print("1) Remember TaskFlow with Flask...")
    print(json.dumps(call(f"/users/{USER}/memories", {
        "message": "I'm building TaskFlow using Flask, and I prefer dark mode."
    }), indent=2))
    print("2) Replace Flask with FastAPI...")
    print(json.dumps(call(f"/users/{USER}/memories", {
        "message": "I switched TaskFlow from Flask to FastAPI."
    }), indent=2))
    print("3) Retrieve current frameworks...")
    print(json.dumps(call(f"/users/{USER}/memories/search", {
        "query": "What backend framework does TaskFlow currently use?", "top_k": 5
    }), indent=2))
    print("4) Retrieve past technology...")
    print(json.dumps(call(f"/users/{USER}/memories/search", {
        "query": "What did TaskFlow use before FastAPI?", "include_history": True
    }), indent=2))
    print(f"Open {ROOT}/docs and Neo4j Browser to explore the graph.")


if __name__ == "__main__":
    try:
        main()
    except urllib.error.URLError as exc:
        raise SystemExit(f"API is unavailable: {exc}") from exc
