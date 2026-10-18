"""Offline-friendly setup diagnostics (does not invoke Gemini)."""
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.graph_memory.config import Settings


async def main() -> None:
    settings = Settings(_env_file=ROOT / ".env")
    print("PostgreSQL + LangGraph memory setup check")
    print("-" * 42)
    failures = 0
    print(f"[OK] Python {sys.version.split()[0]}")
    if not (ROOT / ".env").is_file():
        print("[X] .env not found. Copy .env.example to .env first")
        failures += 1
    else:
        print("[OK] .env exists")
    if settings.gemini_api_key.get_secret_value().strip():
        print(f"[OK] Gemini configured (model: {settings.gemini_model}); key NOT displayed")
    else:
        print("[X] GEMINI_API_KEY is missing")
        failures += 1
    dsn = settings.postgres_dsn.get_secret_value()
    if dsn:
        try:
            from psycopg import AsyncConnection

            async with await AsyncConnection.connect(dsn, connect_timeout=5) as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute("SELECT EXISTS (SELECT 1 FROM pg_available_extensions WHERE name = 'vector')")
                    if (await cursor.fetchone())[0]:
                        print("[OK] PostgreSQL reachable with available pgvector extension")
                    else:
                        print("[X] PostgreSQL lacks the pgvector extension; use the Compose image")
                        failures += 1
        except Exception as exc:
            print(f"[X] PostgreSQL unavailable ({type(exc).__name__}); run docker compose up -d")
            failures += 1
    else:
        print("[X] POSTGRES_DSN is missing")
        failures += 1

    path = ROOT / settings.embedding_model_path
    if path.is_dir() and (path / "modules.json").is_file():
        print(f"[OK] Local MiniLM weights found: {path}")
    else:
        print("[X] Model not downloaded. Run: python scripts/download_model.py")
        failures += 1

    redis_url = settings.redis_url.get_secret_value()
    if redis_url:
        try:
            from redis.asyncio import Redis

            client = Redis.from_url(redis_url, socket_timeout=2)
            try:
                await client.ping()
                print("[OK] Optional Redis cache reachable")
            finally:
                await client.aclose()
        except Exception:
            print("[!] Optional Redis unavailable; app will run without cache")
    else:
        print("[OK] Redis disabled (optional)")

    print("-" * 42)
    if failures:
        print(f"{failures} item(s) need attention. See README.md setup instructions.")
        raise SystemExit(1)
    print("All required setup checks passed.")


if __name__ == "__main__":
    asyncio.run(main())
