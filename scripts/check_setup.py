"""Friendly offline-first configuration diagnostics; no Gemini quota used."""
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.graph_memory.config import Settings


async def main() -> None:
    settings = Settings(_env_file=ROOT / '.env')
    print('Graph Memory laptop setup check')
    print('-' * 36)
    failures = 0
    print(f'[OK] Python {sys.version.split()[0]}')
    if not (ROOT / '.env').is_file():
        print('[X] .env not found. Copy .env.example to .env first')
        failures += 1
    else:
        print('[OK] .env exists')
    if settings.gemini_api_key.get_secret_value().strip():
        print(f'[OK] Gemini key configured (model: {settings.gemini_model}); key NOT displayed')
    else:
        print('[X] GEMINI_API_KEY is missing')
        failures += 1
    if settings.neo4j_password.get_secret_value().strip():
        print('[OK] Neo4j password configured (not displayed)')
    else:
        print('[X] NEO4J_PASSWORD is missing')
        failures += 1
    model_path = ROOT / settings.embedding_model_path
    if model_path.is_dir() and (model_path / 'modules.json').is_file():
        print(f'[OK] Local MiniLM weights found: {model_path}')
    else:
        print('[X] Model not downloaded. Run: python scripts/download_model.py')
        failures += 1
    try:
        from neo4j import AsyncGraphDatabase
        driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password.get_secret_value()),
            connection_timeout=5,
        )
        try:
            await driver.verify_connectivity()
            print(f'[OK] Neo4j reachable: {settings.neo4j_uri}')
        finally:
            await driver.close()
    except Exception as exc:
        print(f'[X] Neo4j unreachable ({type(exc).__name__}). Run docker compose up -d')
        failures += 1
    print('-' * 36)
    if failures:
        print(f'{failures} item(s) need attention. See README.md sections 1 and 12.')
        raise SystemExit(1)
    print('All setup checks passed. You can start the API now.')


if __name__ == '__main__':
    asyncio.run(main())
