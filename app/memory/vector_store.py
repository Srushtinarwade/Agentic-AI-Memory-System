
import os
import uuid

import psycopg
from dotenv import load_dotenv
from google import genai
from pgvector.psycopg import register_vector
from pgvector import Vector
from psycopg.types.json import Jsonb

load_dotenv()

_client = None


def embed(text: str) -> list[float]:
    global _client

    if _client is None:
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    result = _client.models.embed_content(
        model="gemini-embedding-2",
        contents=text,
        config={"output_dimensionality": 768},
    )
    return list(result.embeddings[0].values)


def connect():
    conn = psycopg.connect(os.environ["DATABASE_URL"])
    register_vector(conn)
    return conn


def init_db():
    with connect() as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS long_term_memories (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                embedding vector(768) NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_events (
                id BIGSERIAL PRIMARY KEY,
                user_id TEXT NOT NULL,
                memory_id TEXT,
                event_type TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)


def save_memory(user_id, memory_id, content, memory_type, metadata=None):
    vector = embed(content)

    with connect() as conn:
        conn.execute("""
            INSERT INTO long_term_memories
                (id, user_id, content, memory_type, metadata, embedding)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                content = EXCLUDED.content,
                memory_type = EXCLUDED.memory_type,
                metadata = EXCLUDED.metadata,
                embedding = EXCLUDED.embedding
            WHERE long_term_memories.user_id = EXCLUDED.user_id
        """, (
            memory_id,
            user_id,
            content,
            memory_type,
            Jsonb(metadata or {}),
            Vector(vector),
        ))

        conn.execute("""
            INSERT INTO memory_events (user_id, memory_id, event_type)
            VALUES (%s, %s, 'save')
        """, (user_id, memory_id))


def search_memories(user_id, query, limit=5):
    vector = Vector(embed(query))

    with connect() as conn:
        rows = conn.execute("""
            SELECT id, content, memory_type, metadata,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM long_term_memories
            WHERE user_id = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (vector, user_id, vector, limit)).fetchall()

        for row in rows:
            conn.execute("""
                INSERT INTO memory_events (user_id, memory_id, event_type)
                VALUES (%s, %s, 'retrieve')
            """, (user_id, row[0]))

    return [
        {
            "id": row[0],
            "content": row[1],
            "type": row[2],
            "metadata": row[3],
            "similarity": float(row[4]),
        }
        for row in rows
    ]


def list_memories(user_id, memory_type=None):
    with connect() as conn:
        if memory_type:
            rows = conn.execute("""
                SELECT id, user_id, content, memory_type, metadata, created_at
                FROM long_term_memories
                WHERE user_id = %s AND memory_type = %s
                ORDER BY created_at DESC
            """, (user_id, memory_type)).fetchall()
        else:
            rows = conn.execute("""
                SELECT id, user_id, content, memory_type, metadata, created_at
                FROM long_term_memories
                WHERE user_id = %s
                ORDER BY created_at DESC
            """, (user_id,)).fetchall()

    return [
        {
            "id": row[0],
            "user_id": row[1],
            "content": row[2],
            "type": row[3],
            "metadata": row[4],
            "created_at": row[5].isoformat(),
        }
        for row in rows
    ]
def get_memory(user_id, memory_id):
    with connect() as conn:
        row = conn.execute("""
            SELECT id, user_id, content, memory_type, metadata, created_at
            FROM long_term_memories
            WHERE user_id = %s AND id = %s
        """, (user_id, memory_id)).fetchone()

    if not row:
        return None

    return {
        "id": row[0],
        "user_id": row[1],
        "content": row[2],
        "type": row[3],
        "metadata": row[4],
        "created_at": row[5].isoformat(),
    }

def add_memory(user_id, content, memory_type, metadata=None):
    memory_id = str(uuid.uuid4())

    save_memory(
        user_id, memory_id, content, memory_type, metadata
    )

    return {
        "id": memory_id,
        "user_id": user_id,
        "content": content,
        "type": memory_type,
        "metadata": metadata or {},
    }


def update_memory(user_id, memory_id, content=None, metadata=None):
    with connect() as conn:
        row = conn.execute("""
            SELECT content, memory_type, metadata
            FROM long_term_memories
            WHERE user_id = %s AND id = %s
        """, (user_id, memory_id)).fetchone()

        if not row:
            return None

        new_content = content if content is not None else row[0]
        new_metadata = metadata if metadata is not None else row[2]

        # Regenerate embedding only if the content changed.
        vector = Vector(embed(new_content)) if content is not None else None

        if vector is not None:
            conn.execute("""
                UPDATE long_term_memories
                SET content = %s, metadata = %s, embedding = %s
                WHERE user_id = %s AND id = %s
            """, (
                new_content, Jsonb(new_metadata), vector,
                user_id, memory_id,
            ))
        else:
            conn.execute("""
                UPDATE long_term_memories
                SET metadata = %s
                WHERE user_id = %s AND id = %s
            """, (Jsonb(new_metadata), user_id, memory_id))

        conn.execute("""
            INSERT INTO memory_events (user_id, memory_id, event_type)
            VALUES (%s, %s, 'update')
        """, (user_id, memory_id))

    return {
        "id": memory_id,
        "user_id": user_id,
        "content": new_content,
        "type": row[1],
        "metadata": new_metadata,
    }


def delete_memory(user_id, memory_id):
    with connect() as conn:
        result = conn.execute("""
            DELETE FROM long_term_memories
            WHERE user_id = %s AND id = %s
            RETURNING id
        """, (user_id, memory_id)).fetchone()

        if result:
            conn.execute("""
                INSERT INTO memory_events (user_id, memory_id, event_type)
                VALUES (%s, %s, 'delete')
            """, (user_id, memory_id))

    return result is not None


def forget_memory(user_id, memory_id):
    return delete_memory(user_id, memory_id)
