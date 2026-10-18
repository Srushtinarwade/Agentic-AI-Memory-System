-- Application memory tables. LangGraph's AsyncPostgresSaver owns its separate
-- checkpoints, checkpoint_blobs, and checkpoint_writes tables.
-- Requires PostgreSQL with the pgvector extension installed.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS memory_users (
    user_id text PRIMARY KEY,
    name text NOT NULL DEFAULT 'User',
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS memory_entities (
    user_id text NOT NULL REFERENCES memory_users(user_id),
    entity_id text NOT NULL,
    name text NOT NULL,
    normalized_name text NOT NULL,
    type text NOT NULL,
    embedding vector(384) NOT NULL,
    aliases text[] NOT NULL DEFAULT '{}',
    normalized_aliases text[] NOT NULL DEFAULT '{}',
    description text NOT NULL DEFAULT '',
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, entity_id),
    UNIQUE (user_id, type, normalized_name)
);

CREATE INDEX IF NOT EXISTS memory_entity_lookup_idx
    ON memory_entities (user_id, type, normalized_name);

CREATE TABLE IF NOT EXISTS memory_facts (
    user_id text NOT NULL REFERENCES memory_users(user_id),
    fact_id text NOT NULL,
    subject_id text NOT NULL,
    object_id text NOT NULL,
    predicate text NOT NULL,
    statement text NOT NULL,
    embedding vector(384) NOT NULL,
    confidence double precision NOT NULL DEFAULT 0.95,
    seen_count integer NOT NULL DEFAULT 1 CHECK (seen_count > 0),
    status text NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'superseded', 'deleted')),
    source_message_id text NOT NULL,
    last_source_message_id text NOT NULL,
    supersedes_fact_id text,
    timeframe text,
    context text,
    valid_to timestamptz,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, fact_id),
    FOREIGN KEY (user_id, object_id)
        REFERENCES memory_entities (user_id, entity_id)
);

CREATE INDEX IF NOT EXISTS memory_fact_active_idx
    ON memory_facts (user_id, status, updated_at DESC);
CREATE INDEX IF NOT EXISTS memory_fact_subject_idx
    ON memory_facts (user_id, subject_id, status);
CREATE INDEX IF NOT EXISTS memory_fact_object_idx
    ON memory_facts (user_id, object_id, status);
