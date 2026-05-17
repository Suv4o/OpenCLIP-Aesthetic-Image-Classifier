-- Runs automatically the first time the container starts with an empty data volume.
-- To re-run, stop the stack, delete the ./pgdata directory, and `docker compose up -d` again.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS image_embeddings (
    id          BIGSERIAL PRIMARY KEY,
    filename    TEXT          NOT NULL UNIQUE,
    path        TEXT          NOT NULL,
    embedding   VECTOR(1024)  NOT NULL,
    category    TEXT,
    confidence  REAL,
    created_at  TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

-- HNSW with inner-product distance.
-- We L2-normalise every embedding before insert, so cosine similarity equals inner product
-- up to a constant. Inner product is ~30% faster in pgvector than cosine.
CREATE INDEX IF NOT EXISTS image_embeddings_hnsw_ip
    ON image_embeddings
    USING hnsw (embedding vector_ip_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS image_embeddings_category_idx
    ON image_embeddings (category);
