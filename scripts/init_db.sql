CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS documents (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    filename    TEXT NOT NULL,
    file_type   TEXT NOT NULL,
    source_url  TEXT,
    status      TEXT NOT NULL DEFAULT 'pending',  -- pending|processing|ready|failed
    chunk_count INT NOT NULL DEFAULT 0,
    error_msg   TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chunks (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    doc_id      UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    raw_text    TEXT NOT NULL,
    context     TEXT NOT NULL DEFAULT '',   -- LLM-generated contextual prefix
    full_text   TEXT NOT NULL,              -- context || raw_text — what gets embedded
    page_num    INT,
    char_start  INT,
    char_end    INT,
    qdrant_id   UUID,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(doc_id, chunk_index)
);

CREATE TABLE IF NOT EXISTS eval_runs (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    git_sha             TEXT,
    faithfulness        FLOAT,
    context_precision   FLOAT,
    context_recall      FLOAT,
    answer_relevancy    FLOAT,
    passed_ci           BOOLEAN NOT NULL DEFAULT FALSE,
    question_count      INT,
    run_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS golden_questions (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    question        TEXT NOT NULL,
    ground_truth    TEXT NOT NULL,
    relevant_chunks TEXT[],
    category        TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_qdrant_id ON chunks(qdrant_id);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
