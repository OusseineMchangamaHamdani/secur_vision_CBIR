CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE users (
    user_id SERIAL PRIMARY KEY,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    email VARCHAR(255) UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE user_embeddings (
    embedding_id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    image_path TEXT NOT NULL,
    embedding VECTOR(512),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

CREATE INDEX idx_user_embeddings
ON user_embeddings
USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS search_history (
    id SERIAL PRIMARY KEY,
    timestamp VARCHAR(50),
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    department VARCHAR(100),
    position VARCHAR(100),
    similarity_percent FLOAT,
    status VARCHAR(50),
    image_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);