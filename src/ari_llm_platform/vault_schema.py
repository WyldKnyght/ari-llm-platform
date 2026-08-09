import sqlite3

PHASE3_TABLES = (
    "chunk_embedding_vectors",
    "chunk_embeddings",
    "document_tags",
    "tags",
    "chunks",
    "documents",
)


def migrate_phase3(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA foreign_keys = ON")

    statements = [
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY,
            source_type TEXT NOT NULL,
            source_ref TEXT,
            title TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT NOT NULL DEFAULT '{}'
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            document_id INTEGER NOT NULL,
            position INTEGER NOT NULL,
            text TEXT NOT NULL,
            token_count INTEGER,
            FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
            UNIQUE (document_id, position)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS tags (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            description TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS document_tags (
            document_id INTEGER NOT NULL,
            tag_id INTEGER NOT NULL,
            PRIMARY KEY (document_id, tag_id),
            FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
            FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS chunk_embeddings (
            id INTEGER PRIMARY KEY,
            chunk_id INTEGER NOT NULL UNIQUE,
            embedding BLOB NOT NULL,
            dim INTEGER NOT NULL,
            model TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
        )
        """,
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS chunk_embedding_vectors
        USING vec0(
            embedding float[384]
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_documents_source_type
        ON documents(source_type)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_chunks_document_id
        ON chunks(document_id)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_model
        ON chunk_embeddings(model)
        """,
    ]

    try:
        for statement in statements:
            connection.execute(statement)
        connection.commit()
    except Exception:
        connection.rollback()
        raise


def rollback_phase3(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA foreign_keys = OFF")

    try:
        for table_name in PHASE3_TABLES:
            connection.execute(f"DROP TABLE IF EXISTS {table_name}")
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.execute("PRAGMA foreign_keys = ON")