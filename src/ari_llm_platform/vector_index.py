import json
import sqlite3
from pathlib import Path

import sqlite_vec


class VectorIndex:
    DEFAULT_DB_PATH = Path("src") / "data" / "db" / "ari.db"

    def __init__(self, embedding_client=None):
        self.db_path = str(self.DEFAULT_DB_PATH)
        self.backend = None
        self.initialized = False
        self.connection = None
        self.embedding_client = embedding_client

    def initialize(self, db_path: str | None = None, backend: str = "sqlite-vec") -> None:
        self.db_path = db_path or str(self.DEFAULT_DB_PATH)
        self.backend = backend
        self.initialized = True

    def is_ready(self) -> bool:
        return self.initialized

    def load_backend(self) -> str:
        if self.backend != "sqlite-vec":
            raise ValueError(f"Unsupported backend: {self.backend}")

        return "sqlite-vec-loaded"

    def connect(self) -> sqlite3.Connection:
        if not self.initialized:
            self.initialize()

        database_path = Path(self.db_path)
        database_path.parent.mkdir(parents=True, exist_ok=True)

        connection = sqlite3.connect(
            database_path,
            check_same_thread=False,
        )
        try:
            connection.enable_load_extension(True)
            sqlite_vec.load(connection)
        except Exception:
            connection.close()
            raise
        finally:
            connection.enable_load_extension(False)

        self.connection = connection
        return connection

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def embed_text(self, text: str):
        return {
            "text": text,
            "embedding": [],
            "dim": 0,
            "model": None,
            "backend": self.backend,
        }

    def upsert_chunk_embedding(self, chunk_id, embedding, dim, model, backend):
        return {
            "chunk_id": chunk_id,
            "stored": True,
            "dim": dim,
            "model": model,
            "backend": backend,
        }

    def semantic_search(
        self,
        query_text: str,
        k: int = 5,
        filters: dict | None = None,
    ):
        if not query_text.strip():
            raise ValueError("Search query must not be empty.")

        if k <= 0:
            raise ValueError("Search result count must be greater than zero.")

        if self.embedding_client is None:
            raise RuntimeError(
                "VectorIndex requires an embedding_client for semantic search."
            )

        connection = self.connection or self.connect()
        query_embedding = self.embedding_client.embed(query_text)

        vector_rows = connection.execute(
            """
            SELECT rowid, distance
            FROM chunk_embedding_vectors
            WHERE embedding MATCH ?
            AND k = ?
            """,
            (
                sqlite_vec.serialize_float32(query_embedding),
                k * 4,
            ),
        ).fetchall()

        results = []

        for chunk_id, distance in vector_rows:
            row = connection.execute(
                """
                SELECT
                    documents.id,
                    chunks.id,
                    chunks.text,
                    documents.source_type,
                    documents.source_ref,
                    documents.title,
                    documents.metadata
                FROM chunks
                JOIN documents ON documents.id = chunks.document_id
                WHERE chunks.id = ?
                """,
                (chunk_id,),
            ).fetchone()

            if row is None:
                continue

            (
                document_id,
                stored_chunk_id,
                text,
                source_type,
                source_ref,
                title,
                metadata_json,
            ) = row

            metadata = json.loads(metadata_json)

            if (
                source_type == "memory"
                and (not filters or filters.get("source_type") != "memory")
            ):
                continue

            filter_values = {
                "document_id": document_id,
                "source_type": source_type,
                **metadata,
            }

            if filters and any(
                filter_values.get(name) != value
                for name, value in filters.items()
            ):
                continue

            results.append(
                {
                    "document_id": document_id,
                    "chunk_id": stored_chunk_id,
                    "text": text,
                    "source_type": source_type,
                    "source_ref": source_ref,
                    "title": title,
                    "distance": distance,
                }
            )

            if len(results) == k:
                break

        return {
            "query_text": query_text,
            "k": k,
            "filters": filters,
            "results": results,
        }

    def reindex_document(self, document_id):
        return {
            "document_id": document_id,
            "reindexed": False,
        }

    def vacuum(self):
        return {
            "vacuumed": False,
            "backend": self.backend,
        }