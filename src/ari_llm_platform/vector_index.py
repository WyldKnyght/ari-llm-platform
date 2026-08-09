import sqlite3
from pathlib import Path

import sqlite_vec


class VectorIndex:
    DEFAULT_DB_PATH = Path("src") / "data" / "db" / "ari.db"

    def __init__(self):
        self.db_path = str(self.DEFAULT_DB_PATH)
        self.backend = None
        self.initialized = False
        self.connection = None

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

        connection = sqlite3.connect(database_path)
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

    def semantic_search(self, query_text: str, k: int = 5, filters=None):
        return {
            "query_text": query_text,
            "k": k,
            "filters": filters,
            "results": [],
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