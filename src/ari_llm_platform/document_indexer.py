import json
import sqlite3
from typing import Any

import sqlite_vec

from .embedding_client import EmbeddingClient


class DocumentIndexer:
    def __init__(
        self,
        connection: sqlite3.Connection,
        embedding_client: EmbeddingClient,
        embedding_dimensions: int = 384,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")

        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be zero or greater and smaller than chunk_size."
            )

        self.connection = connection
        self.embedding_client = embedding_client
        self.embedding_dimensions = embedding_dimensions
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def index_document(
        self,
        source_type: str,
        source_ref: str | None,
        title: str,
        raw_text: str,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, int]:
        if not raw_text.strip():
            raise ValueError("Document text must not be empty.")

        chunks = self._chunk_text(raw_text)
        metadata_json = json.dumps(metadata or {}, sort_keys=True)

        try:
            self.connection.execute("BEGIN")

            document_cursor = self.connection.execute(
                """
                INSERT INTO documents (
                    source_type,
                    source_ref,
                    title,
                    metadata
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    source_type,
                    source_ref,
                    title,
                    metadata_json,
                ),
            )

            document_id = document_cursor.lastrowid

            self._attach_tags(document_id, tags or [])

            for position, chunk_text in enumerate(chunks):
                chunk_cursor = self.connection.execute(
                    """
                    INSERT INTO chunks (
                        document_id,
                        position,
                        text,
                        token_count
                    )
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        document_id,
                        position,
                        chunk_text,
                        len(chunk_text.split()),
                    ),
                )

                chunk_id = chunk_cursor.lastrowid
                embedding = self.embedding_client.embed(chunk_text)

                if len(embedding) != self.embedding_dimensions:
                    raise RuntimeError(
                        f"Expected {self.embedding_dimensions} dimensions, "
                        f"received {len(embedding)}."
                    )

                embedding_blob = sqlite_vec.serialize_float32(embedding)

                self.connection.execute(
                    """
                    INSERT INTO chunk_embeddings (
                        chunk_id,
                        embedding,
                        dim,
                        model
                    )
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        chunk_id,
                        embedding_blob,
                        self.embedding_dimensions,
                        self.embedding_client.model,
                    ),
                )

                self.connection.execute(
                    """
                    INSERT INTO chunk_embedding_vectors (
                        rowid,
                        embedding
                    )
                    VALUES (?, ?)
                    """,
                    (
                        chunk_id,
                        embedding_blob,
                    ),
                )

            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

        return {
            "document_id": document_id,
            "chunk_count": len(chunks),
        }

    def _attach_tags(self, document_id: int, tags: list[str]) -> None:
        for tag_name in tags:
            clean_tag_name = tag_name.strip()

            if not clean_tag_name:
                continue

            self.connection.execute(
                """
                INSERT INTO tags (name)
                VALUES (?)
                ON CONFLICT(name) DO NOTHING
                """,
                (clean_tag_name,),
            )

            tag_id = self.connection.execute(
                """
                SELECT id
                FROM tags
                WHERE name = ?
                """,
                (clean_tag_name,),
            ).fetchone()[0]

            self.connection.execute(
                """
                INSERT OR IGNORE INTO document_tags (
                    document_id,
                    tag_id
                )
                VALUES (?, ?)
                """,
                (
                    document_id,
                    tag_id,
                ),
            )

    def _chunk_text(self, raw_text: str) -> list[str]:
        chunks = []
        start = 0

        while start < len(raw_text):
            end = min(start + self.chunk_size, len(raw_text))
            chunks.append(raw_text[start:end])

            if end == len(raw_text):
                break

            start = end - self.chunk_overlap

        return chunks