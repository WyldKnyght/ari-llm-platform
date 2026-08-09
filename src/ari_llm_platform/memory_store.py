import json
import sqlite3
from typing import Any

from sqlite_vec import serialize_float32

from .document_indexer import DocumentIndexer
from .embedding_client import EmbeddingClient


class MemoryStore:
    VALID_MEMORY_TYPES = {"episodic", "semantic"}
    VALID_STATUSES = {
        "pending",
        "active",
        "superseded",
        "archived",
        "rejected",
    }

    def __init__(
        self,
        connection: sqlite3.Connection,
        document_indexer: DocumentIndexer,
        embedding_client: EmbeddingClient,
    ):
        self.connection = connection
        self.document_indexer = document_indexer
        self.embedding_client = embedding_client

    def store_memory_fact(
        self,
        text: str,
        memory_type: str,
        importance: str | float | int | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        status: str = "pending",
    ) -> int:
        if memory_type not in self.VALID_MEMORY_TYPES:
            raise ValueError(
                f"Unsupported memory_type: {memory_type}"
            )

        if status not in self.VALID_STATUSES:
            raise ValueError(f"Unsupported memory status: {status}")

        memory_metadata = dict(metadata or {})
        memory_metadata.update(
            {
                "memory_type": memory_type,
                "importance": importance,
                "origin_system": memory_metadata.get(
                    "origin_system",
                    "ari",
                ),
                "review_required": status == "pending",
                "status": status,
            }
        )

        result = self.document_indexer.index_document(
            source_type="memory",
            source_ref=None,
            title=f"{memory_type.capitalize()} memory",
            raw_text=text,
            tags=tags,
            metadata=memory_metadata,
        )

        return result["document_id"]

    def retrieve_memory(
        self,
        query_text: str,
        memory_types: list[str] | None = None,
        k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if memory_types is not None:
            invalid_types = set(memory_types) - self.VALID_MEMORY_TYPES

            if invalid_types:
                raise ValueError(
                    f"Unsupported memory types: {sorted(invalid_types)}"
                )

        query_embedding = self.embedding_client.embed(query_text)
        candidate_limit = max(k * 4, k)

        vector_rows = self.connection.execute(
            """
            SELECT rowid, distance
            FROM chunk_embedding_vectors
            WHERE embedding MATCH ?
              AND k = ?
            """,
            (
                serialize_float32(query_embedding),
                candidate_limit,
            ),
        ).fetchall()

        results = []
        seen_document_ids = set()

        for chunk_id, distance in vector_rows:
            row = self.connection.execute(
                """
                SELECT
                    documents.id,
                    documents.metadata,
                    chunks.text
                FROM chunks
                JOIN documents ON documents.id = chunks.document_id
                WHERE chunks.id = ?
                  AND documents.source_type = 'memory'
                """,
                (chunk_id,),
            ).fetchone()

            if row is None:
                continue

            document_id, metadata_json, text = row

            if document_id in seen_document_ids:
                continue

            metadata = json.loads(metadata_json)

            if metadata.get("status") != "active":
                continue

            if (
                memory_types is not None
                and metadata.get("memory_type") not in memory_types
            ):
                continue

            if not self._matches_filters(metadata, filters or {}):
                continue

            results.append(
                {
                    "document_id": document_id,
                    "text": text,
                    "distance": distance,
                    "memory_type": metadata.get("memory_type"),
                    "importance": metadata.get("importance"),
                    "status": metadata.get("status"),
                }
            )

            seen_document_ids.add(document_id)

            if len(results) == k:
                break

        return results

    def promote_memory(self, document_id: int) -> None:
        metadata = self._get_memory_metadata(document_id)
        metadata["status"] = "active"
        metadata["review_required"] = False
        self._save_metadata(document_id, metadata)

    def reject_memory(
        self,
        document_id: int,
        reason: str | None = None,
    ) -> None:
        metadata = self._get_memory_metadata(document_id)
        metadata["status"] = "rejected"
        metadata["review_required"] = False

        if reason:
            metadata["rejection_reason"] = reason

        self._save_metadata(document_id, metadata)

    def consolidate_memory(
        self,
        document_ids: list[int],
        strategy: str = "supersede_with_latest",
    ) -> dict[str, Any]:
        if strategy != "supersede_with_latest":
            raise ValueError(
                "Only supersede_with_latest is implemented."
            )

        unique_document_ids = sorted(set(document_ids))

        if len(unique_document_ids) < 2:
            raise ValueError(
                "Consolidation requires at least two memory documents."
            )

        active_document_id = unique_document_ids[-1]
        superseded_document_ids = unique_document_ids[:-1]

        active_metadata = self._get_memory_metadata(active_document_id)
        active_metadata["status"] = "active"
        active_metadata["review_required"] = False
        active_metadata["supersedes"] = superseded_document_ids
        self._save_metadata(active_document_id, active_metadata)

        for document_id in superseded_document_ids:
            metadata = self._get_memory_metadata(document_id)
            metadata["status"] = "superseded"
            metadata["superseded_by"] = active_document_id
            metadata["review_required"] = False
            self._save_metadata(document_id, metadata)

        return {
            "active_document_id": active_document_id,
            "superseded_document_ids": superseded_document_ids,
        }

    def _get_memory_metadata(self, document_id: int) -> dict[str, Any]:
        row = self.connection.execute(
            """
            SELECT source_type, metadata
            FROM documents
            WHERE id = ?
            """,
            (document_id,),
        ).fetchone()

        if row is None:
            raise ValueError(f"Memory document not found: {document_id}")

        source_type, metadata_json = row

        if source_type != "memory":
            raise ValueError(
                f"Document {document_id} is not a memory record."
            )

        return json.loads(metadata_json)

    def _save_metadata(
        self,
        document_id: int,
        metadata: dict[str, Any],
    ) -> None:
        self.connection.execute(
            """
            UPDATE documents
            SET
                metadata = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (
                json.dumps(metadata, sort_keys=True),
                document_id,
            ),
        )
        self.connection.commit()

    @staticmethod
    def _matches_filters(
        metadata: dict[str, Any],
        filters: dict[str, Any],
    ) -> bool:
        return all(
            metadata.get(name) == value
            for name, value in filters.items()
        )
    