import os

from sqlite_vec import serialize_float32

from ari_llm_platform import (
    DocumentIndexer,
    EmbeddingClient,
    VectorIndex,
    migrate_phase3,
)


def require_environment_value(name: str) -> str:
    value = os.environ.get(name)

    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")

    return value


def main() -> None:
    embedding_base_url = require_environment_value("ARI_EMBEDDING_BASE_URL")
    embedding_model = require_environment_value("ARI_EMBEDDING_MODEL")
    embedding_dimensions = int(
        require_environment_value("ARI_EMBEDDING_DIMENSIONS")
    )

    vector_index = VectorIndex()
    vector_index.initialize()
    connection = vector_index.connect()

    try:
        migrate_phase3(connection)

        embedding_client = EmbeddingClient(
            base_url=embedding_base_url,
            model=embedding_model,
            expected_dimension=embedding_dimensions,
        )

        indexer = DocumentIndexer(
            connection=connection,
            embedding_client=embedding_client,
            embedding_dimensions=embedding_dimensions,
        )

        result = indexer.index_document(
            source_type="note",
            source_ref="scripts/verify_document_ingestion.py",
            title="Generic ARI document-ingestion verification",
            raw_text=(
                "This generic ARI fixture verifies the local document "
                "ingestion path. It is not personal data and contains no "
                "private assistant memory. The test confirms that document "
                "chunks receive local embeddings and enter sqlite-vec."
            ),
            tags=["generic-fixture", "phase-3"],
            metadata={
                "purpose": "live document-ingestion verification",
            },
        )

        query_embedding = embedding_client.embed(
            "How does the generic ARI ingestion verification work?"
        )

        nearest_neighbor = connection.execute(
            """
            SELECT rowid, distance
            FROM chunk_embedding_vectors
            WHERE embedding MATCH ?
              AND k = 1
            """,
            (serialize_float32(query_embedding),),
        ).fetchone()

        if nearest_neighbor is None:
            raise RuntimeError("No vector retrieval result was returned.")

        nearest_chunk_id, distance = nearest_neighbor

        nearest_chunk = connection.execute(
            """
            SELECT text
            FROM chunks
            WHERE id = ?
            """,
            (nearest_chunk_id,),
        ).fetchone()

        if nearest_chunk is None:
            raise RuntimeError("Retrieved vector did not map to a chunk.")

        print(f"document_id={result['document_id']}")
        print(f"chunk_count={result['chunk_count']}")
        print(f"nearest_chunk_id={nearest_chunk_id}")
        print(f"distance={distance}")
        print(f"nearest_chunk_text={nearest_chunk[0]}")
    finally:
        vector_index.close()


if __name__ == "__main__":
    main()