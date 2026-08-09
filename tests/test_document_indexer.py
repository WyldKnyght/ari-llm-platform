from ari_llm_platform import (
    DocumentIndexer,
    EmbeddingClient,
    VectorIndex,
    migrate_phase3,
)


def test_index_document_stores_chunks_embeddings_and_tags(tmp_path):
    database_path = tmp_path / "ari.db"

    vector_index = VectorIndex()
    vector_index.initialize(str(database_path))
    connection = vector_index.connect()

    def transport(payload):
        return {
            "data": [
                {
                    "embedding": [0.1] * 384,
                }
            ]
        }

    embedding_client = EmbeddingClient(
        base_url="http://example.invalid/v1/embeddings",
        model="test-embedding-model",
        expected_dimension=384,
        transport=transport,
    )

    try:
        migrate_phase3(connection)

        indexer = DocumentIndexer(
            connection=connection,
            embedding_client=embedding_client,
            chunk_size=24,
            chunk_overlap=0,
        )

        result = indexer.index_document(
            source_type="note",
            source_ref="tests/document-indexer",
            title="Generic ARI ingestion fixture",
            raw_text=(
                "Alpha bravo charlie delta echo foxtrot "
                "golf hotel india juliet kilo lima."
            ),
            tags=["phase-3", "fixture"],
            metadata={"status": "test"},
        )

        assert result["document_id"] > 0
        assert result["chunk_count"] > 1

        chunk_count = connection.execute(
            "SELECT COUNT(*) FROM chunks WHERE document_id = ?",
            (result["document_id"],),
        ).fetchone()[0]

        embedding_count = connection.execute(
            "SELECT COUNT(*) FROM chunk_embeddings",
        ).fetchone()[0]

        vector_count = connection.execute(
            "SELECT COUNT(*) FROM chunk_embedding_vectors",
        ).fetchone()[0]

        tag_count = connection.execute(
            "SELECT COUNT(*) FROM document_tags WHERE document_id = ?",
            (result["document_id"],),
        ).fetchone()[0]

        assert chunk_count == result["chunk_count"]
        assert embedding_count == result["chunk_count"]
        assert vector_count == result["chunk_count"]
        assert tag_count == 2
    finally:
        vector_index.close()