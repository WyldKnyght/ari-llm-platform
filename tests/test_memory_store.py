from ari_llm_platform import (
    DocumentIndexer,
    EmbeddingClient,
    MemoryStore,
    VectorIndex,
    migrate_phase3,
)


def create_memory_store(tmp_path):
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

    migrate_phase3(connection)

    document_indexer = DocumentIndexer(
        connection=connection,
        embedding_client=embedding_client,
    )

    memory_store = MemoryStore(
        connection=connection,
        document_indexer=document_indexer,
        embedding_client=embedding_client,
    )

    return vector_index, memory_store


def test_pending_memory_is_not_retrieved_until_promoted(tmp_path):
    vector_index, memory_store = create_memory_store(tmp_path)

    try:
        document_id = memory_store.store_memory_fact(
            text="The generic ARI fixture is used for testing.",
            memory_type="semantic",
            tags=["fixture"],
        )

        assert memory_store.retrieve_memory("What is the fixture?") == []

        memory_store.promote_memory(document_id)

        results = memory_store.retrieve_memory(
            "What is the fixture?"
        )

        assert len(results) == 1
        assert results[0]["document_id"] == document_id
        assert results[0]["status"] == "active"
    finally:
        vector_index.close()


def test_rejected_memory_is_not_retrieved(tmp_path):
    vector_index, memory_store = create_memory_store(tmp_path)

    try:
        document_id = memory_store.store_memory_fact(
            text="This generic test fact should be rejected.",
            memory_type="episodic",
        )

        memory_store.reject_memory(
            document_id,
            reason="Generic test rejection.",
        )

        assert memory_store.retrieve_memory(
            "What test fact was rejected?"
        ) == []
    finally:
        vector_index.close()


def test_consolidation_supersedes_older_memory(tmp_path):
    vector_index, memory_store = create_memory_store(tmp_path)

    try:
        old_document_id = memory_store.store_memory_fact(
            text="The old generic setting is value A.",
            memory_type="semantic",
        )

        new_document_id = memory_store.store_memory_fact(
            text="The current generic setting is value B.",
            memory_type="semantic",
        )

        result = memory_store.consolidate_memory(
            [old_document_id, new_document_id],
        )

        assert result == {
            "active_document_id": new_document_id,
            "superseded_document_ids": [old_document_id],
        }
    finally:
        vector_index.close()