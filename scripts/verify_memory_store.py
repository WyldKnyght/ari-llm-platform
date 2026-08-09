import os

from ari_llm_platform import (
    DocumentIndexer,
    EmbeddingClient,
    MemoryStore,
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

        document_indexer = DocumentIndexer(
            connection=connection,
            embedding_client=embedding_client,
            embedding_dimensions=embedding_dimensions,
        )

        memory_store = MemoryStore(
            connection=connection,
            document_indexer=document_indexer,
            embedding_client=embedding_client,
        )

        pending_document_id = memory_store.store_memory_fact(
            text=(
                "This generic ARI memory fixture verifies that pending "
                "records are not returned until explicitly promoted."
            ),
            memory_type="semantic",
            tags=["generic-fixture", "phase-3"],
            metadata={
                "purpose": "live memory-helper verification",
            },
        )

        pending_results = memory_store.retrieve_memory(
            "How are pending generic memory records handled?"
        )

        if any(
            result["document_id"] == pending_document_id
            for result in pending_results
        ):
            raise RuntimeError(
                "Pending memory was retrieved before promotion."
            )

        memory_store.promote_memory(pending_document_id)

        active_results = memory_store.retrieve_memory(
            "How are pending generic memory records handled?"
        )

        if not any(
            result["document_id"] == pending_document_id
            for result in active_results
        ):
            raise RuntimeError(
                "Promoted memory was not retrieved as active."
            )

        old_document_id = memory_store.store_memory_fact(
            text="The old generic Phase 3 fixture setting is value A.",
            memory_type="semantic",
            tags=["generic-fixture", "phase-3"],
        )
        memory_store.promote_memory(old_document_id)

        new_document_id = memory_store.store_memory_fact(
            text="The current generic Phase 3 fixture setting is value B.",
            memory_type="semantic",
            tags=["generic-fixture", "phase-3"],
        )
        memory_store.promote_memory(new_document_id)

        consolidation_result = memory_store.consolidate_memory(
            [old_document_id, new_document_id],
        )

        rejected_document_id = memory_store.store_memory_fact(
            text="This generic memory fixture is intentionally rejected.",
            memory_type="episodic",
            tags=["generic-fixture", "phase-3"],
        )
        memory_store.reject_memory(
            rejected_document_id,
            reason="Live generic rejection check.",
        )

        rejected_results = memory_store.retrieve_memory(
            "Which generic memory fixture is intentionally rejected?"
        )

        if any(
            result["document_id"] == rejected_document_id
            for result in rejected_results
        ):
            raise RuntimeError(
                "Rejected memory was retrieved."
            )

        print(f"pending_document_id={pending_document_id}")
        print(f"active_document_id={pending_document_id}")
        print(f"superseded_document_id={old_document_id}")
        print(
            "consolidated_active_document_id="
            f"{consolidation_result['active_document_id']}"
        )
        print(f"rejected_document_id={rejected_document_id}")
        print("memory_store_live_verification=passed")
    finally:
        vector_index.close()


if __name__ == "__main__":
    main()
    