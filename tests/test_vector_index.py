import threading

import sqlite_vec

from ari_llm_platform import VectorIndex, migrate_phase3


def test_vector_index_initialize_sets_ready_path_and_backend():
    vi = VectorIndex()
    assert vi.is_ready() is False
    assert vi.db_path == r"src\data\db\ari.db"
    assert vi.backend is None

    vi.initialize(r"D:\ARI\Data\ari.db")

    assert vi.is_ready() is True
    assert vi.db_path == r"D:\ARI\Data\ari.db"
    assert vi.backend == "sqlite-vec"


def test_vector_index_load_backend_missing_returns_status():
    vi = VectorIndex()
    vi.initialize(r"D:\ARI\Data\ari.db")
    assert vi.load_backend() in ("sqlite-vec-loaded", "sqlite-vec-missing")


def test_vector_index_embed_text_contract():
    vi = VectorIndex()
    vi.initialize(r"D:\ARI\Data\ari.db")
    result = vi.embed_text("hello")
    assert result["text"] == "hello"
    assert result["embedding"] == []
    assert result["dim"] == 0
    assert result["backend"] == "sqlite-vec"


def test_vector_index_upsert_chunk_embedding_contract():
    vi = VectorIndex()
    result = vi.upsert_chunk_embedding(
        chunk_id=7,
        embedding=[0.1, 0.2],
        dim=2,
        model="test-model",
        backend="sqlite-vec",
    )
    assert result["chunk_id"] == 7
    assert result["stored"] is True
    assert result["dim"] == 2
    assert result["model"] == "test-model"
    assert result["backend"] == "sqlite-vec"

class FakeEmbeddingClient:
    def embed(self, text: str) -> list[float]:
        assert text == "How does generic retrieval work?"
        return [0.0] * 384


def test_vector_index_semantic_search_returns_ranked_chunk(tmp_path):
    database_path = tmp_path / "semantic_search.db"

    vector_index = VectorIndex(
        embedding_client=FakeEmbeddingClient(),
    )
    vector_index.initialize(str(database_path))
    connection = vector_index.connect()

    try:
        _extracted_from_test_vector_index_semantic_search_returns_ranked_chunk_11(
            connection, vector_index
        )
    finally:
        vector_index.close()


# TODO Rename this here and in `test_vector_index_semantic_search_returns_ranked_chunk`
def _extracted_from_test_vector_index_semantic_search_returns_ranked_chunk_11(connection, vector_index):
    migrate_phase3(connection)

    document_id = connection.execute(
        """
            INSERT INTO documents (source_type, title, metadata)
            VALUES (?, ?, ?)
            """,
        (
            "document",
            "Generic retrieval fixture",
            "{}",
        ),
    ).lastrowid

    chunk_id = connection.execute(
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
            0,
            "Generic ARI retrieval fixture.",
            4,
        ),
    ).lastrowid

    vector = sqlite_vec.serialize_float32([0.0] * 384)

    connection.execute(
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
            vector,
            384,
            "test-embedding-model",
        ),
    )

    connection.execute(
        """
            INSERT INTO chunk_embedding_vectors (
                rowid,
                embedding
            )
            VALUES (?, ?)
            """,
        (
            chunk_id,
            vector,
        ),
    )
    connection.commit()

    result = vector_index.semantic_search(
        "How does generic retrieval work?",
        k=1,
    )

    assert result["query_text"] == "How does generic retrieval work?"
    assert result["k"] == 1
    assert result["filters"] is None
    assert len(result["results"]) == 1
    assert result["results"][0]["document_id"] == document_id
    assert result["results"][0]["chunk_id"] == chunk_id
    assert result["results"][0]["text"] == "Generic ARI retrieval fixture."
    assert result["results"][0]["source_type"] == "document"
    assert result["results"][0]["distance"] == 0.0

def test_vector_index_semantic_search_excludes_memory_by_default(tmp_path):
    database_path = tmp_path / "semantic_search_memory_filter.db"

    vector_index = VectorIndex(
        embedding_client=FakeEmbeddingClient(),
    )
    vector_index.initialize(str(database_path))
    connection = vector_index.connect()

    try:
        _extracted_from_test_vector_index_semantic_search_excludes_memory_by_default_11(
            connection, vector_index
        )
    finally:
        vector_index.close()


# TODO Rename this here and in `test_vector_index_semantic_search_excludes_memory_by_default`
def _extracted_from_test_vector_index_semantic_search_excludes_memory_by_default_11(connection, vector_index):
    migrate_phase3(connection)

    document_id = connection.execute(
        """
            INSERT INTO documents (source_type, title, metadata)
            VALUES (?, ?, ?)
            """,
        (
            "document",
            "Generic retrieval fixture",
            "{}",
        ),
    ).lastrowid

    document_chunk_id = connection.execute(
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
            0,
            "Generic ARI retrieval fixture.",
            4,
        ),
    ).lastrowid

    memory_document_id = connection.execute(
        """
            INSERT INTO documents (source_type, title, metadata)
            VALUES (?, ?, ?)
            """,
        (
            "memory",
            "Semantic memory",
            '{"memory_type": "semantic", "status": "active"}',
        ),
    ).lastrowid

    memory_chunk_id = connection.execute(
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
            memory_document_id,
            0,
            "Generic memory fixture that must not become RAG context.",
            9,
        ),
    ).lastrowid

    document_vector = sqlite_vec.serialize_float32([0.1] * 384)
    memory_vector = sqlite_vec.serialize_float32([0.0] * 384)

    for chunk_id, vector in (
        (document_chunk_id, document_vector),
        (memory_chunk_id, memory_vector),
    ):
        connection.execute(
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
                vector,
                384,
                "test-embedding-model",
            ),
        )

        connection.execute(
            """
                INSERT INTO chunk_embedding_vectors (
                    rowid,
                    embedding
                )
                VALUES (?, ?)
                """,
            (
                chunk_id,
                vector,
            ),
        )

    connection.commit()

    result = vector_index.semantic_search(
        "How does generic retrieval work?",
        k=1,
    )

    assert len(result["results"]) == 1
    assert result["results"][0]["document_id"] == document_id
    assert result["results"][0]["source_type"] == "document"
    assert result["results"][0]["document_id"] != memory_document_id

def test_vector_index_reindex_document_contract():
    vi = VectorIndex()
    result = vi.reindex_document(42)
    assert result["document_id"] == 42
    assert result["reindexed"] is False

def test_vector_index_vacuum_contract():
    vi = VectorIndex()
    result = vi.vacuum()
    assert result["vacuumed"] is False
    assert result["backend"] is None

def test_connect_allows_connection_use_from_a_request_thread(tmp_path):
    vector_index = VectorIndex()
    vector_index.initialize(str(tmp_path / "ari.db"))
    connection = vector_index.connect()

    errors = []

    def execute_query():
        try:
            connection.execute("SELECT 1").fetchone()
        except Exception as error:  # noqa: BLE001
            errors.append(error)

    request_thread = threading.Thread(target=execute_query)
    request_thread.start()
    request_thread.join()

    vector_index.close()

    assert not errors
