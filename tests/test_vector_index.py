from ari_llm_platform import VectorIndex


def test_vector_index_initialize_sets_ready_path_and_backend():
    vi = VectorIndex()
    assert vi.is_ready() is False
    assert vi.db_path is None
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


def test_vector_index_semantic_search_empty_contract():
    vi = VectorIndex()
    result = vi.semantic_search("hello world", k=3, filters={"tag": "x"})
    assert result["query_text"] == "hello world"
    assert result["k"] == 3
    assert result["filters"] == {"tag": "x"}
    assert result["results"] == []


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