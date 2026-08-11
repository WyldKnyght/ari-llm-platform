from unittest.mock import Mock

from fastapi.testclient import TestClient

import ari_llm_platform.api as api_module
from ari_llm_platform.api import create_app


class FakeLLMClient:
    def __init__(self):
        self.prompt = None

    def generate(self, prompt: str) -> str:
        self.prompt = prompt
        return "Test API response."


def test_ask_returns_llm_reply_and_builds_prompt():
    fake_llm_client = FakeLLMClient()

    app = create_app(
        llm_client=fake_llm_client,
        system_prompt="Test system prompt.",
    )
    client = TestClient(app)

    response = client.post(
        "/ask",
        json={
            "message": "Test user message.",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "reply": "Test API response.",
    }
    assert fake_llm_client.prompt == (
        "Test system prompt.\n\n"
        "User: Test user message.\n"
        "Assistant:"
    )

class FakeMemoryStore:
    def retrieve_memory(self, query_text: str, k: int = 5):
        assert query_text == "What is the generic ARI retrieval fixture?"
        assert k == 5
        return [
            {
                "document_id": 9,
                "text": "The generic fixture preference is value B.",
                "memory_type": "semantic",
                "status": "active",
                "distance": 0.2,
            }
        ]


class FakeVectorIndex:
    def semantic_search(
        self,
        query_text: str,
        k: int = 5,
        filters=None,
    ):
        assert query_text == "What is the generic ARI retrieval fixture?"
        assert k == 5
        assert filters is None
        return {
            "query_text": query_text,
            "k": k,
            "filters": filters,
            "results": [
                {
                    "document_id": 6,
                    "chunk_id": 6,
                    "text": "Generic ARI retrieval fixture context.",
                    "source_type": "note",
                    "distance": 0.1,
                }
            ],
        }


def test_ask_includes_feature_gated_memory_and_context():
    fake_llm_client = FakeLLMClient()

    app = create_app(
        llm_client=fake_llm_client,
        system_prompt="Test system prompt.",
        memory_store=FakeMemoryStore(),
        vector_index=FakeVectorIndex(),
        memory_enabled=True,
        rag_enabled=True,
    )
    client = TestClient(app)

    response = client.post(
        "/ask",
        json={
            "message": "What is the generic ARI retrieval fixture?",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "reply": "Test API response.",
    }
    assert fake_llm_client.prompt == (
        "Test system prompt.\n\n"
        "Memory:\n"
        "- The generic fixture preference is value B.\n\n"
        "Context:\n"
        "- Generic ARI retrieval fixture context.\n\n"
        "User: What is the generic ARI retrieval fixture?\n"
        "Assistant:"
    )

def test_ask_rejects_empty_message():
    app = create_app(
        llm_client=FakeLLMClient(),
        system_prompt="Test system prompt.",
    )
    client = TestClient(app)

    response = client.post(
        "/ask",
        json={
            "message": "   ",
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Message must not be empty.",
    }

class FailingMemoryStore:
    def retrieve_memory(self, query_text: str, k: int = 5):
        raise AssertionError("Memory retrieval must not run when disabled.")


class FailingVectorIndex:
    def semantic_search(
        self,
        query_text: str,
        k: int = 5,
        filters=None,
    ):
        raise AssertionError("Generic RAG search must not run when disabled.")


def test_ask_does_not_retrieve_when_features_are_disabled():
    fake_llm_client = FakeLLMClient()

    app = create_app(
        llm_client=fake_llm_client,
        system_prompt="Test system prompt.",
        memory_store=FailingMemoryStore(),
        vector_index=FailingVectorIndex(),
        memory_enabled=False,
        rag_enabled=False,
    )
    client = TestClient(app)

    response = client.post(
        "/ask",
        json={
            "message": "Keep retrieval disabled.",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "reply": "Test API response.",
    }
    assert fake_llm_client.prompt == (
        "Test system prompt.\n\n"
        "User: Keep retrieval disabled.\n"
        "Assistant:"
    )

def test_ask_retrieves_only_for_the_enabled_feature():
    memory_only_llm_client = FakeLLMClient()

    memory_only_app = create_app(
        llm_client=memory_only_llm_client,
        system_prompt="Test system prompt.",
        memory_store=FakeMemoryStore(),
        vector_index=FailingVectorIndex(),
        memory_enabled=True,
        rag_enabled=False,
    )
    _extracted_from_test_ask_retrieves_only_for_the_enabled_feature_12(
        memory_only_app,
        memory_only_llm_client,
        "Test system prompt.\n\n"
        "Memory:\n"
        "- The generic fixture preference is value B.\n\n"
        "User: What is the generic ARI retrieval fixture?\n"
        "Assistant:",
    )
    rag_only_llm_client = FakeLLMClient()

    rag_only_app = create_app(
        llm_client=rag_only_llm_client,
        system_prompt="Test system prompt.",
        memory_store=FailingMemoryStore(),
        vector_index=FakeVectorIndex(),
        memory_enabled=False,
        rag_enabled=True,
    )
    _extracted_from_test_ask_retrieves_only_for_the_enabled_feature_12(
        rag_only_app,
        rag_only_llm_client,
        "Test system prompt.\n\n"
        "Context:\n"
        "- Generic ARI retrieval fixture context.\n\n"
        "User: What is the generic ARI retrieval fixture?\n"
        "Assistant:",
    )


# TODO Rename this here and in `test_ask_retrieves_only_for_the_enabled_feature`
def _extracted_from_test_ask_retrieves_only_for_the_enabled_feature_12(arg0, arg1, arg2):
    memory_only_client = TestClient(arg0)

    memory_only_response = memory_only_client.post(
        "/ask",
        json={
            "message": "What is the generic ARI retrieval fixture?",
        },
    )

    assert memory_only_response.status_code == 200
    assert arg1.prompt == arg2

def test_ask_reads_retrieval_feature_flags_from_environment(monkeypatch):
    monkeypatch.setenv("MEMORY_ENABLED", "true")
    monkeypatch.setenv("RAG_ENABLED", "true")

    fake_llm_client = FakeLLMClient()

    app = create_app(
        llm_client=fake_llm_client,
        system_prompt="Test system prompt.",
        memory_store=FakeMemoryStore(),
        vector_index=FakeVectorIndex(),
    )
    client = TestClient(app)

    response = client.post(
        "/ask",
        json={
            "message": "What is the generic ARI retrieval fixture?",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "reply": "Test API response.",
    }
    assert fake_llm_client.prompt == (
        "Test system prompt.\n\n"
        "Memory:\n"
        "- The generic fixture preference is value B.\n\n"
        "Context:\n"
        "- Generic ARI retrieval fixture context.\n\n"
        "User: What is the generic ARI retrieval fixture?\n"
        "Assistant:"
    )

def test_ask_bootstraps_retrieval_dependencies_from_environment(monkeypatch):
    monkeypatch.setenv("MEMORY_ENABLED", "true")
    monkeypatch.setenv("RAG_ENABLED", "true")
    monkeypatch.setenv(
        "ARI_EMBEDDING_BASE_URL",
        "http://127.0.0.1:1234/v1/embeddings",
    )
    monkeypatch.setenv(
        "ARI_EMBEDDING_MODEL",
        "text-embedding-bge-small-en-v1.5",
    )
    monkeypatch.setenv("ARI_EMBEDDING_DIMENSIONS", "384")
    monkeypatch.setenv("ARI_DATABASE_PATH", r"C:\runtime\ari.db")

    fake_embedding_client = object()
    fake_connection = object()

    embedding_client_factory = Mock(
        return_value=fake_embedding_client,
    )

    fake_vector_index = Mock()
    fake_vector_index.connect.return_value = fake_connection
    fake_vector_index.semantic_search.return_value = {
        "results": [
            {
                "text": "Generic runtime RAG context.",
            }
        ]
    }
    vector_index_factory = Mock(
        return_value=fake_vector_index,
    )

    document_indexer_factory = Mock(
        return_value=object(),
    )

    fake_memory_store = Mock()
    fake_memory_store.retrieve_memory.return_value = [
        {
            "text": "Active runtime memory.",
        }
    ]
    memory_store_factory = Mock(
        return_value=fake_memory_store,
    )

    migrate_phase3 = Mock()

    monkeypatch.setattr(
        api_module,
        "EmbeddingClient",
        embedding_client_factory,
        raising=False,
    )
    monkeypatch.setattr(
        api_module,
        "VectorIndex",
        vector_index_factory,
        raising=False,
    )
    monkeypatch.setattr(
        api_module,
        "DocumentIndexer",
        document_indexer_factory,
        raising=False,
    )
    monkeypatch.setattr(
        api_module,
        "MemoryStore",
        memory_store_factory,
        raising=False,
    )
    monkeypatch.setattr(
        api_module,
        "migrate_phase3",
        migrate_phase3,
        raising=False,
    )

    fake_llm_client = FakeLLMClient()

    app = api_module.create_app(
        llm_client=fake_llm_client,
        system_prompt="Test system prompt.",
    )
    client = TestClient(app)

    response = client.post(
        "/ask",
        json={
            "message": "What is runtime retrieval?",
        },
    )

    assert response.status_code == 200

    embedding_client_factory.assert_called_once_with(
        base_url="http://127.0.0.1:1234/v1/embeddings",
        model="text-embedding-bge-small-en-v1.5",
        expected_dimension=384,
    )
    vector_index_factory.assert_called_once_with(
        embedding_client=fake_embedding_client,
    )
    fake_vector_index.initialize.assert_called_once_with(
        r"C:\runtime\ari.db"
    )
    fake_vector_index.connect.assert_called_once_with()
    migrate_phase3.assert_called_once_with(fake_connection)

    document_indexer_factory.assert_called_once_with(
        connection=fake_connection,
        embedding_client=fake_embedding_client,
        embedding_dimensions=384,
    )
    memory_store_factory.assert_called_once_with(
        connection=fake_connection,
        document_indexer=document_indexer_factory.return_value,
        embedding_client=fake_embedding_client,
    )

    assert fake_llm_client.prompt == (
        "Test system prompt.\n\n"
        "Memory:\n"
        "- Active runtime memory.\n\n"
        "Context:\n"
        "- Generic runtime RAG context.\n\n"
        "User: What is runtime retrieval?\n"
        "Assistant:"
    )