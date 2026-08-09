import pytest

from ari_llm_platform import EmbeddingClient


def test_embed_returns_expected_vector_and_payload():
    captured_payload = {}

    def transport(payload):
        captured_payload.update(payload)
        return {
            "data": [
                {
                    "embedding": [0.1, 0.2, 0.3],
                }
            ]
        }

    client = EmbeddingClient(
        base_url="http://example.invalid/v1/embeddings",
        model="test-embedding-model",
        expected_dimension=3,
        transport=transport,
    )

    vector = client.embed("Embedding test text.")

    assert vector == [0.1, 0.2, 0.3]
    assert captured_payload == {
        "model": "test-embedding-model",
        "input": "Embedding test text.",
    }


def test_embed_rejects_wrong_vector_dimension():
    client = EmbeddingClient(
        base_url="http://example.invalid/v1/embeddings",
        model="test-embedding-model",
        expected_dimension=3,
        transport=lambda payload: {
            "data": [
                {
                    "embedding": [0.1, 0.2],
                }
            ]
        },
    )

    with pytest.raises(RuntimeError, match="Expected 3 dimensions"):
        client.embed("Embedding test text.")
        