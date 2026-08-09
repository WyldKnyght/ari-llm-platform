import pytest

from ari_llm_platform import LLMClient


def test_generate_returns_response_text_and_payload():
    captured_payload = {}

    def transport(payload):
        captured_payload.update(payload)
        return {
            "choices": [
                {
                    "text": " LLM client test response. ",
                }
            ]
        }

    client = LLMClient(
        base_url="http://example.invalid/v1/completions",
        model="test-model",
        max_tokens=64,
        temperature=0.2,
        transport=transport,
    )

    response = client.generate("Test prompt.")

    assert response == "LLM client test response."
    assert captured_payload == {
        "model": "test-model",
        "prompt": "Test prompt.",
        "max_tokens": 64,
        "temperature": 0.2,
    }


def test_generate_rejects_empty_prompt():
    client = LLMClient(
        base_url="http://example.invalid/v1/completions",
        model="test-model",
        transport=lambda payload: {},
    )

    with pytest.raises(ValueError, match="Prompt must not be empty"):
        client.generate("   ")