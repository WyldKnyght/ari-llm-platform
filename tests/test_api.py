from fastapi.testclient import TestClient

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