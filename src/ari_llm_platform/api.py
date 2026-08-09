import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .llm_client import LLMClient


class AskRequest(BaseModel):
    message: str


class AskResponse(BaseModel):
    reply: str


def create_app(
    llm_client: LLMClient | None = None,
    system_prompt: str | None = None,
) -> FastAPI:
    if llm_client is None:
        llm_client = LLMClient(
            base_url=_require_environment_value("ARI_LLM_BASE_URL"),
            model=_require_environment_value("ARI_LLM_MODEL"),
        )

    if system_prompt is None:
        system_prompt = _load_system_prompt(
            _require_environment_value("ARI_SYSTEM_PROMPT_FILE")
        )

    app = FastAPI(title="ARI LLM Platform")

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        user_message = request.message.strip()

        if not user_message:
            raise HTTPException(
                status_code=400,
                detail="Message must not be empty.",
            )

        prompt = (
            f"{system_prompt}\n\n"
            f"User: {user_message}\n"
            "Assistant:"
        )

        try:
            reply = llm_client.generate(prompt)
        except RuntimeError as error:
            raise HTTPException(
                status_code=502,
                detail=str(error),
            ) from error

        return AskResponse(reply=reply)

    return app


def _require_environment_value(name: str) -> str:
    value = os.environ.get(name)

    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}"
        )

    return value


def _load_system_prompt(path: str) -> str:
    system_prompt = Path(path).read_text(encoding="utf-8").strip()

    if not system_prompt:
        raise RuntimeError("System prompt file is empty.")

    return system_prompt