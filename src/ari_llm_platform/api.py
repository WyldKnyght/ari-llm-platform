import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .document_indexer import DocumentIndexer
from .embedding_client import EmbeddingClient
from .llm_client import LLMClient
from .memory_store import MemoryStore
from .vault_schema import migrate_phase3
from .vector_index import VectorIndex


class AskRequest(BaseModel):
    message: str


class AskResponse(BaseModel):
    reply: str


def create_app(
    llm_client: LLMClient | None = None,
    system_prompt: str | None = None,
    memory_store=None,
    vector_index=None,
    memory_enabled: bool | None = None,
    rag_enabled: bool | None = None,
) -> FastAPI:
    if memory_enabled is None:
        memory_enabled = _environment_flag("MEMORY_ENABLED")

    if rag_enabled is None:
        rag_enabled = _environment_flag("RAG_ENABLED")

    if llm_client is None:
        llm_client = LLMClient(
            base_url=_require_environment_value("ARI_LLM_BASE_URL"),
            model=_require_environment_value("ARI_LLM_MODEL"),
        )

    if system_prompt is None:
        system_prompt = _load_system_prompt(
            _require_environment_value("ARI_SYSTEM_PROMPT_FILE")
        )

    if (
        (memory_enabled and memory_store is None)
        or (rag_enabled and vector_index is None)
    ):
        runtime_memory_store, runtime_vector_index = (
            _bootstrap_retrieval_dependencies()
        )

        if memory_store is None:
            memory_store = runtime_memory_store

        if vector_index is None:
            vector_index = runtime_vector_index

    app = FastAPI(title="ARI LLM Platform")

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        user_message = request.message.strip()

        if not user_message:
            raise HTTPException(
                status_code=400,
                detail="Message must not be empty.",
            )

        prompt_sections = [system_prompt]

        if memory_enabled and memory_store is not None:
            memory_results = memory_store.retrieve_memory(
                user_message,
                k=5,
            )

            if memory_results:
                memory_lines = [
                    f"- {item['text']}"
                    for item in memory_results
                ]
                prompt_sections.append(
                    "Memory:\n" + "\n".join(memory_lines)
                )

        if rag_enabled and vector_index is not None:
            search_result = vector_index.semantic_search(
                user_message,
                k=5,
            )
            context_results = search_result["results"]

            if context_results:
                context_lines = [
                    f"- {item['text']}"
                    for item in context_results
                ]
                prompt_sections.append(
                    "Context:\n" + "\n".join(context_lines)
                )

        prompt_sections.append(
            f"User: {user_message}\nAssistant:"
        )

        prompt = "\n\n".join(prompt_sections)
        try:
            reply = llm_client.generate(prompt)
        except RuntimeError as error:
            raise HTTPException(
                status_code=502,
                detail=str(error),
            ) from error

        return AskResponse(reply=reply)

    return app

def _bootstrap_retrieval_dependencies() -> tuple[
    MemoryStore,
    VectorIndex,
]:
    embedding_dimensions = int(
        _require_environment_value("ARI_EMBEDDING_DIMENSIONS")
    )

    if embedding_dimensions != 384:
        raise RuntimeError(
            "The current sqlite-vec Phase 3 schema requires "
            "ARI_EMBEDDING_DIMENSIONS=384."
        )

    embedding_client = EmbeddingClient(
        base_url=_require_environment_value("ARI_EMBEDDING_BASE_URL"),
        model=_require_environment_value("ARI_EMBEDDING_MODEL"),
        expected_dimension=embedding_dimensions,
    )

    database_path = os.environ.get(
        "ARI_DATABASE_PATH",
        str(VectorIndex.DEFAULT_DB_PATH),
    )

    vector_index = VectorIndex(
        embedding_client=embedding_client,
    )
    vector_index.initialize(database_path)
    connection = vector_index.connect()

    try:
        migrate_phase3(connection)

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
    except Exception:
        vector_index.close()
        raise

    return memory_store, vector_index

def _environment_flag(name: str) -> bool:
    value = os.environ.get(name, "")

    return value.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
def _require_environment_value(name: str) -> str:
    if value := os.environ.get(name):
        return value
    else:
        raise RuntimeError(
            f"Missing required environment variable: {name}"
        )


def _load_system_prompt(path: str) -> str:
    if system_prompt := Path(path).read_text(encoding="utf-8").strip():
        return system_prompt
    else:
        raise RuntimeError("System prompt file is empty.")