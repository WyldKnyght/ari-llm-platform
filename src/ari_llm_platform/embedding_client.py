import json
from collections.abc import Callable
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class EmbeddingClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        expected_dimension: int | None = None,
        timeout_seconds: int = 120,
        transport: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ):
        self.base_url = base_url
        self.model = model
        self.expected_dimension = expected_dimension
        self.timeout_seconds = timeout_seconds
        self.transport = transport

    def embed(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("Embedding text must not be empty.")

        payload = {
            "model": self.model,
            "input": text,
        }

        response = (
            self.transport(payload)
            if self.transport is not None
            else self._post(payload)
        )

        try:
            embedding = response["data"][0]["embedding"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError(
                "Embedding response did not contain data[0].embedding."
            ) from error

        vector = [float(value) for value in embedding]

        if (
            self.expected_dimension is not None
            and len(vector) != self.expected_dimension
        ):
            raise RuntimeError(
                f"Expected {self.expected_dimension} dimensions, "
                f"received {len(vector)}."
            )

        return vector

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            raise RuntimeError(
                f"Embedding backend returned HTTP {error.code}."
            ) from error
        except URLError as error:
            raise RuntimeError(
                f"Embedding backend is unavailable: {error.reason}."
            ) from error