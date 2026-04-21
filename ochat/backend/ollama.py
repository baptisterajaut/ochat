"""Ollama backend implementation."""

import ollama


class OllamaBackend:
    """Backend for Ollama API."""

    def __init__(self, host: str = "http://localhost:11434", verify_ssl: bool = True) -> None:
        self.client = ollama.Client(host=host, verify=verify_ssl)
        self._type = "ollama"

    @property
    def type(self) -> str:
        return self._type

    def chat(self, model: str, messages: list[dict], stream: bool,
             num_ctx: int = 4096, model_options: dict | None = None) -> dict:
        options = {"num_ctx": num_ctx, **(model_options or {})}
        return self.client.chat(
            model=model, messages=messages, stream=stream, options=options,
        )

    def list_models(self) -> list[str]:
        response = self.client.list()
        return [m.model for m in response.models]

    def extract_chunk(self, chunk) -> str:
        return chunk.get("message", {}).get("content", "")

    def extract_result(self, result) -> tuple[str, int]:
        content = result["message"]["content"]
        tokens = result.get("eval_count", len(content) // 4)
        return content, tokens
