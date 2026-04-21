"""Ollama backend implementation."""

import ollama


class OllamaBackend:
    """Backend for Ollama API."""

    def __init__(self, host: str = "http://localhost:11434", verify_ssl: bool = True) -> None:
        self.client = ollama.Client(host=host, verify=verify_ssl)
        self._type = "ollama"
        self._context_tokens: int = 0  # from last call eval_count

    @property
    def type(self) -> str:
        return self._type

    @property
    def n_ctx(self) -> int:
        """Ollama context size (client-configurable)."""
        return 4096  # default; actual value tracked by caller

    @property
    def context_tokens(self) -> int:
        return self._context_tokens

    def chat(self, model: str, messages: list[dict], stream: bool,
             num_ctx: int = 4096, model_options: dict | None = None) -> dict:
        options = {"num_ctx": num_ctx, **(model_options or {})}
        result = self.client.chat(
            model=model, messages=messages, stream=stream, options=options,
        )
        # Track usage from the first chunk (ollama streaming yields chunks with eval_count)
        if isinstance(result, dict):
            # Non-streaming
            self._context_tokens = result.get("eval_count", 0)
        return result

    def list_models(self) -> list[str]:
        response = self.client.list()
        return [m.model for m in response.models]

    def extract_chunk(self, chunk) -> str:
        content = chunk.get("message", {}).get("content", "")
        if chunk.get("eval_count"):
            self._context_tokens = chunk["eval_count"]
        return content

    def extract_result(self, result) -> tuple[str, int]:
        content = result["message"]["content"]
        tokens = result.get("eval_count", len(content) // 4)
        self._context_tokens = tokens
        return content, tokens
