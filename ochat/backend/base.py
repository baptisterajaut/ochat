from typing import Protocol


class BackendProtocol(Protocol):
    """Protocol for LLM backend implementations."""

    def chat(self, messages: list[dict], stream: bool):
        """Make a chat completion call. Returns stream or result."""
        ...

    def list_models(self) -> list[str]:
        """List available model names."""
        ...

    def extract_chunk(self, chunk) -> str:
        """Extract text content from a streaming chunk."""
        ...

    def extract_result(self, result) -> tuple[str, int]:
        """Extract (content, token_count) from a non-streaming result."""
        ...

    @property
    def type(self) -> str:
        """Backend type identifier ('ollama', 'openai', 'llama_cpp')."""
        ...

    @property
    def n_ctx(self) -> int:
        """Context window size. For llama.cpp: from /info (server-determined).
        For ollama: from client config. For openai: 0 (unknown)."""
        ...

    @property
    def context_tokens(self) -> int:
        """Actual prompt token count from the last API call."""
        ...

    def get_info(self) -> dict:
        """Fetch backend-specific server info (e.g., llama.cpp /info endpoint)."""
        ...
