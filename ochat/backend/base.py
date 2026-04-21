from typing import Protocol


class BackendProtocol(Protocol):
    """Protocol for LLM backend implementations.

    All I/O methods are async. Streaming calls return an async iterator of chunks.
    """

    async def initialize(self) -> None:
        """Optional one-time setup (e.g., backend detection for AutoBackend).

        Must be idempotent: concrete backends can no-op, AutoBackend performs
        detection exactly once under an internal lock.
        """
        ...

    async def chat(self, model: str, messages: list[dict], stream: bool,
                   num_ctx: int = 4096, model_options: dict | None = None):
        """Make a chat completion call.

        When stream=True, returns an async iterator yielding chunks.
        When stream=False, returns a single result object.
        """
        ...

    async def list_models(self) -> list[str]:
        """List available model names."""
        ...

    def extract_chunk(self, chunk) -> tuple[str, str]:
        """Extract (reasoning, content) from a streaming chunk. Pure/sync."""
        ...

    def extract_result(self, result) -> tuple[str, int]:
        """Extract (content, token_count) from a non-streaming result. Pure/sync."""
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

    async def get_info(self) -> dict:
        """Fetch backend-specific server info (e.g., llama.cpp /info endpoint)."""
        ...
