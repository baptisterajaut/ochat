"""Backend abstraction package for ochat."""

import asyncio

from ochat.backend.base import BackendProtocol
from ochat.backend.ollama import OllamaBackend
from ochat.backend.openai import OpenAIBackend
from ochat.backend.llama_cpp import LlamaCppBackend

__all__ = [
    "BackendProtocol",
    "OllamaBackend",
    "OpenAIBackend",
    "LlamaCppBackend",
    "create_backend",
    "AutoBackend",
]


def create_backend(backend_type: str, host: str, verify_ssl: bool,
                   num_ctx: int = 4096) -> BackendProtocol:
    """Factory to create backend instances."""
    match backend_type:
        case "ollama":
            return OllamaBackend(host=host, verify_ssl=verify_ssl, num_ctx=num_ctx)
        case "openai":
            return OpenAIBackend(host=host, verify_ssl=verify_ssl)
        case "llama_cpp":
            return LlamaCppBackend(host=host, verify_ssl=verify_ssl)
        case _:
            raise ValueError(f"Unknown backend type: {backend_type}")


class AutoBackend:
    """Automatically detect backend by trying Ollama first, then llama.cpp, then OpenAI-compatible."""

    def __init__(self, host: str = "http://localhost:11434", verify_ssl: bool = True,
                 num_ctx: int = 4096) -> None:
        self.host = host
        self.verify_ssl = verify_ssl
        self._ollama = OllamaBackend(host=host, verify_ssl=verify_ssl, num_ctx=num_ctx)
        self._llama_cpp = LlamaCppBackend(host=host, verify_ssl=verify_ssl)
        self._openai = OpenAIBackend(host=host, verify_ssl=verify_ssl)
        self._detected_backend: BackendProtocol | None = None
        self._type = "auto"
        self._init_lock = asyncio.Lock()

    @property
    def type(self) -> str:
        if self._detected_backend is None:
            return "auto"
        return self._detected_backend.type

    @property
    def context_tokens(self) -> int:
        if self._detected_backend is not None:
            return self._detected_backend.context_tokens
        return 0

    @property
    def n_ctx(self) -> int:
        if self._detected_backend is not None:
            return self._detected_backend.n_ctx
        return 0

    async def initialize(self) -> None:
        """Detect the underlying backend exactly once (locked, idempotent)."""
        if self._detected_backend is not None:
            return
        async with self._init_lock:
            if self._detected_backend is not None:
                return  # another coroutine won the race

            # Try Ollama first
            try:
                await self._ollama.list_models()
                self._detected_backend = self._ollama
                return
            except Exception:
                pass

            # Try llama.cpp via /v1/models
            try:
                await self._llama_cpp.list_models()
                self._detected_backend = self._llama_cpp
                return
            except Exception:
                pass

            # Try OpenAI-compatible
            try:
                await self._openai.list_models()
                self._detected_backend = self._openai
                return
            except Exception:
                pass

            raise RuntimeError("Could not detect backend: Ollama, llama.cpp, and OpenAI-compatible all failed")

    async def chat(self, model: str, messages: list[dict], stream: bool,
                   num_ctx: int = 4096, model_options: dict | None = None):
        await self.initialize()
        assert self._detected_backend is not None
        return await self._detected_backend.chat(model, messages, stream, num_ctx, model_options)

    async def list_models(self) -> list[str]:
        await self.initialize()
        assert self._detected_backend is not None
        return await self._detected_backend.list_models()

    def extract_chunk(self, chunk) -> tuple[str, str]:
        # By the time chunks arrive, chat() has already been awaited → detection is done.
        assert self._detected_backend is not None
        return self._detected_backend.extract_chunk(chunk)

    def extract_result(self, result) -> tuple[str, int]:
        assert self._detected_backend is not None
        return self._detected_backend.extract_result(result)

    async def get_info(self) -> dict:
        await self.initialize()
        assert self._detected_backend is not None
        return await self._detected_backend.get_info()
