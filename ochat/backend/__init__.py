"""Backend abstraction package for ochat."""

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


def create_backend(backend_type: str, host: str, verify_ssl: bool) -> BackendProtocol:
    """Factory to create backend instances."""
    match backend_type:
        case "ollama":
            return OllamaBackend(host=host, verify_ssl=verify_ssl)
        case "openai":
            return OpenAIBackend(host=host, verify_ssl=verify_ssl)
        case "llama_cpp":
            return LlamaCppBackend(host=host, verify_ssl=verify_ssl)
        case _:
            raise ValueError(f"Unknown backend type: {backend_type}")


class AutoBackend:
    """Automatically detect backend by trying Ollama first, then llama.cpp, then OpenAI-compatible."""

    def __init__(self, host: str = "http://localhost:11434", verify_ssl: bool = True) -> None:
        self.host = host
        self.verify_ssl = verify_ssl
        self._ollama = OllamaBackend(host=host, verify_ssl=verify_ssl)
        self._llama_cpp = LlamaCppBackend(host=host, verify_ssl=verify_ssl)
        self._openai = OpenAIBackend(host=host, verify_ssl=verify_ssl)
        self._detected_backend: BackendProtocol | None = None
        self._type = "auto"

    @property
    def type(self) -> str:
        if self._detected_backend is None:
            return "auto"
        return self._detected_backend.type

    def _detect(self) -> BackendProtocol:
        if self._detected_backend is not None:
            return self._detected_backend

        # Try Ollama first
        try:
            self._ollama.list_models()
            self._detected_backend = self._ollama
            return self._detected_backend
        except Exception:
            pass

        # Try llama.cpp via /v1/models
        try:
            self._llama_cpp.list_models()
            self._detected_backend = self._llama_cpp
            return self._detected_backend
        except Exception:
            pass

        # Try OpenAI-compatible
        try:
            self._openai.list_models()
            self._detected_backend = self._openai
            return self._detected_backend
        except Exception:
            pass

        raise RuntimeError("Could not detect backend: Ollama, llama.cpp, and OpenAI-compatible all failed")

    def chat(self, model: str, messages: list[dict], stream: bool,
             num_ctx: int = 4096, model_options: dict | None = None) -> dict:
        backend = self._detect()
        return backend.chat(model, messages, stream, num_ctx, model_options)

    def list_models(self) -> list[str]:
        backend = self._detect()
        return backend.list_models()

    def extract_chunk(self, chunk) -> str:
        backend = self._detect()
        return backend.extract_chunk(chunk)

    def extract_result(self, result) -> tuple[str, int]:
        backend = self._detect()
        return backend.extract_result(result)
