"""Ollama backend implementation (async)."""

import httpx
import ollama


class OllamaBackend:
    """Backend for Ollama API using the async client."""

    def __init__(self, host: str = "http://localhost:11434", verify_ssl: bool = True,
                 num_ctx: int = 4096) -> None:
        self.host = host
        self.verify_ssl = verify_ssl
        self.client = ollama.AsyncClient(host=host, verify=verify_ssl)
        self._type = "ollama"
        self._n_ctx = num_ctx
        self._context_tokens: int = 0  # from last call eval_count

    async def initialize(self) -> None:
        """No-op: concrete backends have nothing to detect."""
        return None

    @property
    def type(self) -> str:
        return self._type

    @property
    def n_ctx(self) -> int:
        """Ollama context size (from config passed at construction)."""
        return self._n_ctx

    @property
    def context_tokens(self) -> int:
        return self._context_tokens

    async def chat(self, model: str, messages: list[dict], stream: bool,
                   num_ctx: int = 4096, model_options: dict | None = None,
                   thinking: bool | None = None):
        opts = {"num_ctx": num_ctx, **(model_options or {})}
        # `think` is a top-level kwarg in ollama-python, not an option.
        # Explicit `thinking` arg wins over anything in model_options.
        think_from_opts = opts.pop("think", None)
        effective_think = thinking if thinking is not None else think_from_opts
        kwargs = {"model": model, "messages": messages, "stream": stream, "options": opts}
        if effective_think is not None:
            kwargs["think"] = effective_think
        return await self.client.chat(**kwargs)

    async def list_models(self) -> list[str]:
        from pydantic import ValidationError
        try:
            response = await self.client.list()
            return [m.model for m in response.models]
        except ValidationError:
            # Some servers (e.g. llama.cpp's ollama compat) return partial schema.
            # Fall back to a direct httpx call rather than ollama's private API.
            url = f"{self.host.rstrip('/')}/api/tags"
            async with httpx.AsyncClient(verify=self.verify_ssl) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                data = resp.json()
            return [m.get("model", m.get("name", "")) for m in data.get("models", [])]

    def extract_chunk(self, chunk) -> tuple[str, str]:
        message = chunk.get("message", {})
        reasoning = message.get("thinking", "") or ""
        content = message.get("content", "") or ""
        if chunk.get("eval_count"):
            self._context_tokens = chunk["eval_count"]
        return reasoning, content

    def extract_result(self, result) -> tuple[str, int]:
        content = result["message"]["content"]
        tokens = result.get("eval_count", len(content) // 4)
        self._context_tokens = tokens
        return content, tokens

    async def get_info(self) -> dict:
        return {}
