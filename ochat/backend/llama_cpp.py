"""Llama.cpp server backend (OpenAI-compatible with real usage tracking)."""

import httpx
import openai

from .base import BackendProtocol  # noqa: F401


class LlamaCppBackend:
    """Backend for llama.cpp server (/v1/chat/completions + /info)."""

    def __init__(self, host: str = "http://localhost:8080", verify_ssl: bool = True) -> None:
        self.host = host
        self.verify_ssl = verify_ssl
        self._type = "llama_cpp"
        self._client: openai.OpenAI | None = None
        self._n_ctx: int = 0  # from /info
        self._context_tokens: int = 0  # from last call usage
        self._info_cache: dict | None = None

    @property
    def type(self) -> str:
        return self._type

    @property
    def n_ctx(self) -> int:
        if self._info_cache is None:
            self._refresh_info()
        return self._n_ctx

    @property
    def context_tokens(self) -> int:
        return self._context_tokens

    @property
    def client(self):
        if self._client is None:
            base_url = f"{self.host.rstrip('/')}/v1"
            if not self.verify_ssl:
                http_client = httpx.Client(verify=False)
                self._client = openai.OpenAI(
                    base_url=base_url, api_key="llama.cpp",
                    http_client=http_client,
                )
            else:
                self._client = openai.OpenAI(
                    base_url=base_url, api_key="llama.cpp",
                )
        return self._client

    def _refresh_info(self) -> None:
        """Fetch n_ctx from /v1/models or /info (llama.cpp-specific)."""
        self._info_cache = {}
        self._n_ctx = 4096  # fallback
        for path in ["/v1/models", "/info"]:
            try:
                url = f"{self.host.rstrip('/')}{path}"
                with httpx.Client(verify=self.verify_ssl) as client:
                    resp = client.get(url)
                    resp.raise_for_status()
                    data = resp.json()
                    if path == "/v1/models":
                        models_data = data.get("data", [])
                        if models_data:
                            self._n_ctx = models_data[0].get("n_ctx", 4096)
                            return
                    else:
                        self._info_cache = data
                        self._n_ctx = data.get("n_ctx", 4096)
                        return
            except Exception:
                continue

    def get_info(self) -> dict:
        if self._info_cache is None:
            self._refresh_info()
        return self._info_cache or {}

    def chat(self, model: str, messages: list[dict], stream: bool,
             num_ctx: int = 4096, model_options: dict | None = None) -> dict:
        opts = {"n_ctx": num_ctx}
        if model_options:
            opts.update(model_options)
        extra_body = opts if opts else None
        return self.client.chat.completions.create(
            model=model, messages=messages, stream=stream,
            stream_options={"include_usage": True} if stream else None,
            extra_body=extra_body,
        )

    def list_models(self) -> list[str]:
        response = self.client.models.list()
        return [m.id for m in response.data]

    def extract_chunk(self, chunk) -> str:
        choices = chunk.choices
        if not choices:
            return ""
        delta = choices[0].delta
        content = delta.content if (delta and delta.content) else ""
        # Check if this is the last chunk with usage (delta is None when usage included)
        if chunk.usage is not None:
            self._context_tokens = chunk.usage.prompt_tokens
        return content

    def extract_result(self, result) -> tuple[str, int]:
        content = result.choices[0].message.content
        usage = result.usage
        if usage is not None:
            self._context_tokens = usage.prompt_tokens
            total_tokens = usage.prompt_tokens + usage.completion_tokens
        else:
            total_tokens = len(content) // 4
        return content, total_tokens
