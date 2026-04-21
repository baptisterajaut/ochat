"""Llama.cpp server backend (OpenAI-compatible with real usage tracking, async)."""

import httpx
import openai


class LlamaCppBackend:
    """Backend for llama.cpp server (/v1/chat/completions + /info)."""

    def __init__(self, host: str = "http://localhost:8080", verify_ssl: bool = True) -> None:
        self.host = host
        self.verify_ssl = verify_ssl
        self._type = "llama_cpp"
        self._client: openai.AsyncOpenAI | None = None
        self._n_ctx: int = 0  # from /info
        self._context_tokens: int = 0  # from last call usage
        self._info_cache: dict | None = None

    async def initialize(self) -> None:
        return None

    @property
    def type(self) -> str:
        return self._type

    @property
    def n_ctx(self) -> int:
        # NOTE: cannot fetch lazily here (sync property). Callers that need a
        # correct n_ctx should await get_info() first — _show_greeting does this
        # via list_models → but n_ctx specifically is refreshed on demand via
        # get_info(). Returns 0 until get_info() has been awaited at least once.
        return self._n_ctx

    @property
    def context_tokens(self) -> int:
        return self._context_tokens

    @property
    def client(self) -> openai.AsyncOpenAI:
        if self._client is None:
            base_url = f"{self.host.rstrip('/')}/v1"
            if not self.verify_ssl:
                http_client = httpx.AsyncClient(verify=False)
                self._client = openai.AsyncOpenAI(
                    base_url=base_url, api_key="llama.cpp",
                    http_client=http_client,
                )
            else:
                self._client = openai.AsyncOpenAI(
                    base_url=base_url, api_key="llama.cpp",
                )
        return self._client

    async def _refresh_info(self) -> None:
        """Fetch n_ctx from /v1/models or /info (llama.cpp-specific)."""
        self._info_cache = {}
        self._n_ctx = 4096  # fallback
        async with httpx.AsyncClient(verify=self.verify_ssl) as client:
            for path in ["/v1/models", "/info"]:
                try:
                    url = f"{self.host.rstrip('/')}{path}"
                    resp = await client.get(url)
                    resp.raise_for_status()
                    data = resp.json()
                    if path == "/v1/models":
                        models_data = data.get("data", [])
                        if models_data and "n_ctx" in models_data[0]:
                            self._n_ctx = models_data[0]["n_ctx"]
                            return
                        # n_ctx not exposed on /v1/models, try /info
                    else:
                        self._info_cache = data
                        self._n_ctx = data.get("n_ctx", 4096)
                        return
                except Exception:
                    continue

    async def get_info(self) -> dict:
        if self._info_cache is None:
            await self._refresh_info()
        return self._info_cache or {}

    async def chat(self, model: str, messages: list[dict], stream: bool,
                   num_ctx: int = 4096, model_options: dict | None = None):
        opts = {"n_ctx": num_ctx}
        if model_options:
            opts.update(model_options)
        extra_body = opts if opts else None
        return await self.client.chat.completions.create(
            model=model, messages=messages, stream=stream,
            stream_options={"include_usage": True} if stream else None,
            extra_body=extra_body,
        )

    async def list_models(self) -> list[str]:
        # Also refresh n_ctx opportunistically — llama.cpp exposes it on /v1/models
        if self._info_cache is None:
            await self._refresh_info()
        response = await self.client.models.list()
        return [m.id for m in response.data]

    def extract_chunk(self, chunk) -> tuple[str, str]:
        choices = chunk.choices
        if not choices:
            # Last chunk carries only usage (no choices)
            if chunk.usage is not None:
                self._context_tokens = chunk.usage.prompt_tokens
            return "", ""
        delta = choices[0].delta
        reasoning = getattr(delta, "reasoning_content", "") or ""
        content = delta.content if (delta and delta.content) else ""
        if chunk.usage is not None:
            self._context_tokens = chunk.usage.prompt_tokens
        return reasoning, content

    def extract_result(self, result) -> tuple[str, int]:
        content = result.choices[0].message.content
        usage = result.usage
        if usage is not None:
            self._context_tokens = usage.prompt_tokens
            total_tokens = usage.prompt_tokens + usage.completion_tokens
        else:
            total_tokens = len(content) // 4
        return content, total_tokens
