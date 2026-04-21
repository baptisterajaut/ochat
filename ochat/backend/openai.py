"""OpenAI-compatible backend implementation (async)."""

import httpx
import openai


class OpenAIBackend:
    """Backend for OpenAI-compatible APIs (LM Studio, vLLM, llama.cpp, etc.)."""

    def __init__(self, host: str = "http://localhost:1234", verify_ssl: bool = True) -> None:
        self.host = host
        self.verify_ssl = verify_ssl
        self._type = "openai"
        self._client: openai.AsyncOpenAI | None = None
        self._context_tokens: int = 0  # from last call usage.prompt_tokens

    async def initialize(self) -> None:
        return None

    @property
    def type(self) -> str:
        return self._type

    @property
    def n_ctx(self) -> int:
        """OpenAI mode: unknown (server-determined)."""
        return 0

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
                    base_url=base_url, api_key="not-needed",
                    http_client=http_client,
                )
            else:
                self._client = openai.AsyncOpenAI(
                    base_url=base_url, api_key="not-needed",
                )
        return self._client

    async def chat(self, model: str, messages: list[dict], stream: bool,
                   num_ctx: int = 4096, model_options: dict | None = None,
                   thinking: bool | None = None):
        # num_ctx and model_options are ignored by OpenAI backend — kept for
        # BackendProtocol interface compat. Acknowledge to silence unused-param.
        _ = num_ctx, model_options
        extra_body = None
        if thinking is not None:
            # Servers supporting Qwen3-style thinking toggle via chat_template_kwargs
            # (vLLM, LM Studio with Qwen3, etc.). Real OpenAI API ignores/rejects —
            # only sent when caller explicitly asks.
            extra_body = {"chat_template_kwargs": {"enable_thinking": thinking}}
        return await self.client.chat.completions.create(
            model=model, messages=messages, stream=stream,
            extra_body=extra_body,
        )

    async def list_models(self) -> list[str]:
        response = await self.client.models.list()
        return [m.id for m in response.data]

    def extract_chunk(self, chunk) -> tuple[str, str]:
        delta = chunk.choices[0].delta if chunk.choices else None
        reasoning = getattr(delta, "reasoning_content", "") or "" if delta else ""
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

    async def get_info(self) -> dict:
        return {}
