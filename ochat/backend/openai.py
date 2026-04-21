"""OpenAI-compatible backend implementation."""

import httpx
import openai


class OpenAIBackend:
    """Backend for OpenAI-compatible APIs (LM Studio, vLLM, llama.cpp, etc.)."""

    def __init__(self, host: str = "http://localhost:1234", verify_ssl: bool = True) -> None:
        self.host = host
        self.verify_ssl = verify_ssl
        self._type = "openai"
        self._client: openai.OpenAI | None = None
        self._context_tokens: int = 0  # from last call usage.prompt_tokens

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
    def client(self):
        if self._client is None:
            base_url = f"{self.host.rstrip('/')}/v1"
            if not self.verify_ssl:
                http_client = httpx.Client(verify=False)
                self._client = openai.OpenAI(
                    base_url=base_url, api_key="not-needed",
                    http_client=http_client,
                )
            else:
                self._client = openai.OpenAI(
                    base_url=base_url, api_key="not-needed",
                )
        return self._client

    def chat(self, model: str, messages: list[dict], stream: bool,
             num_ctx: int = 4096, model_options: dict | None = None) -> dict:
        return self.client.chat.completions.create(
            model=model, messages=messages, stream=stream,
        )

    def list_models(self) -> list[str]:
        response = self.client.models.list()
        return [m.id for m in response.data]

    def extract_chunk(self, chunk) -> str:
        content = chunk.choices[0].delta.content or ""
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
