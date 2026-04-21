"""OpenAI-compatible backend implementation."""

import httpx
import openai


class OpenAIBackend:
    """Backend for OpenAI-compatible APIs (LM Studio, vLLM, llama.cpp, etc.)."""

    def __init__(self, host: str = "http://localhost:1234", verify_ssl: bool = True) -> None:
        self.host = host
        self.verify_ssl = verify_ssl
        self._type = "openai"
        self._client = None

    @property
    def type(self) -> str:
        return self._type

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
        return chunk.choices[0].delta.content or ""

    def extract_result(self, result) -> tuple[str, int]:
        content = result.choices[0].message.content
        tokens = getattr(result.usage, "completion_tokens", None) or len(content) // 4
        return content, tokens
