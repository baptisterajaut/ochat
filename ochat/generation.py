"""Generation mixin: streaming, API calls, response generation."""

import asyncio
import json
import logging
import time
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING, Any

from textual.widgets import Input, Static

from ochat.widgets import ChatContainer, Message

if TYPE_CHECKING:
    from ochat.backend.base import BackendProtocol

_log = logging.getLogger(__name__)

# Minimum interval between Markdown widget updates during streaming.
# mistune re-parses the full buffer on every update → throttle to avoid O(n²) blowup.
_MARKDOWN_RENDER_INTERVAL = 0.05  # seconds


def _clean_impersonate_response(response: str) -> str:
    """Strip quotes and collapse whitespace from impersonate results."""
    response = response.strip()
    if response.startswith('"') and response.endswith('"'):
        response = response[1:-1]
    return " ".join(response.split())


class GenerationMixin:
    """Mixin providing LLM generation capabilities for OChat."""

    # Type-check-only declarations: these attributes and methods are actually
    # provided by the composed OChat class (see ochat.app.OChat). Declaring
    # them here satisfies pyright without runtime cost. Never instantiate
    # GenerationMixin directly — it is a pure mixin.
    # pylint: disable=missing-function-docstring,unused-argument
    if TYPE_CHECKING:
        # --- state owned by OChat.__init__ ---
        backend: "BackendProtocol"
        model: str
        num_ctx: int
        model_options: dict
        messages: list[dict]
        is_generating: bool
        streaming: bool
        auto_suggest: bool
        sys_instructions: dict
        last_gen_time: float
        last_tokens: int
        last_ttft: float
        total_tokens: int
        _auto_suggest_task: asyncio.Task | None
        _pending_suggestion: str
        _generation_cancelled: bool
        _context_warning_shown: bool
        SPINNER_FRAMES: list[str]

        # --- methods defined on OChat ---
        # Stub bodies raise NotImplementedError so pylint/astroid don't treat
        # them as no-op shadows of the real implementations on OChat.
        def _context_pct(self) -> float:
            raise NotImplementedError
        def _generating_lock(self) -> AbstractAsyncContextManager[None]:
            raise NotImplementedError
        def _status_text(self, extra: str = "") -> str:
            raise NotImplementedError
        async def _show_system_message(self, text: str) -> None:
            raise NotImplementedError

        # --- subset of textual.app.App surface actually used here ---
        def query_one(self, selector: str, expect_type: type | None = None) -> Any:
            raise NotImplementedError

    async def _chat_call(self, messages: list[dict], stream: bool):
        return await self.backend.chat(self.model, messages, stream, self.num_ctx, self.model_options)

    def _extract_chunk(self, chunk) -> tuple[str, str]:
        """Extract (reasoning, content) from a streaming chunk."""
        return self.backend.extract_chunk(chunk)

    def _extract_result(self, result) -> tuple[str, int]:
        """Extract (content, token_count) from a non-streaming result."""
        return self.backend.extract_result(result)

    async def _generate_response(self) -> None:
        """Generate assistant response (streaming or not)."""
        self._generation_cancelled = False
        chat = self.query_one("#chat", ChatContainer)
        status = self.query_one("#status", Static)

        assistant_msg = Message("...", "assistant")
        await chat.mount(assistant_msg)
        chat.scroll_end(animate=False)
        await asyncio.sleep(0)  # Let UI refresh

        start_time = time.time()
        text = ""
        reasoning = ""
        tokens = 0
        cancelled = False

        async with self._generating_lock():
            try:
                text, reasoning, tokens, cancelled = await self._run_stream(
                    assistant_msg, chat, status, start_time,
                )
                await self._finalize_response(
                    assistant_msg, chat, start_time, text, reasoning, tokens, cancelled,
                )
            except json.JSONDecodeError:
                _log.exception("Generation error (response parse failed)")
                await assistant_msg.update(
                    f"● **Error:** server responded but the format didn't match "
                    f"backend `{self.backend.type}` — likely wrong backend in config "
                    f"(`[defaults] backend` in `~/.config/ochat/config.conf`)."
                )
            except Exception as e:  # CancelledError is BaseException → propagates naturally
                _log.exception("Generation error")
                await assistant_msg.update(f"● **Error:** {e}")

        # Auto-suggest after successful generation (background task; lock has released)
        if not cancelled and self.auto_suggest:
            if self._auto_suggest_task and not self._auto_suggest_task.done():
                self._auto_suggest_task.cancel()
            self._auto_suggest_task = asyncio.create_task(self._run_auto_suggest())

        # One-shot context warning
        if not self._context_warning_shown and self._context_pct() > 80:
            self._context_warning_shown = True
            remaining = 100 - self._context_pct()
            await self._show_system_message(
                f"⚠ Approximately {remaining:.0f}% context length remaining, consider compacting (`/compact`)"
            )

    async def _run_stream(self, assistant_msg, chat, status, start_time):
        """Run the chat stream start-to-finish, returning (text, reasoning, tokens, cancelled)."""
        stream, first_chunk = await self._start_stream(
            self.messages, assistant_msg, status, start_time
        )
        self.last_ttft = time.time() - start_time if not self._generation_cancelled else 0.0

        text, reasoning, tokens = "", "", 0
        if first_chunk is not None and not self._generation_cancelled:
            r, c = self._extract_chunk(first_chunk)
            if c:
                text += c
                tokens += 1
            if r:
                reasoning += r

        return await self._consume_chunks(
            stream, assistant_msg, chat, status, start_time, text, reasoning, tokens,
        )

    async def _finalize_response(self, assistant_msg, chat, start_time,
                                 text, reasoning, tokens, cancelled):
        """Render the final message state and commit to history."""
        if cancelled:
            await assistant_msg.update("*[cancelled]*", reasoning=reasoning)
            if text:
                self.messages.append({"role": "assistant", "content": text})
            return

        if text:
            if self.streaming:
                # Throttling may have skipped the last chunk — ensure final render is complete.
                await assistant_msg.update(f"● {text}", reasoning=reasoning)
            else:
                think_time = time.time() - start_time
                await assistant_msg.update(
                    f"*thought for {think_time:.1f}s*\n\n{text}", reasoning=reasoning,
                )
            chat.scroll_end(animate=False)

        if text or reasoning:
            self.messages.append({"role": "assistant", "content": text})
            self.last_gen_time = time.time() - start_time
            self.last_tokens = tokens
            self.total_tokens += tokens
        else:
            await assistant_msg.update("● *[no response]*")

    async def _consume_chunks(self, stream, assistant_msg, chat, status,
                              start_time, response_text, response_reasoning, tokens_generated):
        """Consume stream chunks, updating UI. Returns (text, reasoning, tokens, cancelled).

        Markdown widget updates are throttled (~50ms) to avoid O(n²) re-parsing
        of the accumulated buffer. Status bar updates remain per-chunk (cheap).
        """
        last_render = 0.0
        async for chunk in stream:
            if self._generation_cancelled:
                return response_text, response_reasoning, tokens_generated, True

            reasoning, content = self._extract_chunk(chunk)
            if content:
                response_text += content
                tokens_generated += 1
            if reasoning:
                response_reasoning += reasoning

            elapsed = time.time() - start_time

            if self.streaming:
                now = time.monotonic()
                if (content or reasoning) and (now - last_render) >= _MARKDOWN_RENDER_INTERVAL:
                    await assistant_msg.update(
                        f"● {response_text}",
                        reasoning=response_reasoning,
                    )
                    chat.scroll_end(animate=False)
                    last_render = now
                tps = tokens_generated / elapsed if elapsed > 0 else 0
                status.update(self._status_text(
                    f"generating... {elapsed:.1f}s ({tokens_generated}tok, {tps:.1f}t/s)"
                ))
            else:
                frame = self.SPINNER_FRAMES[int(elapsed * 10) % len(self.SPINNER_FRAMES)]
                await assistant_msg.update(
                    f"● {frame} thinking... {elapsed:.1f}s ({tokens_generated} chunks)",
                    reasoning=response_reasoning,
                )
                chat.scroll_end(animate=False)
                status.update(self._status_text(
                    f"thinking... {elapsed:.1f}s ({tokens_generated} chunks)"
                ))

        return response_text, response_reasoning, tokens_generated, self._generation_cancelled

    async def _start_stream(self, messages: list[dict], msg: Message, status: Static, start_time: float):
        """Start a streaming API call with a spinner while waiting for the first token.

        Returns (stream_async_iterator, first_chunk) — first_chunk may be None.
        """
        spinner = asyncio.create_task(
            self._animate_spinner(msg, status, start_time, "waiting for first token")
        )
        try:
            stream = await self._chat_call(messages, stream=True)
            first_chunk = await self._first_chunk(stream)
        finally:
            spinner.cancel()
            try:
                await spinner
            except asyncio.CancelledError:
                pass
        return stream, first_chunk

    @staticmethod
    async def _first_chunk(stream):
        """Fetch the first item from an async iterator, or None if exhausted."""
        async for chunk in stream:
            return chunk
        return None

    async def _run_auto_suggest(self) -> None:
        """Generate a short suggestion for the user's next message (background)."""
        try:
            suggest_messages = self.messages.copy()
            suggest_messages.append({
                "role": "system",
                "content": self.sys_instructions["impersonate_short"],
            })
            result = await self._chat_call(suggest_messages, stream=False)
            response, _ = self._extract_result(result)
            response = _clean_impersonate_response(response)

            input_widget = self.query_one("#chat-input", Input)
            if not input_widget.value and not self.is_generating:
                self._pending_suggestion = response
                input_widget.placeholder = response
                _log.debug("Auto-suggest set: %s", response[:80])
        except Exception:  # CancelledError is BaseException, propagates through naturally
            _log.debug("Auto-suggest failed", exc_info=True)

    async def _animate_spinner(self, msg: Message, status: Static, start_time: float, label: str) -> None:
        """Animate spinner indicator with given label."""
        i = 0
        while True:
            elapsed = time.time() - start_time
            if self._generation_cancelled:
                await msg.update(f"● {self.SPINNER_FRAMES[i]} cancelling...")
                status.update(self._status_text("cancelling..."))
            else:
                await msg.update(f"● {self.SPINNER_FRAMES[i]} {label}...")
                status.update(self._status_text(f"{label}... {elapsed:.1f}s"))
            self.query_one("#chat", ChatContainer).scroll_end(animate=False)
            i = (i + 1) % len(self.SPINNER_FRAMES)
            await asyncio.sleep(0.1)
