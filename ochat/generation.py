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
        thinking_enabled: bool
        sys_instructions: dict
        last_gen_time: float
        last_tokens: int
        last_ttft: float
        total_tokens: int
        _auto_suggest_task: asyncio.Task | None
        _pending_suggestion: str
        _generation_cancelled: bool
        _context_warning_shown: bool
        _reasoning_collapsed_pref: bool
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

    async def _chat_call(self, messages: list[dict], stream: bool,
                          thinking: bool | None = None):
        return await self.backend.chat(
            self.model, messages, stream, self.num_ctx, self.model_options,
            thinking=thinking,
        )

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

        assistant_msg = Message(
            "...", "assistant",
            reasoning_collapsed=self._reasoning_collapsed_pref,
        )
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

        # Clear spinner leftover, then start the incremental stream. The prefix
        # `● ` goes in via update_content so it's included in the parsed source
        # but not re-parsed on every chunk.
        if self.streaming:
            await assistant_msg.update_content("● ")
            assistant_msg.start_content_stream()

        text, reasoning, tokens = "", "", 0
        if first_chunk is not None and not self._generation_cancelled:
            r, c = self._extract_chunk(first_chunk)
            if r and self.streaming:
                assistant_msg.start_reasoning_stream()
                await assistant_msg.append_reasoning(r)
                reasoning += r
            elif r:
                reasoning += r
            if c:
                text += c
                tokens += 1
                if self.streaming:
                    assistant_msg.mark_reasoning_idle()
                    await assistant_msg.append_content(c)

        return await self._consume_chunks(
            stream, assistant_msg, chat, status, start_time, text, reasoning, tokens,
        )

    async def _finalize_response(self, assistant_msg, chat, start_time,
                                 text, reasoning, tokens, cancelled):
        """Render the final message state and commit to history."""
        if cancelled:
            # update() stops the active stream and replaces content.
            await assistant_msg.update("*[cancelled]*", reasoning=reasoning)
            if text:
                self.messages.append({"role": "assistant", "content": text})
            return

        if text:
            if self.streaming:
                # Streams already rendered both content and reasoning
                # incrementally. Just stop them and scroll.
                await assistant_msg._stop_stream()  # noqa: SLF001 — widget helper
                await assistant_msg.stop_reasoning_stream()
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

        Streaming mode: content and reasoning deltas both go to MarkdownStream
        (incremental parse — only the trailing block re-renders per batch;
        Textual coalesces writes when renders lag). Status bar stays throttled.
        """
        last_status_render = 0.0
        last_spinner_render = 0.0
        chunks_received = 0
        async for chunk in stream:
            if self._generation_cancelled:
                return response_text, response_reasoning, tokens_generated, True

            chunks_received += 1
            reasoning, content = self._extract_chunk(chunk)
            if content:
                response_text += content
                tokens_generated += 1
            if reasoning:
                response_reasoning += reasoning

            elapsed = time.time() - start_time

            if self.streaming:
                if content:
                    assistant_msg.mark_reasoning_idle()
                    await assistant_msg.append_content(content)
                    chat.scroll_end(animate=False)
                if reasoning:
                    # Lazy start — reasoning block stays hidden until first token.
                    assistant_msg.start_reasoning_stream()
                    await assistant_msg.append_reasoning(reasoning)
                    chat.scroll_end(animate=False)
                now = time.monotonic()
                if now - last_status_render >= 0.2:
                    tps = tokens_generated / elapsed if elapsed > 0 else 0
                    status.update(self._status_text(
                        f"generating... {elapsed:.1f}s ({tokens_generated}tok, {tps:.1f}t/s)"
                    ))
                    last_status_render = now
            else:
                # Non-stream: show only the spinner in the body. Reasoning
                # is deliberately NOT forwarded per-chunk — each call to
                # Message.update(reasoning=X) does a full Markdown replace
                # on the reasoning block. Throttle the body spinner too:
                # update_content runs Markdown.update (parse + remove &
                # mount all blocks, ~20-50ms per call), so doing it at
                # chunk-rate (up to 100 Hz) starves the event loop and
                # causes the UI to lag seconds behind the stream. 10 Hz is
                # indistinguishable to the eye and leaves headroom.
                now = time.monotonic()
                if now - last_spinner_render < 0.1:
                    continue
                last_spinner_render = now
                if tokens_generated > 0:
                    phase = "on it"
                elif response_reasoning:
                    phase = "thinking"
                else:
                    phase = "waiting for first token"
                frame = self.SPINNER_FRAMES[int(elapsed * 10) % len(self.SPINNER_FRAMES)]
                await assistant_msg.update_content(
                    f"● {frame} {phase}... {elapsed:.1f}s ({chunks_received} chunks)",
                )
                chat.scroll_end(animate=False)
                status.update(self._status_text(
                    f"{phase}... {elapsed:.1f}s ({chunks_received} chunks)"
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
            # thinking: None=default; False=force-off via user `/thinking` toggle
            thinking = None if self.thinking_enabled else False
            stream = await self._chat_call(messages, stream=True, thinking=thinking)
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
                "role": "user",
                "content": self.sys_instructions["impersonate_short"],
            })
            # thinking=False: meta-prompt, reasoning tokens are pure waste here
            result = await self._chat_call(suggest_messages, stream=False, thinking=False)
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
        """Animate spinner indicator with given label.

        Uses update_content (not update) — each tick is just a body refresh,
        we don't want to stop the reasoning stream or post CollapseChanged
        events at 10 Hz.
        """
        i = 0
        while True:
            elapsed = time.time() - start_time
            if self._generation_cancelled:
                await msg.update_content(f"● {self.SPINNER_FRAMES[i]} cancelling...")
                status.update(self._status_text("cancelling..."))
            else:
                await msg.update_content(f"● {self.SPINNER_FRAMES[i]} {label}...")
                status.update(self._status_text(f"{label}... {elapsed:.1f}s"))
            self.query_one("#chat", ChatContainer).scroll_end(animate=False)
            i = (i + 1) % len(self.SPINNER_FRAMES)
            await asyncio.sleep(0.1)
