"""Generation mixin: streaming, API calls, response generation."""

import asyncio
import logging
import time

from textual.widgets import Input, Static

from ochat.widgets import ChatContainer, Message

_log = logging.getLogger(__name__)

# Sentinel for async stream iteration
_STREAM_DONE = object()


def _clean_impersonate_response(response: str) -> str:
    """Strip quotes and collapse whitespace from impersonate results."""
    response = response.strip()
    if response.startswith('"') and response.endswith('"'):
        response = response[1:-1]
    return " ".join(response.split())


class GenerationMixin:
    """Mixin providing LLM generation capabilities for OChat."""

    def _chat_call(self, messages: list[dict], stream: bool):
        """Make API call, return stream iterator or result."""
        if self.backend_type == "auto":
            backend = self.backend._detected_backend or self.backend._ollama
            return backend.chat(self.model, messages, stream, self.num_ctx, self.model_options)
        return self.backend.chat(self.model, messages, stream, self.num_ctx, self.model_options)

    def _extract_chunk(self, chunk) -> str:
        """Extract text content from a streaming chunk."""
        return self.backend.extract_chunk(chunk)

    def _extract_result(self, result) -> tuple[str, int]:
        """Extract (content, token_count) from a non-streaming result."""
        return self.backend.extract_result(result)

    async def _generate_response(self) -> None:
        """Generate assistant response (streaming or not)."""
        self.is_generating = True
        self._generation_cancelled = False
        chat = self.query_one("#chat", ChatContainer)
        status = self.query_one("#status", Static)

        assistant_msg = Message("...", "assistant")
        await chat.mount(assistant_msg)
        chat.scroll_end(animate=False)
        await asyncio.sleep(0)  # Let UI refresh

        response_text = ""
        start_time = time.time()
        tokens_generated = 0
        cancelled = False

        try:
            stream, first_chunk = await self._start_stream(
                self.messages, assistant_msg, status, start_time
            )
            self.last_ttft = time.time() - start_time if not self._generation_cancelled else 0.0

            if first_chunk and not self._generation_cancelled:
                content = self._extract_chunk(first_chunk)
                if content:
                    response_text += content
                    tokens_generated += 1

                response_text, tokens_generated, cancelled = await self._consume_chunks(
                    stream, assistant_msg, chat, status, start_time,
                    response_text, tokens_generated,
                )

                # Non-streaming: show buffered response after completion
                if not self.streaming and not cancelled:
                    think_time = time.time() - start_time
                    await assistant_msg.update(
                        f"● *thought for {think_time:.1f}s*\n\n{response_text}"
                    )
                    chat.scroll_end(animate=False)

            if cancelled:
                await assistant_msg.update("● *[cancelled]*")
                if response_text:
                    self.messages.append({"role": "assistant", "content": response_text})
            else:
                self.messages.append({"role": "assistant", "content": response_text})
                self.last_gen_time = time.time() - start_time
                self.last_tokens = tokens_generated
                self.total_tokens += tokens_generated

        except Exception as e:
            _log.exception("Generation error")
            response_text = f"**Error:** {e}"
            await assistant_msg.update(f"● {response_text}")

        finally:
            self.is_generating = False
            status.update(self._status_text())

        # Auto-suggest after successful generation
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

    async def _consume_chunks(self, stream, assistant_msg, chat, status,
                              start_time, response_text, tokens_generated):
        """Consume stream chunks, updating UI. Returns (text, tokens, cancelled)."""
        while True:
            chunk = await self._anext(stream)
            if chunk is _STREAM_DONE or self._generation_cancelled:
                return response_text, tokens_generated, self._generation_cancelled

            content = self._extract_chunk(chunk)
            if content:
                response_text += content
                tokens_generated += 1

            elapsed = time.time() - start_time

            if self.streaming:
                if content:
                    await assistant_msg.update(f"● {response_text}")
                    chat.scroll_end(animate=False)
                tps = tokens_generated / elapsed if elapsed > 0 else 0
                status.update(self._status_text(
                    f"generating... {elapsed:.1f}s ({tokens_generated}tok, {tps:.1f}t/s)"
                ))
            else:
                frame = self.SPINNER_FRAMES[int(elapsed * 10) % len(self.SPINNER_FRAMES)]
                await assistant_msg.update(
                    f"● {frame} thinking... {elapsed:.1f}s ({tokens_generated} chunks)"
                )
                chat.scroll_end(animate=False)
                status.update(self._status_text(
                    f"thinking... {elapsed:.1f}s ({tokens_generated} chunks)"
                ))

    async def _start_stream(self, messages: list[dict], msg: Message, status: Static, start_time: float):
        """Start a streaming API call with a spinner while waiting for the first token.

        Returns (stream_iterator, first_chunk) — first_chunk may be None.
        """
        spinner = asyncio.create_task(
            self._animate_spinner(msg, status, start_time, "waiting for first token")
        )
        try:
            stream = await asyncio.to_thread(
                lambda: self._chat_call(messages, stream=True)
            )
            first_chunk = await asyncio.to_thread(lambda: next(iter(stream), None))
        finally:
            spinner.cancel()
            try:
                await spinner
            except asyncio.CancelledError:
                pass
        return stream, first_chunk

    async def _anext(self, iterator):
        """Get next item from sync iterator without blocking event loop."""
        return await asyncio.to_thread(lambda: next(iterator, _STREAM_DONE))

    async def _run_auto_suggest(self) -> None:
        """Generate a short suggestion for the user's next message (background)."""
        try:
            suggest_messages = self.messages.copy()
            suggest_messages.append({
                "role": "system",
                "content": self.sys_instructions["impersonate_short"],
            })
            result = await asyncio.to_thread(
                lambda: self._chat_call(suggest_messages, stream=False)
            )
            response, _ = self._extract_result(result)
            response = _clean_impersonate_response(response)

            input_widget = self.query_one("#chat-input", Input)
            if not input_widget.value and not self.is_generating:
                self._pending_suggestion = response
                input_widget.placeholder = response
                _log.debug("Auto-suggest set: %s", response[:80])
        except Exception:  # CancelledError is BaseException in 3.9+, propagates through
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
