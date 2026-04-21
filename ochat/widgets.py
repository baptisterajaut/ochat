"""Chat widgets for the TUI."""

import asyncio

from markdown_it import MarkdownIt

from textual import events
from textual.app import ComposeResult
from textual.await_complete import AwaitComplete
from textual.containers import Container, ScrollableContainer
from textual.message import Message as TextualMessage
from textual.suggester import Suggester
from textual.timer import Timer
from textual.widgets import Markdown, Static
from textual.widgets._markdown import MarkdownBlock, MarkdownHeader, MarkdownStream

SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


class StreamingMarkdown(Markdown):
    """Markdown subclass whose `append()` parses in an executor.

    Textual's stock `Markdown.append()` runs `parser.parse(updated_source)`
    synchronously on the event loop (`_markdown.py:1413`). When the trailing
    block is a long monolithic paragraph (typical for LLM reasoning output
    without blank lines), each append freezes the UI until the parse returns.

    This override mirrors what `Markdown.update()` already does at
    `_markdown.py:1348` — push `parser.parse` to a thread so the event loop
    stays responsive. Everything else (block diffing, token-reuse at the
    block level, mount) stays on the loop.
    """

    def append(self, markdown: str) -> AwaitComplete:
        parser = (
            MarkdownIt("gfm-like")
            if self._parser_factory is None
            else self._parser_factory()
        )

        self._markdown = self.source + markdown
        updated_source = "".join(
            self._markdown.splitlines(keepends=True)[self._last_parsed_line:]
        )

        async def await_append() -> None:
            async with self.lock:
                tokens = await asyncio.get_running_loop().run_in_executor(
                    None, parser.parse, updated_source
                )
                existing_blocks = [
                    child for child in self.children if isinstance(child, MarkdownBlock)
                ]
                start_line = self._last_parsed_line
                for token in reversed(tokens):
                    if token.map is not None and token.level == 0:
                        self._last_parsed_line += token.map[0]
                        break

                new_blocks = list(self._parse_markdown(tokens))
                any_headers = any(
                    isinstance(block, MarkdownHeader) for block in new_blocks
                )
                for block in new_blocks:
                    start, end = block.source_range
                    block.source_range = (start + start_line, end + start_line)

                with self.app.batch_update():
                    if existing_blocks and new_blocks:
                        last_block = existing_blocks[-1]
                        last_block.source_range = new_blocks[0].source_range
                        try:
                            await last_block._update_from_block(new_blocks[0])  # noqa: SLF001
                        except IndexError:
                            pass
                        else:
                            new_blocks = new_blocks[1:]

                    if new_blocks:
                        await self.mount_all(new_blocks)

                if any_headers:
                    self._table_of_contents = None
                    self.post_message(
                        Markdown.TableOfContentsUpdated(
                            self, self.table_of_contents
                        ).set_sender(self)
                    )

        return AwaitComplete(await_append())


class CommandSuggester(Suggester):
    """Suggest slash commands."""

    COMMANDS = [
        "/help", "/h",
        "/clear", "/c",
        "/retry", "/r",
        "/copy",
        "/context", "/ctx",
        "/prompt",
        "/sys", "/system",
        "/model", "/m",
        "/personality", "/p",
        "/project",
        "/config",
        "/impersonate", "/imp",
        "/imps",
        "/suggest",
        "/stats", "/st",
        "/compact",
        "/thinking",
    ]

    async def get_suggestion(self, value: str) -> str | None:
        if not value.startswith("/"):
            return None
        value_lower = value.lower()
        for cmd in self.COMMANDS:
            if cmd.startswith(value_lower) and cmd != value_lower:
                return cmd
        return None


class ReasoningBlock(Container):
    """Collapsible reasoning area. Click to toggle expanded/collapsed.

    Expanded: renders the reasoning as Markdown (incremental stream).
    Collapsed: shows a one-line placeholder. While streaming + collapsed
    the placeholder runs an animated spinner; the parent Message body
    (i.e. the actual response) is unaffected and keeps streaming live.
    """

    class CollapseChanged(TextualMessage):
        """Posted when the user clicks to toggle collapse.

        Used by the app to persist the user's preference so new
        assistant messages start in the same state.
        """
        def __init__(self, collapsed: bool) -> None:
            self.collapsed = collapsed
            super().__init__()

    def __init__(self, initial_collapsed: bool = False) -> None:
        super().__init__(classes="reasoning-block")
        self._body: StreamingMarkdown | None = None
        self._placeholder: Static | None = None
        self._stream: MarkdownStream | None = None
        self._collapsed = initial_collapsed
        self._streaming = True  # placeholder shows a live cue while streaming
        self._hovered = False
        self._spinner_timer: Timer | None = None
        self._spinner_frame = 0
        self.display = False  # hidden until first reasoning token arrives

    def compose(self) -> ComposeResult:
        self._placeholder = Static("› thinking… (click to expand)",
                                   classes="reasoning-placeholder")
        self._placeholder.display = self._collapsed
        yield self._placeholder
        self._body = StreamingMarkdown("", classes="reasoning-body")
        self._body.display = not self._collapsed
        yield self._body

    def start_stream(self) -> None:
        if self._body is not None and self._stream is None:
            self._stream = Markdown.get_stream(self._body)
        # Re-arm streaming state — `Message.update()` may have flipped
        # it to False earlier (e.g. _animate_spinner's pre-stream calls
        # pass through `update()` which calls `stop_stream()` on us).
        # Without this, _sync_spinner would see streaming=False and skip
        # the spinner on the first actual reasoning token.
        self._streaming = True
        self._sync_spinner()
        self._refresh_placeholder()
        self.display = True

    async def append(self, delta: str) -> None:
        if self._stream is None or not delta:
            return
        await self._stream.write(delta)

    async def stop_stream(self) -> None:
        if self._stream is not None:
            await self._stream.stop()
            self._stream = None
        self._streaming = False
        self._sync_spinner()
        self._refresh_placeholder()

    def mark_idle(self) -> None:
        """Signal that reasoning is done (content has started).

        Stops the spinner and flips the placeholder to the static
        "done" label. The underlying MarkdownStream stays open — late
        reasoning tokens (rare) are still accepted; we just don't
        advertise in-progress work anymore.
        """
        if not self._streaming:
            return
        self._streaming = False
        self._sync_spinner()
        self._refresh_placeholder()

    def _refresh_placeholder(self) -> None:
        if self._placeholder is None:
            return
        if self._collapsed and self._streaming:
            # Spinner drives the label while streaming + collapsed.
            frame = SPINNER_FRAMES[self._spinner_frame]
            self._placeholder.update(f"{frame} thinking...")
            return
        label = ("› thinking… (click to expand)" if self._streaming
                 else "✓ thinking done (click to expand)")
        self._placeholder.update(label)

    def _sync_spinner(self) -> None:
        """Start/stop the spinner timer based on (collapsed & streaming)."""
        should_run = self._collapsed and self._streaming
        if should_run and self._spinner_timer is None:
            self._spinner_timer = self.set_interval(0.1, self._tick_spinner)
        elif not should_run and self._spinner_timer is not None:
            self._spinner_timer.stop()
            self._spinner_timer = None

    def _tick_spinner(self) -> None:
        if self._placeholder is None or not (self._collapsed and self._streaming):
            return
        self._spinner_frame = (self._spinner_frame + 1) % len(SPINNER_FRAMES)
        frame = SPINNER_FRAMES[self._spinner_frame]
        self._placeholder.update(f"{frame} thinking...")

    def on_click(self) -> None:
        self._collapsed = not self._collapsed
        if self._body is not None:
            self._body.display = not self._collapsed
        if self._placeholder is not None:
            self._placeholder.display = self._collapsed
        self._sync_spinner()
        self._refresh_placeholder()
        self.post_message(self.CollapseChanged(self._collapsed))

    # Hover feedback via inline styles (not CSS :hover) to avoid
    # update_node_styles() walking the entire streamed-reasoning subtree
    # on every mouse move. Enter/Leave bubble up from descendants, so we
    # track whether the pointer is still anywhere inside our region.
    #
    # Caveat: each descendant caches its composited foreground /
    # background in `_visual_style_cache` (widget.py:1202). Setting our
    # inline background does NOT invalidate those caches, so text cells
    # would keep the pre-hover background while empty cells pick up the
    # new tint. We manually clear the descendant caches — cheap
    # (dict.clear() per node) vs. the full stylesheet re-apply that the
    # CSS :hover cascade was doing.
    def _invalidate_descendant_style_caches(self) -> None:
        for descendant in self.walk_children(with_self=False):
            descendant.notify_style_update()
            descendant.refresh()

    def on_enter(self, _event: events.Enter) -> None:
        if self._hovered:
            return
        self._hovered = True
        boost = self.app.theme_variables.get("boost", "white 4%")
        self.styles.background = boost
        self.styles.border_left = ("outer", "#888")
        self._invalidate_descendant_style_caches()

    def on_leave(self, _event: events.Leave) -> None:
        new_over = self.app.mouse_over
        if new_over is not None and self in new_over.ancestors_with_self:
            return
        if not self._hovered:
            return
        self._hovered = False
        self.styles.background = None
        self.styles.border_left = None
        self._invalidate_descendant_style_caches()


class Message(Container):
    """A single chat message.

    Assistant messages get a ReasoningBlock (collapsible Markdown) above the
    main body. Both bodies use MarkdownStream for incremental parsing: only
    the trailing block is re-rendered per batch, so streaming cost is
    O(tail) instead of O(buffer).
    """

    def __init__(self, content: str, role: str = "user",
                 reasoning_collapsed: bool = False) -> None:
        super().__init__(classes=f"message {role}")
        self.role = role
        self.reasoning = ""
        self._initial_content = content.strip()
        self._reasoning_collapsed = reasoning_collapsed
        self._body: StreamingMarkdown | None = None
        self._reasoning_block: ReasoningBlock | None = None
        self._stream: MarkdownStream | None = None

    def compose(self) -> ComposeResult:
        if self.role == "assistant":
            self._reasoning_block = ReasoningBlock(
                initial_collapsed=self._reasoning_collapsed,
            )
            yield self._reasoning_block
        initial = self._initial_content
        if self.role == "assistant":
            initial = f"● {initial}"
        elif self.role == "user":
            initial = f"› {initial}"
        self._body = StreamingMarkdown(initial, classes="message-body")
        yield self._body

    async def update(self, content: str, *, reasoning: str = "") -> None:
        """Full replace — stops any active stream. Used for finalize/cancel/error."""
        await self._stop_stream()
        if self._reasoning_block is not None:
            await self._reasoning_block.stop_stream()
        await self.update_reasoning(reasoning)
        if self._body is not None:
            await self._body.update(content)

    async def update_content(self, content: str) -> None:
        """Full content replace — stops any active content stream."""
        await self._stop_stream()
        if self._body is not None:
            await self._body.update(content)

    def start_content_stream(self) -> None:
        if self._body is not None and self._stream is None:
            self._stream = Markdown.get_stream(self._body)

    async def append_content(self, delta: str) -> None:
        if self._stream is None or not delta:
            return
        await self._stream.write(delta)

    async def _stop_stream(self) -> None:
        if self._stream is not None:
            await self._stream.stop()
            self._stream = None

    def start_reasoning_stream(self) -> None:
        if self._reasoning_block is not None:
            self._reasoning_block.start_stream()

    async def append_reasoning(self, delta: str) -> None:
        self.reasoning += delta
        if self._reasoning_block is not None:
            await self._reasoning_block.append(delta)

    async def stop_reasoning_stream(self) -> None:
        if self._reasoning_block is not None:
            await self._reasoning_block.stop_stream()

    def mark_reasoning_idle(self) -> None:
        """Signal reasoning phase is done — called when the first
        content token arrives. Stops the spinner, flips the placeholder
        to the static "thinking done" label. No-op if reasoning never
        started or already marked idle.
        """
        if self._reasoning_block is not None:
            self._reasoning_block.mark_idle()

    async def update_reasoning(self, reasoning: str) -> None:
        """Full replace (used on non-streaming finalize/cancel).

        `Markdown.update()` returns an AwaitComplete whose `__init__`
        schedules the coroutine via `asyncio.gather()`. If we don't
        await it, the Task runs detached; if it later gets cancelled
        (app shutdown, lock contention) the exception is never
        retrieved → `CancelledError was never retrieved`. Await is not
        optional.
        """
        self.reasoning = reasoning
        if self._reasoning_block is None:
            return
        if reasoning:
            self._reasoning_block.display = True
            if self._reasoning_block._body is not None:
                await self._reasoning_block._body.update(reasoning)
        else:
            self._reasoning_block.display = False


class ChatContainer(ScrollableContainer):
    """Scrollable container for chat messages."""
