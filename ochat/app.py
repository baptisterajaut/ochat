"""Main application module: OllamaChat App and main() entry point."""

import argparse
import asyncio
from contextlib import asynccontextmanager
import json
import logging
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import ollama
import openai

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.events import Click
from textual.widgets import Footer, Input, Static

from ochat.backend import OllamaBackend, OpenAIBackend, LlamaCppBackend, AutoBackend
from ochat.commands import CommandsMixin
from ochat.config import (
    CONFIG_DIR,
    CONFIG_FILE,
    list_configs,
    load_config,
    load_project_prompt,
    load_system_prompt,
    run_setup,
    switch_config_to_default,
    update_config,
)
from ochat.generation import GenerationMixin
from ochat.widgets import ChatContainer, CommandSuggester, Message

_log = logging.getLogger(__name__)


def _cleanup_old_logs():
    """Remove log files older than 7 days."""
    log_dir = Path(tempfile.gettempdir())
    cutoff = time.time() - 7 * 86400
    for f in log_dir.glob("ollama-chat-*.log"):
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
        except OSError:
            pass


class OChat(CommandsMixin, GenerationMixin, App):
    """ochat - Simple TUI chat for any LLM backend."""

    CSS_PATH = "chat.tcss"
    SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    DEFAULT_PLACEHOLDER = "Message... (/help for commands)"

    BINDINGS = [
        Binding("ctrl+c", "clear_input", "Clear input", show=False),
        Binding("ctrl+d", "quit", "Quit"),
        Binding("ctrl+l", "clear", "Clear"),
        Binding("ctrl+o", "toggle_streaming", "Stream"),
        Binding("escape", "cancel_or_quit", "Cancel/Quit"),
        Binding("tab", "focus_input", show=False, priority=True),
        Binding("shift+tab", "focus_input", show=False, priority=True),
    ]

    def __init__(
        self,
        model: str = "llama3.2",
        system_prompt: str | None = None,
        num_ctx: int = 4096,
        personality_name: str | None = None,
        append_local_prompt: bool = True,
        streaming: bool = True,
        model_options: dict | None = None,
        config_name: str = "",
        host: str | None = None,
        verify_ssl: bool = True,
        auto_suggest: bool = True,
        backend_type: str = "ollama",
    ) -> None:
        super().__init__()
        self.model = model
        self.system_prompt = system_prompt
        self.num_ctx = num_ctx
        self.personality_name = personality_name
        self.append_local_prompt = append_local_prompt
        self.model_options = model_options or {}
        self.config_name = config_name
        self.host = host or "http://localhost:11434"
        self.verify_ssl = verify_ssl
        self.backend_type = backend_type
        self.backend = self._create_backend(backend_type, host or "http://localhost:11434", verify_ssl)
        self.api_mode = backend_type  # "ollama", "openai", "llama_cpp", "auto"
        self.messages: list[dict] = []
        self.is_generating = False
        self.streaming = streaming
        self.auto_suggest = auto_suggest
        self._auto_suggest_task: asyncio.Task | None = None
        self._pending_suggestion: str = ""
        self._generation_cancelled = False
        self.total_tokens = 0
        self.last_gen_time = 0.0
        self.last_tokens = 0
        self.last_ttft = 0.0
        self._context_warning_shown = False
        self.sys_instructions = self._load_system_instructions()
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

       @property
    def openai_client(self):
        """Legacy property for backwards compatibility."""
        if self.backend_type == "openai":
            return self.backend._client
        if self.backend_type == "llama_cpp":
            return self.backend._client
        raise AttributeError("openai_client only available for openai/llama_cpp backends")

    def _create_backend(self, backend_type: str, host: str, verify_ssl: bool):
        """Create backend instance based on type."""
        match backend_type:
            case "ollama":
                return OllamaBackend(host=host, verify_ssl=verify_ssl)
            case "openai":
                return OpenAIBackend(host=host, verify_ssl=verify_ssl)
            case "llama_cpp":
                return LlamaCppBackend(host=host, verify_ssl=verify_ssl)
            case "auto":
                return AutoBackend(host=host, verify_ssl=verify_ssl)
            case _:
                raise ValueError(f"Unknown backend type: {backend_type}")

    @staticmethod
    def _load_system_instructions() -> dict:
        """Load system_instructions.json. Fatal error if missing or invalid."""
        path = Path(__file__).parent.parent / "system_instructions.json"
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: {path} not found. Please restore the file.")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: {path} is not valid JSON: {e}")
            sys.exit(1)

    def compose(self) -> ComposeResult:
        yield ChatContainer(id="chat")
        yield Vertical(
            Input(
                placeholder=self.DEFAULT_PLACEHOLDER,
                id="chat-input",
                suggester=CommandSuggester(use_cache=False),
            ),
            id="input-container",
        )
        yield Static(self._status_text(), id="status")
        yield Footer()

    def _estimate_context_tokens(self) -> int:
        """Rough token estimate for all messages (~4 chars/token)."""
        return sum(len(m["content"]) // 4 for m in self.messages)

    def _reset_stats(self) -> None:
        """Reset generation stats and context warning."""
        self.total_tokens = 0
        self.last_gen_time = 0.0
        self.last_tokens = 0
        self.last_ttft = 0.0
        self._context_warning_shown = False

    @asynccontextmanager
    async def _generating_lock(self):
        """Context manager: set is_generating, disable input, restore on exit."""
        self.is_generating = True
        input_widget = self.query_one("#chat-input", Input)
        input_widget.disabled = True
        try:
            yield
        finally:
            input_widget.disabled = False
            self.is_generating = False
            self.query_one("#status", Static).update(self._status_text())
            input_widget.focus()

    def _context_pct(self) -> float:
        """Estimated context usage as percentage."""
        if self.num_ctx <= 0 or self.api_mode == "openai":
            return 0.0
        return self._estimate_context_tokens() / self.num_ctx * 100

    def _context_info(self) -> str:
        msg_count = len([m for m in self.messages if m["role"] != "system"])
        estimated = self._estimate_context_tokens()
        pct = self._context_pct()
        return f"Messages: {msg_count} | Tokens used: ~{estimated} ({pct:.0f}%) | Context size: {self.num_ctx}"

    def _status_text(self, extra: str = "") -> str:
       if self.backend_type == "auto":
            base = f"{self.model} (auto)"
        elif self.backend_type == "openai":
            base = f"{self.model} (openai)"
        elif self.backend_type == "llama_cpp":
            base = f"{self.model} (llama.cpp)"
        else:
            base = f"{self.model} | ctx:{self.num_ctx}"
        if self.last_gen_time > 0:
            tps = self.last_tokens / self.last_gen_time if self.last_gen_time else 0
            ttft_str = f", ttft:{self.last_ttft:.2f}s" if self.last_ttft > 0 else ""
            base += f" | last: {self.last_gen_time:.1f}s ({self.last_tokens}tok, {tps:.1f}t/s{ttft_str})"
        if self.total_tokens > 0:
            base += f" | total: {self.total_tokens}tok"
        # Context warnings
        pct = self._context_pct()
        if pct > 100:
            base += " | ⚠ Context length probably exceeded"
        elif pct > 85:
            base += f" | ctx: {100 - pct:.0f}% remaining"
        if extra:
            base += f" | {extra}"
        return base

    async def on_mount(self) -> None:
        _log.info("App mounted, model=%s, config=%s", self.model, self.config_name)
        self.query_one("#chat-input", Input).focus()
        await self._show_greeting()

    def on_click(self, _event: Click) -> None:
        """Keep focus on input when clicking anywhere."""
        self.query_one("#chat-input", Input).focus()

async def _show_greeting(self) -> None:
        """Show greeting with ASCII art after validating connection."""
        chat = self.query_one("#chat", ChatContainer)
        config_line = f"config: {self.config_name} · " if self.config_name else ""

        # Try Ollama first, then OpenAI fallback (auto mode)
        mode_label = ""
        if self.backend_type == "auto":
            try:
                await asyncio.to_thread(self.backend._ollama.list_models)
                self.api_mode = "ollama"
                _log.info("Connected in Ollama mode")
                mode_label = "Connected"
            except Exception as e:
                _log.info("Ollama list failed (%s), trying OpenAI fallback", e)
                try:
                    await asyncio.to_thread(self.backend._openai.list_models)
                    self.api_mode = "openai"
                    _log.info("Connected in OpenAI-compatible mode")
                    mode_label = "Connected (OpenAI mode)"
                except Exception as e2:
                    _log.warning("OpenAI fallback failed: %s", e2)
                    await self._show_system_message("Warning: Cannot connect to server")
                    return
        else:
            try:
                await asyncio.to_thread(self.backend.list_models)
                self.api_mode = self.backend_type
                _log.info("Connected in %s mode", self.backend_type)
                mode_label = "Connected"

        logo = f"""\
  ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
   ___   ___| |__   __ _| |_
  / _ \\ / __| '_ \\ / _` | __|
| (_) | (__| | | | (_| | |_
 \\___/ \\___|_| |_|\\__,_|\\__|

{config_line}{mode_label} · /help for commands"""
        msg = Static(logo, classes="greeting")
        await chat.mount(msg)
        chat.scroll_end(animate=False)

        # Auto-suggest first message if project context is available
        if self.auto_suggest and self.append_local_prompt and load_project_prompt():
            self._auto_suggest_task = asyncio.create_task(self._run_auto_suggest())

    async def _show_system_message(self, text: str) -> None:
        """Show a system info message in the chat."""
        chat = self.query_one("#chat", ChatContainer)
        msg = Message(text, "system-info")
        await chat.mount(msg)
        chat.scroll_end(animate=False)

    def _clear_suggestion(self, input_widget: Input | None = None) -> None:
        """Clear pending auto-suggestion and restore placeholder."""
        if self._pending_suggestion:
            self._pending_suggestion = ""
            if input_widget is None:
                input_widget = self.query_one("#chat-input", Input)
            input_widget.placeholder = self.DEFAULT_PLACEHOLDER

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if not event.value.strip() or self.is_generating:
            return

        # Cancel any pending auto-suggest
        if self._auto_suggest_task and not self._auto_suggest_task.done():
            self._auto_suggest_task.cancel()
            self._auto_suggest_task = None
        self._clear_suggestion(event.input)

        user_input = event.value.strip()
        event.input.value = ""

        if user_input.startswith("/"):
            handled = await self._handle_command(user_input)
            if handled:
                return

        self.messages.append({"role": "user", "content": user_input})
        chat = self.query_one("#chat", ChatContainer)
        await chat.mount(Message(user_input, "user"))
        chat.scroll_end(animate=False)

        # Let UI refresh before blocking on generation
        await asyncio.sleep(0)

        await self._generate_response()

    def action_clear_input(self) -> None:
        """Clear the input field."""
        self.query_one("#chat-input", Input).value = ""

    def action_focus_input(self) -> None:
        """Keep focus on input and accept suggestion if any."""
        input_widget = self.query_one("#chat-input", Input)
        # Accept pending auto-suggestion on Tab when input is empty
        if not input_widget.value and self._pending_suggestion:
            input_widget.value = self._pending_suggestion
            input_widget.cursor_position = len(self._pending_suggestion)
            self._clear_suggestion(input_widget)
            return
        if input_widget.cursor_at_end:
            input_widget.action_cursor_right()
        input_widget.focus()

    def action_cancel_or_quit(self) -> None:
        """Cancel generation if running, otherwise quit."""
        if self.is_generating:
            self._generation_cancelled = True
            self.notify("Cancelled", timeout=2)
        else:
            self.exit()

    def action_toggle_streaming(self) -> None:
        """Toggle streaming mode."""
        self.streaming = not self.streaming
        update_config(streaming=self.streaming)
        mode = "ON" if self.streaming else "OFF"
        self.notify(f"Streaming: {mode}", timeout=2)

    def action_clear(self) -> None:
        """Clear chat history."""
        if self._auto_suggest_task and not self._auto_suggest_task.done():
            self._auto_suggest_task.cancel()
            self._auto_suggest_task = None
        self._clear_suggestion()
        chat = self.query_one("#chat", ChatContainer)
        chat.remove_children()
        self.messages = []
        self._reset_stats()
        if self.system_prompt:
            self.messages.append({"role": "system", "content": self.system_prompt})
        self.query_one("#status", Static).update(self._status_text())


def main():
    # First run: launch setup wizard
    if not CONFIG_FILE.exists():
        run_setup()

    config = load_config()

    parser = argparse.ArgumentParser(description="ochat - Simple TUI chat for any LLM backend")
    parser.add_argument(
        "-C", "--config",
        action="store_true",
        help="Run interactive configuration setup"
    )
    parser.add_argument(
        "-m", "--model",
        default=None,
        help=f"Ollama model to use (default from config: {config['model']})"
    )
    parser.add_argument(
        "-s", "--system",
        help="Path to system prompt file"
    )
    parser.add_argument(
        "--system-prompt",
        help="System prompt string (overrides -s)"
    )
    parser.add_argument(
        "-c", "--num-ctx",
        type=int,
        default=None,
        help=f"Context window size (default from config: {config['num_ctx']})"
    )
    parser.add_argument(
        "--use-config",
        metavar="NAME",
        help="Use a named config file (without .conf extension)"
    )
    parser.add_argument(
        "--as-default",
        action="store_true",
        help="With --use-config: make it the new default config"
    )
    parser.add_argument(
        "--new",
        action="store_true",
        help="Create a new named config profile"
    )
    parser.add_argument(
        "-d", "--debug",
        action="store_true",
        help="Enable debug logging to temp file"
    )

    args = parser.parse_args()

    if args.debug:
        log_file = Path(tempfile.gettempdir()) / f"ollama-chat-{datetime.now():%Y%m%d-%H%M%S}.log"
        handler = logging.FileHandler(str(log_file))
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logger = logging.getLogger("ochat")
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        _log.info("Debug logging enabled, log file: %s", log_file)
        _cleanup_old_logs()

    if args.new:
        run_setup(create_new=True)
        return

    if args.config:
        run_setup()
        return

    # Handle --use-config
    if args.use_config:
        config_file = CONFIG_DIR / f"{args.use_config}.conf"
        if not config_file.exists():
            print(f"Error: Config '{args.use_config}' not found")
            available = list_configs()
            if available:
                print(f"Available configs: {', '.join(available)}")
            sys.exit(1)

        if args.as_default:
            success, msg = switch_config_to_default(args.use_config)
            if not success:
                print(f"Error: {msg}")
                sys.exit(1)
            # Reload config from the new default
            config = load_config()
        else:
            # Just load the specified config for this session
            config = load_config(config_file)

    model = args.model if args.model else config["model"]
    num_ctx = args.num_ctx if args.num_ctx else config["num_ctx"]

    host = config["host"]

    if args.system_prompt:
        system_prompt = args.system_prompt
        personality_name = None
    else:
        system_prompt, personality_name = load_system_prompt(
            args.system, config["personality"], config["append_local_prompt"]
        )

    app = OChat(
        model=model,
        system_prompt=system_prompt,
        num_ctx=num_ctx,
        personality_name=personality_name,
        append_local_prompt=config["append_local_prompt"],
        streaming=config["streaming"],
        model_options=config["model_options"],
        config_name=config["config_name"],
        host=host,
        verify_ssl=config["verify_ssl"],
        auto_suggest=config["auto_suggest"],
        backend_type=config["backend"],
    )
    try:
        app.run()
    except Exception:
        _log.exception("Unhandled exception in app.run()")
        raise
    finally:
        _log.info("App exited")
