"""Commands mixin: slash command handling."""

import asyncio
import logging
import os
import shutil
import subprocess
import sys
import time

from textual.widgets import Input, Static

from ochat.config import (
    list_configs,
    list_personalities,
    load_personality,
    load_project_prompt,
    load_system_prompt,
    switch_config_to_default,
    update_config,
)
from ochat.generation import _STREAM_DONE, _clean_impersonate_response
from ochat.widgets import ChatContainer, Message

_log = logging.getLogger(__name__)


class CommandsMixin:
    """Mixin providing slash command handling for OllamaChat."""

    async def _handle_command(self, raw_cmd: str) -> bool:
        """Handle slash commands. Returns True if command was handled."""
        _log.debug("Command: %s", raw_cmd)
        raw_cmd = raw_cmd.strip()
        parts = raw_cmd.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd in ("/help", "/h", "/?"):
            help_text = """**Commands:**
- `/retry` or `/r` - Regenerate last response
- `/copy` - Copy last response to clipboard
- `/clear` or `/c` - Clear chat history
- `/context` or `/ctx` - Show current context info
- `/prompt` - Show system prompt
- `/sys <msg>` or `/system <msg>` - Inject a system message
- `/model` or `/m` - Show current model
- `/personality` or `/p` - List/change personality
- `/config` - List/switch config profiles
- `/impersonate` or `/imp` - Generate user response suggestion
- `/imps` - Short impersonate (suggestion-length)
- `/suggest` - Toggle auto-suggest after responses
- `/project` - Toggle local prompt append (agent.md)
- `/stats` or `/st` - Show generation statistics
- `/compact` - Summarize conversation to free context
- `/help` or `/h` - Show this help"""
            await self._show_system_message(help_text)
            return True

        elif cmd in ("/retry", "/r"):
            await self._handle_retry()
            return True

        elif cmd == "/copy":
            await self._handle_copy()
            return True

        elif cmd in ("/clear", "/c"):
            self.action_clear()
            await self._show_system_message("Chat cleared.")
            return True

        elif cmd in ("/context", "/ctx"):
            await self._show_system_message(self._context_info())
            return True

        elif cmd == "/prompt":
            if self.system_prompt:
                await self._show_system_message(f"**System prompt:**\n{self.system_prompt}")
            else:
                await self._show_system_message("No system prompt set.")
            return True

        elif cmd in ("/sys", "/system"):
            if arg:
                self.messages.append({"role": "system", "content": arg})
                await self._show_system_message(f"*[injected]* {arg}")
            else:
                await self._show_system_message("Usage: `/sys <message>`")
            return True

        elif cmd in ("/model", "/m"):
            await self._show_system_message(f"Model: `{self.model}`")
            return True

        elif cmd in ("/personality", "/p"):
            await self._handle_personality_command(arg)
            return True

        elif cmd == "/project":
            await self._handle_project_toggle()
            return True

        elif cmd in ("/config", "/cfg"):
            await self._handle_config_command(arg)
            return True

        elif cmd in ("/impersonate", "/imp"):
            await self._handle_impersonate()
            return True

        elif cmd == "/imps":
            await self._handle_impersonate_short()
            return True

        elif cmd == "/suggest":
            await self._handle_suggest_toggle()
            return True

        elif cmd in ("/stats", "/st"):
            await self._handle_stats()
            return True

        elif cmd == "/compact":
            await self._handle_compact()
            return True

        return False

    async def _handle_retry(self) -> None:
        """Regenerate the last assistant response."""
        if self.is_generating:
            return

        # Find and remove last assistant message
        if not self.messages or self.messages[-1]["role"] != "assistant":
            await self._show_system_message("Nothing to regenerate")
            return

        self.messages.pop()

        # Remove last assistant message widget from chat
        chat = self.query_one("#chat", ChatContainer)
        for child in reversed(list(chat.children)):
            if isinstance(child, Message) and child.role == "assistant":
                await child.remove()
                break

        # Regenerate
        await self._generate_response()

    async def _handle_copy(self) -> None:
        """Copy last assistant response to clipboard."""
        for msg in reversed(self.messages):
            if msg["role"] == "assistant":
                self._copy_text(msg["content"])
                return
        await self._show_system_message("Nothing to copy")

    def _copy_text(self, text: str) -> None:
        """Copy text to clipboard via system tool, with OSC 52 as fallback."""
        # Try system clipboard tools in order of preference
        for cmd in ("wl-copy", "xclip", "xsel", "pbcopy", "clip.exe"):
            path = shutil.which(cmd)
            if not path:
                continue
            args = [path]
            if cmd == "xclip":
                args += ["-selection", "clipboard"]
            elif cmd == "xsel":
                args += ["--clipboard", "--input"]
            try:
                subprocess.run(
                    args, input=text.encode(), check=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                self.notify("Copied to clipboard")
                return
            except (subprocess.CalledProcessError, OSError):
                continue
        # No system tool available — fall back to OSC 52 (terminal support varies)
        self.copy_to_clipboard(text)
        self.notify("Copied to clipboard (OSC 52 — may not work in all terminals)")

    async def _handle_project_toggle(self) -> None:
        """Toggle append_local_prompt and reload system prompt."""
        self.append_local_prompt = not self.append_local_prompt
        update_config(append_local_prompt=self.append_local_prompt)

        if self.personality_name:
            new_prompt, _ = load_system_prompt(
                None, self.personality_name, self.append_local_prompt
            )
            self.system_prompt = new_prompt

            self.messages = [m for m in self.messages if m["role"] != "system"]
            if new_prompt:
                self.messages.insert(0, {"role": "system", "content": new_prompt})

        project_content = load_project_prompt()
        status = "ON" if self.append_local_prompt else "OFF"
        if project_content and self.append_local_prompt:
            await self._show_system_message(f"Local prompt: **{status}** (appended to system prompt)")
        elif project_content:
            await self._show_system_message(f"Local prompt: **{status}** (ignored)")
        else:
            await self._show_system_message(f"Local prompt: **{status}** (no file found)")

    async def _handle_config_command(self, arg: str) -> None:
        """Handle /config command for listing and switching configs."""
        configs = list_configs()

        # Add current config.conf as option
        current_name = self.config_name or "(default)"
        all_configs = [current_name] + [c for c in configs if c != self.config_name]

        if not arg:
            # List configs
            lines = ["**Available configs:**"]
            for i, c in enumerate(all_configs, 1):
                marker = " ← current" if (i == 1) else ""
                lines.append(f"{i}. `{c}`{marker}")
            lines.append("\n*Type `/config <number>` to switch (restarts app)*")
            await self._show_system_message("\n".join(lines))
            return

        choice = arg.strip()
        try:
            idx = int(choice)
            if idx == 1:
                await self._show_system_message("Already using this config")
                return
            if 2 <= idx <= len(all_configs):
                selected_config = all_configs[idx - 1]
            else:
                await self._show_system_message(f"Invalid choice (1-{len(all_configs)})")
                return
        except ValueError:
            if choice in configs:
                selected_config = choice
            else:
                await self._show_system_message(f"Config '{choice}' not found")
                return

        # Switch config and restart
        success, msg = switch_config_to_default(selected_config, interactive=False)
        if not success:
            await self._show_system_message(f"Error: {msg}")
            return

        # Restart the app with new config
        await self._restart_app()

    async def _restart_app(self) -> None:
        """Show restart message and relaunch the app."""
        input_widget = self.query_one("#chat-input", Input)
        input_widget.value = ""
        input_widget.placeholder = "Restarting..."
        input_widget.disabled = True
        await asyncio.sleep(0.1)  # Let UI refresh
        self.exit()
        os.execv(sys.executable, [sys.executable] + sys.argv)

    async def _handle_impersonate(self) -> None:
        """Generate a response as if the user was speaking, put it in input."""
        _log.debug("Impersonate started")
        if self.is_generating:
            return

        if len([m for m in self.messages if m["role"] != "system"]) == 0:
            await self._show_system_message("Need conversation context first")
            return

        async with self._generating_lock():
            input_widget = self.query_one("#chat-input", Input)
            status = self.query_one("#status", Static)

            # Build messages with impersonate instruction
            impersonate_messages = self.messages.copy()
            impersonate_messages.append({
                "role": "system",
                "content": self.sys_instructions["impersonate"],
            })

            status.update(self._status_text("impersonating..."))
            input_widget.value = "Impersonating..."

            try:
                result = await asyncio.to_thread(
                    lambda: self._chat_call(impersonate_messages, stream=False)
                )
                response, _ = self._extract_result(result)
                response = _clean_impersonate_response(response)
                _log.debug("Impersonate result: %s...", response[:100])
                input_widget.value = response
                input_widget.cursor_position = len(response)
            except Exception as e:
                _log.exception("Impersonate error")
                input_widget.value = ""
                await self._show_system_message(f"Error: {e}")

    async def _handle_stats(self) -> None:
        """Show generation statistics."""
        mode = "streaming" if self.streaming else "non-streaming"
        lines = [f"**Stats** — `{self.model}` ({mode})"]

        if self.last_gen_time > 0:
            tps = self.last_tokens / self.last_gen_time if self.last_gen_time else 0
            lines.append("\n**Last generation:**")
            lines.append(f"- Duration: {self.last_gen_time:.2f}s")
            if self.last_ttft > 0:
                lines.append(f"- TTFT: {self.last_ttft:.2f}s")
            lines.append(f"- Tokens: {self.last_tokens}")
            lines.append(f"- Speed: {tps:.1f} t/s")
        else:
            lines.append("\nNo generation yet.")

        lines.append(f"\n{self._context_info()}")

        await self._show_system_message("\n".join(lines))

    async def _handle_compact(self) -> None:
        """Summarize conversation to free up context."""
        if self.is_generating:
            return

        conv_messages = [m for m in self.messages if m["role"] != "system"]
        if len(conv_messages) < 2:
            await self._show_system_message("Not enough conversation to compact")
            return

        async with self._generating_lock():
            chat = self.query_one("#chat", ChatContainer)
            status = self.query_one("#status", Static)

            # Show spinner message in chat
            spinner_msg = Message("...", "system-info")
            await chat.mount(spinner_msg)
            chat.scroll_end(animate=False)

            start_time = time.time()
            summary = ""
            chunks = 0

            try:
                compact_messages = self.messages.copy()
                compact_messages.append({
                    "role": "system",
                    "content": self.sys_instructions["compact"],
                })

                stream, first_chunk = await self._start_stream(
                    compact_messages, spinner_msg, status, start_time
                )

                # Phase 2: buffer response, show progress
                if first_chunk:
                    content = self._extract_chunk(first_chunk)
                    if content:
                        summary += content
                        chunks += 1

                    while True:
                        chunk = await self._anext(stream)
                        if chunk is _STREAM_DONE:
                            break
                        content = self._extract_chunk(chunk)
                        if content:
                            summary += content
                            chunks += 1
                        elapsed = time.time() - start_time
                        frame = self.SPINNER_FRAMES[int(elapsed * 10) % len(self.SPINNER_FRAMES)]
                        await spinner_msg.update(f"{frame} compacting conversation... {elapsed:.1f}s ({chunks} chunks)")
                        chat.scroll_end(animate=False)
                        status.update(self._status_text(f"compacting... {elapsed:.1f}s ({chunks} chunks)"))

                summary = summary.strip()
                elapsed = time.time() - start_time

                # Rebuild messages: system prompt + summary as context
                self.messages = []
                if self.system_prompt:
                    self.messages.append({"role": "system", "content": self.system_prompt})
                prefix = self.sys_instructions["compact_prefix"]
                self.messages.append({"role": "system", "content": f"{prefix}\n\n{summary}"})

                self._reset_stats()

                # Refresh chat display
                chat.remove_children()
                await self._show_system_message(f"**Conversation compacted** *({elapsed:.1f}s)*\n\n{summary}")
                _log.info("Compacted conversation: %d messages -> summary (%d chars)", len(conv_messages), len(summary))

            except Exception as e:
                _log.exception("Compact error")
                await spinner_msg.remove()
                await self._show_system_message(f"Error compacting: {e}")

    async def _handle_impersonate_short(self) -> None:
        """Generate a short user response suggestion, put it in input."""
        _log.debug("Impersonate short started")
        if self.is_generating:
            return

        if len([m for m in self.messages if m["role"] != "system"]) == 0:
            await self._show_system_message("Need conversation context first")
            return

        async with self._generating_lock():
            input_widget = self.query_one("#chat-input", Input)
            status = self.query_one("#status", Static)

            impersonate_messages = self.messages.copy()
            impersonate_messages.append({
                "role": "system",
                "content": self.sys_instructions["impersonate_short"],
            })

            status.update(self._status_text("impersonating (short)..."))
            input_widget.value = "Impersonating..."

            try:
                result = await asyncio.to_thread(
                    lambda: self._chat_call(impersonate_messages, stream=False)
                )
                response, _ = self._extract_result(result)
                response = _clean_impersonate_response(response)
                _log.debug("Impersonate short result: %s", response[:100])
                input_widget.value = response
                input_widget.cursor_position = len(response)
            except Exception as e:
                _log.exception("Impersonate short error")
                input_widget.value = ""
                await self._show_system_message(f"Error: {e}")

    async def _handle_suggest_toggle(self) -> None:
        """Toggle auto_suggest setting."""
        self.auto_suggest = not self.auto_suggest
        update_config(auto_suggest=self.auto_suggest)
        status = "ON" if self.auto_suggest else "OFF"
        await self._show_system_message(f"Auto-suggest: **{status}**")

    async def _handle_personality_command(self, arg: str) -> None:
        """Handle /personality command for listing and switching personalities."""
        personalities = list_personalities()

        if not arg:
            lines = ["**Available personalities:**"]
            for i, p in enumerate(personalities, 1):
                marker = " ← current" if p == self.personality_name else ""
                lines.append(f"{i}. `{p}`{marker}")
            lines.append("\n*Type `/p <number>` to switch*")
            await self._show_system_message("\n".join(lines))
            return

        choice = arg.strip()
        try:
            idx = int(choice)
            if 1 <= idx <= len(personalities):
                new_personality = personalities[idx - 1]
            else:
                await self._show_system_message(f"Invalid choice (1-{len(personalities)})")
                return
        except ValueError:
            if choice in personalities:
                new_personality = choice
            else:
                await self._show_system_message(f"Personality '{choice}' not found")
                return

        content = load_personality(new_personality)
        if not content:
            await self._show_system_message(f"Error: unable to load '{new_personality}'")
            return

        update_config(personality=new_personality)

        # If conversation has started, restart app to apply cleanly
        has_conversation = any(m["role"] in ("user", "assistant") for m in self.messages)
        if has_conversation:
            await self._restart_app()
            return

        # No conversation yet, just swap the prompt
        self.personality_name = new_personality
        self.system_prompt = content
        self.messages = [{"role": "system", "content": content}]

        await self._show_system_message(f"Personality changed: **{new_personality}**")
