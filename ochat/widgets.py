"""Chat widgets for the TUI."""

from textual.containers import ScrollableContainer
from textual.suggester import Suggester
from textual.widgets import Markdown


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
    ]

    async def get_suggestion(self, value: str) -> str | None:
        if not value.startswith("/"):
            return None
        value_lower = value.lower()
        for cmd in self.COMMANDS:
            if cmd.startswith(value_lower) and cmd != value_lower:
                return cmd
        return None


class Message(Markdown):
    """A single chat message with role-based styling."""

    def __init__(self, content: str, role: str = "user") -> None:
        self.role = role
        self.reasoning = ""
        content = content.strip()
        if role == "assistant":
            content = f"● {content}"
        elif role == "user":
            content = f"› {content}"
        super().__init__(content, classes=f"message {role}")

    def update(self, content: str, *, reasoning: str = ""):
        self.reasoning = reasoning
        if self.role == "assistant" and reasoning:
            reasoning_block = "\n".join(f"> {line}" for line in reasoning.splitlines())
            content = f"{reasoning_block}\n\n---\n\n{content}"
        return super().update(content)


class ChatContainer(ScrollableContainer):
    """Scrollable container for chat messages."""
