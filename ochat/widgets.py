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

    def __init__(self, content: str, role: str = "user",
                 reasoning: str = "") -> None:
        self.role = role
        self.reasoning = reasoning
        display_content = content.strip()
        if role == "assistant" and reasoning:
            display_content = f"● > {reasoning}\n\n---\n\n{content}"
        elif role == "assistant":
            display_content = f"● {display_content}"
        elif role == "user":
            display_content = f"› {display_content}"
        super().__init__(display_content, classes=f"message {role}")

    def update(self, content: str, *, reasoning: str = "") -> None:
        self.reasoning = reasoning
        display_content = content.strip()
        if self.role == "assistant" and reasoning:
            display_content = f"● > {reasoning}\n\n---\n\n{content}"
        elif self.role == "assistant":
            display_content = f"● {display_content}"
        elif self.role == "user":
            display_content = f"› {display_content}"
        super().update(display_content)


class ChatContainer(ScrollableContainer):
    """Scrollable container for chat messages."""
