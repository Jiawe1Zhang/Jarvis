from dataclasses import dataclass
import shutil
import sys


def log_title(message: str) -> None:
    """
    Render a highlighted title line similar to the original chalk-based helper.
    """
    terminal_width = shutil.get_terminal_size((80, 20)).columns
    message = f" {message.strip()} "
    padding = max(0, terminal_width - len(message))
    prefix = "=" * (padding // 2)
    suffix = "=" * (padding - len(prefix))
    sys.stdout.write(f"{prefix}{message}{suffix}\n")


@dataclass
class ToolCall:
    """
    Mirror the OpenAI tool call structure for clarity in type hints.
    """
    id: str
    name: str
    arguments: str
