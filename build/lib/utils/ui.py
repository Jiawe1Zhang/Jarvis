from __future__ import annotations

"""
Lightweight TUI wrapper using rich. Falls back to no-op if disabled or rich is missing.
"""

from collections import deque
from contextlib import nullcontext
from typing import Any, Dict, Iterable, Optional

try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.live import Live
    from rich.table import Table
    from rich.text import Text
    from rich.markdown import Markdown

    RICH_AVAILABLE = True
except Exception:
    RICH_AVAILABLE = False


class BaseUI:
    enabled = False

    def live(self):
        return nullcontext()

    def stage(self, stage: str, status: str) -> None:
        pass

    def log(self, source: str, content: str) -> None:
        pass

    def tool(self, name: str, args: Dict[str, Any], result: Optional[Any] = None) -> None:
        pass

    def detail(self, title: str, content: Any) -> None:
        pass

    def stats(self) -> None:
        pass


class RichUI(BaseUI):
    enabled = True

    def __init__(self) -> None:
        if not RICH_AVAILABLE:
            raise RuntimeError("rich is not available")
        self.console = Console()
        self.messages = deque(maxlen=200)
        self.tool_calls = deque(maxlen=100)
        self.current_detail_title = "Detail"
        self.current_detail = "[dim]Waiting...[/dim]"
        self.stages: Dict[str, str] = {
            "Initialization": "pending",
            "RAG Retrieval": "pending",
            "Query Rewriting": "pending",
            "Agent Reasoning": "pending",
            "Tool Execution": "pending",
            "Final Response": "pending",
        }
        self.layout = self._create_layout()
        self.live_obj: Optional[Live] = None

    def live(self):
        self._update_layout()
        self.live_obj = Live(self.layout, refresh_per_second=4, screen=True)
        return self.live_obj

    def stage(self, stage: str, status: str) -> None:
        if stage in self.stages:
            self.stages[stage] = status
            self._refresh()

    def log(self, source: str, content: str) -> None:
        self.messages.append((source, content))
        # Long model content goes to detail panel
        if source.lower() == "model" and len(content) > 120:
            self.current_detail_title = "Agent Response"
            self.current_detail = content
        self._refresh()

    def tool(self, name: str, args: Dict[str, Any], result: Optional[Any] = None) -> None:
        self.tool_calls.append((name, args, result))
        detail = f"[bold magenta]{name}[/bold magenta]\nArgs: {args}"
        if result is not None:
            detail += f"\nResult: {result}"
        self.current_detail_title = f"Tool: {name}"
        self.current_detail = detail
        self._refresh()

    def detail(self, title: str, content: Any) -> None:
        self.current_detail_title = title
        self.current_detail = content
        self._refresh()

    def stats(self) -> None:
        self._refresh()

    # Internal helpers
    def _create_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3),
        )
        layout["main"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="right", ratio=3),
        )
        layout["left"].split_column(
            Layout(name="progress", size=10),
            Layout(name="events"),
        )
        return layout

    def _render_progress(self) -> Panel:
        table = Table.grid(padding=(0, 1))
        table.add_column("Stage", style="bold white")
        table.add_column("Status", justify="right")
        for stage, status in self.stages.items():
            status_text = {
                "pending": "[dim]pending[/dim]",
                "in_progress": "[yellow]running[/yellow]",
                "completed": "[green]done[/green]",
                "error": "[red]error[/red]",
            }.get(status, status)
            table.add_row(stage, status_text)
        return Panel(table, title="Workflow", border_style="blue")

    def _render_events(self) -> Panel:
        table = Table(show_header=True, header_style="bold", expand=True)
        table.add_column("Source", width=10)
        table.add_column("Content")
        for source, content in list(self.messages)[-15:]:
            color = {"user": "green", "model": "cyan", "system": "yellow"}.get(source.lower(), "magenta")
            truncated = content.replace("\n", " ")
            if len(truncated) > 120:
                truncated = truncated[:117] + "..."
            table.add_row(f"[{color}]{source}[/{color}]", truncated)
        return Panel(table, title="Events", border_style="green")

    def _render_detail(self) -> Panel:
        content = self.current_detail
        if isinstance(content, str):
            content_renderable = Markdown(content) if content.strip().startswith("#") else Text(content)
        else:
            content_renderable = Text(str(content))
        return Panel(content_renderable, title=self.current_detail_title, border_style="yellow")

    def _render_footer(self) -> Panel:
        stats = f"Messages: {len(self.messages)} | Tools: {len(self.tool_calls)}"
        return Panel(Text(stats, justify="center", style="dim"))

    def _update_layout(self) -> None:
        self.layout["header"].update(
            Panel(
                "[bold cyan]Jarvis[/bold cyan]  [dim]MCP + RAG[/dim]",
                border_style="cyan",
            )
        )
        self.layout["progress"].update(self._render_progress())
        self.layout["events"].update(self._render_events())
        self.layout["right"].update(self._render_detail())
        self.layout["footer"].update(self._render_footer())

    def _refresh(self) -> None:
        if not self.live_obj:
            return
        self._update_layout()
        self.live_obj.update(self.layout)


def get_ui(enabled: bool) -> BaseUI:
    if enabled and RICH_AVAILABLE:
        try:
            return RichUI()
        except Exception:
            return BaseUI()
    return BaseUI()
