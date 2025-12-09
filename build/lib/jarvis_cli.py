#!/usr/bin/env python3
"""
Jarvis CLI entrypoint.

Behavior:
- Shows a brief intro (with architecture image if supported in your terminal).
- Provides a tiny prompt: type `run` to start the main pipeline, `quit` to exit.
"""
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

DIAGRAM = r"""
"""

BANNER = r"""
    _       _       ____    __     __   ___    ____  
     | |     / \     |  _ \   \ \   / /  |_ _|  / ___| 
  _  | |    / _ \    | |_) |   \ \ / /    | |   \___ \ 
 | |_| |   / ___ \   |  _ <     \ V /     | |    ___) |
  \___/   /_/   \_\  |_| \_\     \_/     |___|  |____/ 
"""


def render_intro() -> None:
    """Render intro with a banner; fallback to plain text."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.align import Align
    from rich.text import Text

    console = Console()
    console.rule("[bold cyan]Jarvis CLI[/bold cyan]")

    banner_text = Text(BANNER, justify="center", style="bold cyan")
    console.print(Align.center(banner_text))
    info = Table.grid(padding=(0, 1))
    info.add_row("Description", "Jarvis — MCP + RAG agent")
    info.add_row("Command", "type 'run' to start pipeline; 'quit' to exit")
    info_panel = Panel(info, title="Info", expand=False, border_style="green")
    console.print(Align.center(info_panel))
        


def prompt_loop() -> None:
    """Main loop to handle user input."""
    while True:
        try:
            # 增加一个明显的提示符 (如果在终端中运行，通常会有颜色提示)
            # 使用 \n 在每次输入前空一行，视觉更舒适
            cmd = input("\nJARVIS> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            # 处理 Ctrl+C 或 Ctrl+D
            print("\nBye.")
            return

        if cmd in ("run", "start", "main"):
            try:
                # 尝试导入并运行主程序
                from main import main as run_main
                print(">>> Starting Pipeline...")
                run_main()
            except ImportError:
                print("Error: Could not import 'main.py'. Check your file structure.")
            except Exception as e:
                print(f"Runtime Error: {e}")
                
        elif cmd in ("quit", "q", "exit"):
            print("Bye.")
            return
            
        elif cmd in ("help", "h", "?"):
            print("Commands: run | quit")
            
        elif cmd == "":
            continue
            
        else:
            print("Unknown command. Type 'run' or 'quit'.")


def cli() -> None:
    render_intro()
    prompt_loop()


if __name__ == "__main__":
    cli()
