import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class RunTracer:
    """
    Lightweight JSONL tracer to record prompts, responses, tool calls, etc.
    Each event is appended as a JSON object to events.jsonl under a run-specific directory.
    """

    def __init__(self, log_dir: Path) -> None:
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.events_file = self.log_dir / "events.jsonl"

    def log_event(self, event: Dict[str, Any]) -> None:
        payload = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            **event,
        }
        with self.events_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False))
            f.write("\n")

    def info(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        data = extra.copy() if extra else {}
        data["message"] = message
        self.log_event({"type": "info", **data})
