import json
from pathlib import Path
from typing import Any, Dict


def load_user_config() -> Dict[str, Any]:
    """
    Load user-facing configuration.
    Priority: config/user_config.json (user copy) -> config/user_config.example.json (default).
    """
    base_dir = Path(__file__).resolve().parent
    user_cfg = base_dir / "user_config.json"
    example_cfg = base_dir / "user_config.example.json"

    cfg_path = user_cfg if user_cfg.exists() else example_cfg
    if not cfg_path.exists():
        return {}
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)
