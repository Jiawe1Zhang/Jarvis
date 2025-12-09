import json
from pathlib import Path
from typing import Any, Dict, Optional


def load_user_config(
    config_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Load user-facing configuration.
    Priority:
      - if config_path is provided, load that file
      - else config/user_config.json (user copy)
      - else config/user_config.example.json (default)
    If overrides are provided, shallow-merge them on top of the loaded config.
    """
    base_dir = Path(__file__).resolve().parent
    user_cfg = base_dir / "user_config.json"
    example_cfg = base_dir / "user_config.example.json"

    if config_path:
        cfg_path = Path(config_path)
    else:
        cfg_path = user_cfg if user_cfg.exists() else example_cfg

    if not cfg_path.exists():
        cfg = {}
    else:
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)

    if overrides:
        cfg = {**cfg, **overrides}

    return cfg
