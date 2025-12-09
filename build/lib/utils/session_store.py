import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional


class SessionStore:
    """
    Turn-based SQLite store. Each turn groups user->tool(s)->assistant messages.
    We persist turns only when completed to avoid half-written history.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    turn_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def append_turn(self, session_id: str, messages: List[Dict], max_turns: int = 0) -> None:
        """
        Persist a completed turn (list of messages). Older turns pruned if max_turns > 0.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT INTO turns (session_id, turn_json) VALUES (?, ?)",
                (session_id, json.dumps(messages, ensure_ascii=False)),
            )
            if max_turns and max_turns > 0:
                conn.execute(
                    """
                    DELETE FROM turns
                    WHERE id NOT IN (
                        SELECT id FROM turns
                        WHERE session_id = ?
                        ORDER BY id DESC
                        LIMIT ?
                    )
                    AND session_id = ?
                    """,
                    (session_id, max_turns, session_id),
                )
            conn.commit()
        finally:
            conn.close()

    def load_turns(self, session_id: str, limit: int = 0) -> List[List[Dict]]:
        conn = sqlite3.connect(self.db_path)
        try:
            if limit and limit > 0:
                cur = conn.execute(
                    """
                    SELECT turn_json FROM (
                        SELECT id, turn_json FROM turns
                        WHERE session_id = ?
                        ORDER BY id DESC
                        LIMIT ?
                    )
                    ORDER BY id ASC
                    """,
                    (session_id, limit),
                )
            else:
                cur = conn.execute(
                    "SELECT turn_json FROM turns WHERE session_id = ? ORDER BY id ASC",
                    (session_id,),
                )
            rows = cur.fetchall()
        finally:
            conn.close()

        turns: List[List[Dict]] = []
        for row in rows:
            try:
                data = json.loads(row[0])
                if isinstance(data, list):
                    turns.append(data)
            except Exception:
                continue
        return turns
