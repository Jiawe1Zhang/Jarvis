import json
import sqlite3
from pathlib import Path
from typing import Dict, List


class SessionStore:
    """
    Minimal SQLite-backed session store for chat history.
    Stores each message as JSON (OpenAI chat format) keyed by session_id.
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
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    message TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def append(self, session_id: str, message: Dict, max_messages: int = 0) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT INTO messages (session_id, message) VALUES (?, ?)",
                (session_id, json.dumps(message, ensure_ascii=False)),
            )
            if max_messages and max_messages > 0:
                conn.execute(
                    """
                    DELETE FROM messages
                    WHERE id NOT IN (
                        SELECT id FROM messages
                        WHERE session_id = ?
                        ORDER BY id DESC
                        LIMIT ?
                    )
                    AND session_id = ?
                    """,
                    (session_id, max_messages, session_id),
                )
            conn.commit()
        finally:
            conn.close()

    def load(self, session_id: str, limit: int = 0) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        try:
            if limit and limit > 0:
                cur = conn.execute(
                    "SELECT message FROM (SELECT id, message FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?) ORDER BY id ASC",
                    (session_id, limit),
                )
            else:
                cur = conn.execute(
                    "SELECT message FROM messages WHERE session_id = ? ORDER BY id ASC",
                    (session_id,),
                )
            rows = cur.fetchall()
        finally:
            conn.close()
        messages: List[Dict] = []
        for row in rows:
            try:
                messages.append(json.loads(row[0]))
            except Exception:
                continue
        return messages
