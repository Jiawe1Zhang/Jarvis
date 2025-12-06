"""
Utility script: load knowledge files into a SQLite table for MCP sqlite server.

Usage:
    python rag/import_to_sqlite.py \
        --config config/user_config.json \
        --db data/knowledge.db \
        --table docs

Notes:
- Creates the table if it does not exist: (id, source, content).
- Supports .md/.txt/.csv; .pdf is attempted if pypdf is installed, otherwise skipped.
- Uses knowledge_globs from the provided config.
"""
import argparse
import csv
import json
import sqlite3
from pathlib import Path
from typing import List

from config.loader import load_user_config


def safe_load_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        print(f"[warn] pypdf not installed; skip pdf: {path}")
        return ""
    try:
        reader = PdfReader(str(path))
        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text:
                pages.append(text)
        return "\n\n".join(pages)
    except Exception as exc:
        print(f"[warn] failed to read pdf {path}: {exc}")
        return ""


def load_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:
        print(f"[warn] failed to read {path}: {exc}")
        return ""


def load_csv_file(path: Path) -> str:
    rows: List[str] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                parts = [f"{k}: {v}" for k, v in row.items() if v]
                if parts:
                    rows.append(", ".join(parts))
    except Exception as exc:
        print(f"[warn] failed to read csv {path}: {exc}")
        return ""
    return "\n".join(rows)


def load_file_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".md", ".txt", ".json"}:
        return load_text_file(path)
    if ext == ".csv":
        return load_csv_file(path)
    if ext == ".pdf":
        return safe_load_pdf(path)
    print(f"[warn] unsupported file type {ext} for {path}")
    return ""


def ensure_table(conn: sqlite3.Connection, table: str) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT,
            content TEXT
        )
        """
    )
    conn.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Import knowledge files into SQLite.")
    parser.add_argument("--config", default="config/user_config.json", help="Config file with knowledge_globs.")
    parser.add_argument("--db", default="data/knowledge.db", help="SQLite db path.")
    parser.add_argument("--table", default="docs", help="Table name.")
    args = parser.parse_args()

    cfg = load_user_config(args.config)
    globs = cfg.get("knowledge_globs", [])

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    ensure_table(conn, args.table)
    cur = conn.cursor()

    inserted = 0
    for pattern in globs:
        for path in sorted(Path.cwd().glob(pattern)):
            if not path.is_file():
                continue
            content = load_file_text(path)
            if not content.strip():
                continue
            cur.execute(
                f"INSERT INTO {args.table} (source, content) VALUES (?, ?)",
                (str(path.relative_to(Path.cwd())), content),
            )
            inserted += 1
    conn.commit()
    conn.close()
    print(f"Inserted {inserted} docs into {db_path} (table: {args.table})")


if __name__ == "__main__":
    main()
