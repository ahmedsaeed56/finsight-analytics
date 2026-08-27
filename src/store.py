import sqlite3

from src.config import ROOT

DB_PATH = ROOT / "data" / "app.db"


def get_connection():
    """A connection to the application database."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn 