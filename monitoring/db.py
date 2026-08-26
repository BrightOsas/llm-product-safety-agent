"""
monitoring/db.py

Lightweight SQLite logging for every question the agent answers, plus
user feedback (thumbs up/down). This is the data source for the
Streamlit monitoring dashboard.
"""

import json
import sqlite3
import time
import uuid
from pathlib import Path

DB_PATH = Path(__file__).parent / "monitoring.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS interactions (
            id TEXT PRIMARY KEY,
            timestamp REAL,
            question TEXT,
            answer TEXT,
            tool_calls TEXT,
            response_time_seconds REAL,
            feedback INTEGER  -- 1 = thumbs up, -1 = thumbs down, NULL = no feedback
        )
        """
    )
    conn.commit()
    conn.close()


def log_interaction(question: str, answer: str, tool_calls: list[dict], response_time_seconds: float) -> str:
    interaction_id = str(uuid.uuid4())
    conn = get_connection()
    conn.execute(
        """
        INSERT INTO interactions (id, timestamp, question, answer, tool_calls, response_time_seconds, feedback)
        VALUES (?, ?, ?, ?, ?, ?, NULL)
        """,
        (interaction_id, time.time(), question, answer, json.dumps(tool_calls), response_time_seconds),
    )
    conn.commit()
    conn.close()
    return interaction_id


def log_feedback(interaction_id: str, feedback: int):
    """feedback: 1 for thumbs up, -1 for thumbs down."""
    conn = get_connection()
    conn.execute("UPDATE interactions SET feedback = ? WHERE id = ?", (feedback, interaction_id))
    conn.commit()
    conn.close()


def fetch_all_interactions() -> list[dict]:
    conn = get_connection()
    rows = conn.execute("SELECT * FROM interactions ORDER BY timestamp ASC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ensure the table exists as soon as this module is imported
init_db()