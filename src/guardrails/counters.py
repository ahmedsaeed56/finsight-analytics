from src.store import get_connection
import sqlite3 
from datetime import datetime, timedelta, timezone 

conn = get_connection()

conn.execute('''
    CREATE TABLE IF NOT EXISTS counter (
        id INTEGER PRIMARY KEY,
        fired_time TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        guardrail TEXT NOT NULL,
        question TEXT NOT NULL,
        reason TEXT,
        action TEXT NOT NULL
    )
''')
conn.commit() 
conn.close()

def record(guardrail,question,action,reason=None):

    conn= get_connection()

    try:
        conn.execute(
            "INSERT INTO counter (guardrail,question,action,reason) VALUES (?,?,?,?)",
            (guardrail,question,action,reason),
        )
        conn.commit()
        return True
    except sqlite3.Error:
        # A failed log should not break the thing being logged.
        return False
    finally:
        conn.close()

def read_counts(since_days=None):
    """How often each guardrail fired, and what it did.

    The payoff for logging at all. Without this, "the system has five control
    points" is a claim; with it, a number per control point is evidence — and
    a count stuck at zero is how you discover a guardrail that never worked.

    Parameters
    ----------
    since_days
        Only count events from the last N days. None counts everything.
        A cumulative total stops meaning much after a month — "47 blocks"
        over what period? — so the window is what makes it actionable.

    Returns
    -------
    dict — {guardrail: {action: count}}. Nested rather than flat because the
    structure carries the relationship: {"scope": {"blocked": 47}} reads as
    one fact, where three keys on a flat row leave a narrator to reassemble
    it, and a narrator reassembling relationships can get one wrong. Same
    reason aggregate_metric returns {"Punjab": 0.138}.
    """
    # One code path rather than two. A date old enough that everything passes
    # is simpler than building the SQL conditionally, and the query stays a
    # single fixed string.
    if since_days is None:
        cutoff = "1970-01-01 00:00:00"
    else:
        # UTC, because fired_time comes from SQLite's CURRENT_TIMESTAMP which
        # is UTC. datetime.now() is LOCAL — five hours off here, so a
        # since_days=1 window would silently cover 19 hours or 29.
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=since_days)
        ).strftime("%Y-%m-%d %H:%M:%S")
    
    conn = get_connection()
    try:
        # WHERE before GROUP BY — SQL's clause order is fixed. COUNT(*) counts
        # rows per group; AS n names the column so it can be read by name.
        #
        # (cutoff,) with the trailing comma is a one-item TUPLE. Without it
        # this is a string in parentheses and execute rejects it.
        rows = conn.execute(
            """
            SELECT guardrail, action, COUNT(*) AS n
            FROM counter
            WHERE fired_time > ?
            GROUP BY guardrail, action
            """,
            (cutoff,),
        ).fetchall()
    finally:
        # After fetchall, never before — closing kills the cursor and the rows
        # with it.
        conn.close()

    counts = {}
    for row in rows:
        # setdefault returns the existing inner dict, or creates an empty one
        # first. Without it the first action for a guardrail would raise
        # KeyError, because that key does not exist yet.
        counts.setdefault(row["guardrail"], {})[row["action"]] = row["n"]

    return counts 