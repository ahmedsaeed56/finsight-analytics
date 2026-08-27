from src.store import get_connection 
import sqlite3 

_conn= get_connection()

_conn.execute('''
    CREATE TABLE IF NOT EXISTS conversation(
    thread_id TEXT PRIMARY KEY,
    started_at    TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    fingerprint   TEXT,
    summary       TEXT
    )
''')

_conn.execute('''
    CREATE TABLE IF NOT EXISTS turn(
    id            INTEGER PRIMARY KEY,
    created_at    TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    thread_id     TEXT NOT NULL,
    question      TEXT NOT NULL,
    answer        TEXT NOT NULL,
    tool          TEXT,
    fingerprint   TEXT
    )
''')
_conn.commit()
_conn.close()

def start_conversation(thread_id, fingerprint=None):
    """Register a thread, or do nothing if it already exists.

    IDEMPOTENT ON PURPOSE. This is called on every message, not just the
    first — the caller should not have to track whether a thread is new.
    ON CONFLICT DO NOTHING makes the second call and the fiftieth harmless,
    and it works because thread_id is the primary key, so there is something
    for the conflict to be against.

    `fingerprint` records which dataset the thread STARTED on. The turn table
    keeps its own per-turn copy, so a mid-conversation re-upload is visible as
    a divergence between the two.

    Returns
    -------
    bool — True if this call created the thread, False if it already existed.
    The caller may want to know: a new thread has no history to rewrite
    against, and no summary to load.
    """
    conn = get_connection()
    try:
        cursor = conn.execute(
            """
            INSERT INTO conversation (thread_id, fingerprint)
            VALUES (?, ?)
            ON CONFLICT DO NOTHING
            """,
            (thread_id, fingerprint),
        )
        conn.commit()
        # 1 when the row was inserted, 0 when the conflict fired.
        return cursor.rowcount == 1
    except sqlite3.Error:
        # Same rule as record(): a failed write must not break the thing
        # being written about.
        return False
    finally:
        conn.close() 


def add_turn(thread_id, question, answer, tool=None, fingerprint=None):
    """Record one exchange.

    `question` is the RESOLVED question, not the raw follow-up. Storing
    "what about Sindh?" would make the next rewrite resolve against something
    already ambiguous, and the ambiguity would compound down the thread.

    `fingerprint` is the dataset AS IT WAS for this turn. A user who uploads a
    new file mid-conversation leaves earlier turns describing numbers that no
    longer exist — this column is how those turns are recognised and dropped
    rather than fed to the rewriter as context.

    Returns
    -------
    bool — whether the write succeeded. Best-effort, like every other write
    in this project: a broken log must not break a working answer.
    """
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO turn (thread_id, question, answer, tool, fingerprint)
            VALUES (?, ?, ?, ?, ?)
            """,
            (thread_id, question, answer, tool, fingerprint),
        )
        conn.commit()
        return True
    except sqlite3.Error:
        return False
    finally:
        conn.close() 

def recent_turns(thread_id, n=6):
    """The last N exchanges, oldest first.

    ORDER BY id DESC LIMIT n gets the LAST n; ascending with a limit would
    get the FIRST n, which is the wrong end of the conversation. The rows are
    then reversed so they read forward — a rewriter or a summariser needs the
    conversation in the order it happened, not backwards.

    Six by default. Enough to resolve "what about Sindh?" against, few enough
    that the prompt stays small; anything older belongs in the summary.
    """
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT question, answer, tool, fingerprint, created_at
            FROM turn
            WHERE thread_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (thread_id, n),
        ).fetchall()
    except sqlite3.Error:
        # An unreadable history is not a reason to refuse the question — the
        # rewriter simply has nothing to resolve against.
        return []
    finally:
        conn.close()

    # Newest-first out of the query, so reverse for conversation order.
    return [dict(row) for row in reversed(rows)]


def get_summary(thread_id):
    """The running summary of everything older than the recent turns.

    None when the thread is new or short enough not to need one.
    """
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT summary FROM conversation WHERE thread_id = ?",
            (thread_id,),
        ).fetchone()
    except sqlite3.Error:
        return None
    finally:
        conn.close()

    return row["summary"] if row else None


def set_summary(thread_id, summary):
    """Replace the running summary.

    REPLACED, not appended — that is the whole difference between the two
    tables. Turns accumulate because each one happened; the summary is a
    single current statement of everything before the recent window, so a
    new one supersedes the old.

    Returns
    -------
    bool — whether the write succeeded.
    """
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE conversation SET summary = ? WHERE thread_id = ?",
            (summary, thread_id),
        )
        conn.commit()
        return True
    except sqlite3.Error:
        return False
    finally:
        conn.close() 

def turns_before(thread_id, offset):
    """Turns older than the most recent `offset`, oldest first.

    The mirror of recent_turns. That returns the window the rewriter sees;
    this returns everything that has fallen out of it and needs folding into
    the summary.

    OFFSET, not LIMIT. `ORDER BY id DESC LIMIT -1 OFFSET 6` means "skip the
    six newest, take everything else" — the -1 is SQLite's way of saying no
    limit, which OFFSET requires.
    """
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT question, answer, tool, fingerprint, created_at
            FROM turn
            WHERE thread_id = ?
            ORDER BY id DESC
            LIMIT -1 OFFSET ?
            """,
            (thread_id, offset),
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()

    return [dict(row) for row in reversed(rows)] 