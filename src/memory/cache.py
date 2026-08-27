"""
src/memory/cache.py
===================

Answers already computed for this exact dataset.

    (fingerprint, tier, tool, params)  ->  the tool's return

WHY THERE IS NO TTL ON READS
----------------------------
The FINGERPRINT is in the key. A hit therefore means: same dataset, same tool,
same parameters — and the data is byte-identical to when the entry was
written. aggregate_metric("defaulted") returned 0.1403 then and returns 0.1403
now, whether that was an hour ago or a month.

A new upload has a different fingerprint and misses naturally. Invalidation
lives in the key rather than on a clock, which is stronger: a clock can expire
a still-correct answer, and can also serve a stale one inside its window.

LegalSpy needed a TTL because its key was the question alone, with nothing
identifying which documents were loaded. A new document could change the right
answer while the key stayed the same, and time was the only defence available.

CACHE_DAYS IS HOUSEKEEPING, NOT CORRECTNESS
-------------------------------------------
Without it the table grows forever, holding answers for datasets nobody will
load again. clear_old() deletes them. cache_get() does not check age — that
would put a date comparison on every read to solve a disk-space problem.

SQLite has no scheduler, so clear_old has to be called by something. App
startup is the natural place.

WHICH TIERS
-----------
All three. The original plan said never cache Tier 3, but the problem was
never the tier — it was the key. Without the fingerprint, a cached
aggregate_metric("defaulted") would serve the reference extract's 0.1412 to a
user whose upload says 0.1403. With it, Tier 3 becomes the tier most worth
caching, because a hit skips a paid LLM call rather than a millisecond of
pandas.
"""

import hashlib
import json
import sqlite3
from datetime import datetime, timedelta, timezone

from src.store import get_connection

# How long an entry survives before housekeeping removes it. Not a validity
# window — see the module docstring.
CACHE_DAYS = 3

# Separator for the key parts. Chosen because it cannot appear in a
# fingerprint (hex), a tier name, or a tool name. Without one, ("ab", "c") and
# ("a", "bc") both flatten to "abc" and collide.
_SEPARATOR = "|"


_conn = get_connection()
_conn.execute('''
    CREATE TABLE IF NOT EXISTS cache (
        key         TEXT PRIMARY KEY,
        label       TEXT NOT NULL,
        value       TEXT NOT NULL,
        written_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
''') 
_conn.commit()
_conn.close()


def make_key(fingerprint, tier, tool, params=None):
    """One deterministic key for a tool call against a dataset.

    PARAMS ARE SORTED. A dict has no inherent order, so {"a": 1, "b": 2} and
    {"b": 2, "a": 1} would otherwise hash differently — and the same call
    would miss its own cache entry depending on how the router happened to
    build the dict.

    HASHED so the key is a fixed length whatever the params contain. A filter
    dict can be long; a primary key should not be.

    Returns
    -------
    (key, label) — the hash for lookups, and the readable form for anyone
    inspecting the table. A cache you cannot read is a cache you cannot debug.
    """
    encoded = json.dumps(params or {}, sort_keys=True)
    label = _SEPARATOR.join([str(fingerprint), str(tier), str(tool), encoded])

    # .encode() because sha256 takes bytes, not text. fingerprint_sources
    # hashes file bytes directly and needs no equivalent.
    key = hashlib.sha256(label.encode("utf-8")).hexdigest()

    print(f"[MAKE KEY] label={label}")

    return key, label


def cache_get(fingerprint, tier, tool, params=None):
    """The stored answer, or None if there isn't one.

    None means MISS — the caller runs the tool. It never means "the answer was
    None": tools return dicts, so a stored value is always a dict.
    """
    key, _ = make_key(fingerprint, tier, tool, params)

    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT value FROM cache WHERE key = ?",
            (key,),
        ).fetchone()
    except sqlite3.Error:
        return None
    finally:
        conn.close()

    print(f"[CACHE SQL] key={key[:16]}... row={'FOUND' if row else 'NONE'}", flush=True)

    if row is None:
        return None

    try:
        return json.loads(row["value"])
    except json.JSONDecodeError:
        # A corrupted entry is a miss too. Better a recomputed answer than a
        # crash on something that only exists to save time.
        return None 


def cache_set(fingerprint, tier, tool, value, params=None):
    """Store one answer.

    ON CONFLICT DO UPDATE rather than DO NOTHING: rewriting refreshes
    written_at, so an entry that keeps being used stays alive and only genuinely
    idle ones age out.

    Returns
    -------
    bool — whether the write succeeded. Best-effort, like every other write
    here: a cache that cannot store is slow, not broken.
    """
    key, label = make_key(fingerprint, tier, tool, params)

    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO cache (key, label, value)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                written_at = CURRENT_TIMESTAMP
            """,
            (key, label, json.dumps(value)),
        )
        conn.commit()
        return True
    except (sqlite3.Error, TypeError):
        # TypeError as well as a database error: json.dumps raises it on a
        # value it cannot serialise. That is a bug in the tool's return shape,
        # but it must not take the answer down with it.
        return False
    finally:
        conn.close()


def clear_old(days=CACHE_DAYS):
    """Delete entries older than `days`.

    Housekeeping, not invalidation — nothing here is wrong, just unlikely to
    be asked for again. Call it at startup.

    UTC, because written_at comes from SQLite's CURRENT_TIMESTAMP which is UTC
    while datetime.now() is local. Five hours out in Pakistan, which at day
    granularity would quietly delete a few hours early or late.

    Returns
    -------
    int — how many rows were removed.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    conn = get_connection()
    try:
        cursor = conn.execute(
            "DELETE FROM cache WHERE written_at < ?",
            (cutoff,),
        )
        conn.commit()
        return cursor.rowcount
    except sqlite3.Error:
        return 0
    finally:
        conn.close() 