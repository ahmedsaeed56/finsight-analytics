"""
src/tools/dataset.py
====================

The dataset the tools are currently answering about, and what it IS.

Every tier reads the same three feature frames — Tier 1 aggregates them, Tier 2
looks up rows in them, Tier 3 runs generated pandas over them. So the frames
belong to no single tier and live here, in one place all three import.

WHAT CHANGED, AND WHY IT MATTERS
--------------------------------
These used to be three parquet files read at import: a frozen reference extract
that never changed while the app ran. That was correct while the project was
being built and wrong the moment a user could upload their own file. A company
whose real default rate is 19% must be told 19%, not the reference's 14.1%.

So the state starts empty and is filled by whoever ran the pipeline. Nothing is
loaded until an upload arrives, and _frame() says so plainly rather than
letting a None subscript surface as a TypeError the router cannot act on.

THE DATASET HAS AN IDENTITY, NOT JUST CONTENTS
----------------------------------------------
Three frames alone cannot answer "which file is this?". A user who leaves the
page open overnight, or uploads twice, has no way to know which upload an
answer describes — and an answer whose provenance is unknown is worth less than
one that is caveated.

So _META travels with the frames: a fingerprint of the source files, when they
were loaded, the derived as-of anchor, and the row counts. That makes three
things possible — telling the user which file they are querying, recognising a
re-upload of a file already loaded, and stamping every guardrail log row with
the dataset the question was asked against.

Kept as a SEPARATE module-level name rather than nested inside _DATA, so
_frame() and every existing caller are unchanged.

The fingerprint is COMPUTED in the orchestrator, not here. Hashing source files
is an upload concern, and this package must not become something src/pipeline
imports from — that direction is currently one-way, and reversing it for one
function is how import cycles begin.

ONE DATASET AT A TIME
---------------------
Loading a new one replaces the old. Holding several would mean a `dataset`
argument on every tool AND the router deciding which dataset a question refers
to — "how does last month compare" is then a routing decision, not a
parameter. That is a real feature and it is not this version.

MODULE-LEVEL, NOT A PARAMETER
-----------------------------
The router calls tools with arguments an LLM produced. Threading a dataframe
through that would put data plumbing into the routing layer, where it has no
place — the LLM would have to supply it, and it has nothing to supply.

PERSISTENCE — WHY, AND HOW
--------------------------
Module globals do not survive a process death or a framework's occasional
re-import. Streamlit reruns a script on every widget event, and while imports
usually resolve from sys.modules unchanged, they do not always: memory
pressure, a file-watcher event, a hot-reload flag can wipe the entry, and the
next import re-executes the module body — _DATA snaps back to None mid-
conversation, and every subsequent question errors with "no dataset loaded".

The fix: persist the dataset's IDENTITY to disk — a small JSON file naming the
parquets to read and the label to show. Frames themselves are not serialised
in the JSON, only their paths. On import, if that file exists, load_from_disk
re-hydrates _DATA and _META from it. Any fresh import comes up already loaded.

The pointer lives at data/current_dataset.json. Uploads write their parquets
to data/uploads/<fingerprint>/ before the pipeline returns; the reference
extract points at the training paths already on disk.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import (
    CHURN_FEATURES,
    DEFAULT_FEATURES,
    SEGMENT_FEATURES,
)
from src.tools.errors import ToolError

# The current dataset — whatever the last upload produced. One at a time.
_DATA = None

# What that dataset IS. Always set together with _DATA; the two are only
# separate names so _frame() stays a one-line lookup.
_META = None

# Where the identity pointer lives. A tiny JSON file that lets a fresh import
# rebuild _DATA and _META without an explicit reload call from the caller.
# data/ rather than the repo root because that is where every runtime file
# already sits (app.db, uploads/, checkpoints).
_POINTER_PATH = Path("data") / "current_dataset.json"


def load_dataset(features, label=None, fingerprint=None, as_of=None,
                 paths=None):
    """Point the tools at a new upload.

    Parameters
    ----------
    features
        The dict run_pipeline returns — keys default / churn / segment, values
        the three feature frames.
    label
        What to call this dataset when telling a user which one they are
        querying. A filename, or "reference extract".
    fingerprint
        From the orchestrator's fingerprint_sources(). Optional, because the
        reference extract is loaded from parquets rather than uploaded CSVs
        and has no source files to hash.
    as_of
        The anchor the pipeline derived. Part of the dataset's identity: two
        uploads from the same company differ mainly by their freeze date.
    paths
        A dict of {name: parquet_path} for the three frames. When supplied,
        the pointer file is written so a later import can re-hydrate without
        a reload call. Omitted only by callers that cannot persist — tests
        that build frames in memory, for instance.
    """
    global _DATA, _META

    missing = {"default", "churn", "segment"} - set(features)
    if missing:
        raise ToolError(
            f"dataset is incomplete — missing {sorted(missing)}. "
            f"All three feature tables are required."
        )

    _DATA = dict(features)
    _META = {
        "label": label or "unnamed dataset",
        "fingerprint": fingerprint,
        # UTC, so a timestamp means the same thing wherever this runs.
        "loaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "as_of": str(as_of) if as_of is not None else None,
        "rows": {name: int(len(frame)) for name, frame in _DATA.items()},
    }

    # Write the pointer if we have paths to point at. Best-effort — a failed
    # write means the next import will not auto-hydrate, but the current
    # session still has _DATA in memory and works fine.
    if paths is not None:
        try:
            _write_pointer(paths, _META)
        except OSError:
            pass


def load_reference():
    """Load the reference extract from disk.

    The build-and-validate dataset: what the models were fitted on and where
    the EDA benchmarks came from. Kept so the regression calls stay runnable
    without an upload — it is a test fixture, not the production path, and
    the label says so in any answer that mentions it.

    Passes `paths` so the pointer is written — a Streamlit reload after the
    reference is loaded will come back with the reference still loaded.
    """
    paths = {
        "default": str(Path(DEFAULT_FEATURES).resolve()),
        "churn": str(Path(CHURN_FEATURES).resolve()),
        "segment": str(Path(SEGMENT_FEATURES).resolve()),
    }
    load_dataset(
        {
            "default": pd.read_parquet(DEFAULT_FEATURES),
            "churn": pd.read_parquet(CHURN_FEATURES),
            "segment": pd.read_parquet(SEGMENT_FEATURES),
        },
        label="reference extract (training data, not an upload)",
        paths=paths,
    )


def describe():
    """What is loaded, or None when nothing is.

    For the interface, for the narrator, and for stamping guardrail log rows —
    a logged question is far more useful when you know which dataset it was
    asked against.

    Returns a copy, so a caller cannot edit the live state by accident.
    """
    return dict(_META) if _META is not None else None


def is_same_source(fingerprint):
    """Is the dataset already loaded the one these files produced?

    Lets an upload of an unchanged file say "this is the file already loaded"
    rather than silently rebuilding it.

    False when nothing is loaded, and false when the loaded dataset has no
    fingerprint — the reference extract has no source files, so it can never
    match an upload.
    """
    if _META is None or _META["fingerprint"] is None:
        return False
    return _META["fingerprint"] == fingerprint


def _frame(name):
    """The requested table, or a usable error if nothing is loaded.

    Without this the tools would raise TypeError on a None subscript, which
    tells the router nothing it can act on.
    """
    if _DATA is None:
        raise ToolError(
            "no dataset is loaded. Upload the three files first — the tools "
            "answer questions about the current upload, not about any "
            "built-in data."
        )
    return _DATA[name]


# ==========================================================================
#  PERSISTENCE — the pointer file
# ==========================================================================

def _write_pointer(paths, meta):
    """Save the identity of what is loaded, so a fresh import can find it.

    A pointer, not a snapshot: the parquets themselves already sit on disk
    (in data/uploads/<fp>/ for uploads, in the training paths for the
    reference), and re-serialising megabytes of frame content into JSON on
    every load would be wrong for both size and speed reasons.

    Written atomically — a partial write during a crash must not leave a
    corrupted pointer that fails every future import. Write to a temp path,
    then rename, which is atomic on every OS that runs this.
    """
    _POINTER_PATH.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "paths": paths,
        "meta": meta,
    }

    tmp = _POINTER_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(_POINTER_PATH)


def _clear_pointer():
    """Remove the pointer, so the next import comes up empty.

    Not called anywhere yet — reserved for a future "unload dataset" action.
    Included so callers do not have to know the file's path.
    """
    try:
        _POINTER_PATH.unlink()
    except FileNotFoundError:
        pass


def load_from_disk():
    """Re-hydrate _DATA and _META from the pointer, if one exists.

    Called at import time, at the bottom of this module. Silent no-op when
    the pointer is absent (a fresh install) or when any of its paths are
    missing (the parquets were deleted). Both are ordinary states — the
    interface will show "Nothing loaded" and prompt for an upload or a
    reference load, which is the correct response.

    Any exception during hydration is swallowed to a silent no-op: a broken
    pointer must not make the module unimportable. Every downstream call
    sees "nothing loaded" and behaves accordingly.
    """
    global _DATA, _META

    if not _POINTER_PATH.exists():
        return

    try:
        payload = json.loads(_POINTER_PATH.read_text(encoding="utf-8"))
        paths = payload["paths"]
        meta = payload["meta"]

        # Every parquet must still be there. A missing file means the upload
        # was cleaned up but the pointer wasn't — treat as unloaded.
        for name in ("default", "churn", "segment"):
            if not Path(paths[name]).exists():
                return

        _DATA = {
            "default": pd.read_parquet(paths["default"]),
            "churn": pd.read_parquet(paths["churn"]),
            "segment": pd.read_parquet(paths["segment"]),
        }
        _META = meta
    except (KeyError, ValueError, OSError):
        # Malformed pointer or unreadable parquet. Silent — the caller will
        # see "no dataset loaded" and act on that.
        return


# Run at import. Any fresh module load (including a Streamlit-triggered
# re-import) comes up already loaded if a pointer exists on disk.
load_from_disk() 