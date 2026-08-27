"""
src/memory/summarise.py
=======================

Fold turns that have fallen out of the recent window into a running summary.

    old turns + existing summary  ->  one merged summary

WHY
---
rewrite.py sees the last six turns. On turn twenty, turns one to fourteen are
gone — and a question like "and the ratio band breakdown there?" cannot resolve
"there" if the region was named fifteen turns ago.

The summary is what those turns collapse into. Not the figures, which are
re-computable, but the SUBJECTS, which are what later references point at.

MERGE, NOT APPEND
-----------------
Each run takes the existing summary plus whatever has newly fallen out, and
produces one paragraph covering both. Appending would grow without bound and
eventually need summarising itself.

WHY TURN COUNT AND NOT TOKENS
-----------------------------
Token counting is the usual trigger and is more precise. It also needs a
tokeniser and a budget. Here, MAX_ANSWER_CHARS already caps how large a single
turn can be, so turn count is a good enough proxy — and a COUNT(*) is cheaper
than tokenising a conversation on every message.
"""

from pathlib import Path

from src.config import PROMPTS_DIR
from src.llm import get_model
from src.memory.conversation import get_summary, set_summary, turns_before
from src.memory.rewrite import MAX_ANSWER_CHARS, RECENT_TURNS

# Nothing is summarised until this many turns sit OUTSIDE the recent window.
# Running on every single turn would mean an LLM call per message to fold in
# one exchange; batching keeps the cost proportional to the conversation
# rather than to the message count.
SUMMARISE_AFTER = 4

_RULES_PATH = Path(PROMPTS_DIR) / "summarise.md"


def needs_summary(thread_id):
    """Whether enough turns have fallen out of the window to be worth folding.

    Cheap enough to call after every turn: one query, no model.
    """
    return len(turns_before(thread_id, RECENT_TURNS)) >= SUMMARISE_AFTER


def update_summary(thread_id, model=None):
    """Merge everything past the recent window into the running summary.

    Returns
    -------
    dict — summarised (bool), summary, n_turns.

    summarised is False when there was nothing old enough to fold, which is
    the common case and costs no model call.
    """
    old = turns_before(thread_id, RECENT_TURNS)

    if len(old) < SUMMARISE_AFTER:
        return {
            "summarised": False,
            "summary": get_summary(thread_id),
            "n_turns": len(old),
        }

    lines = []
    for turn in old:
        lines.append(f"Q: {turn['question']}")
        lines.append(f"A: {turn['answer'][:MAX_ANSWER_CHARS]}")

    existing = get_summary(thread_id) or "(none)"
    rules = _RULES_PATH.read_text(encoding="utf-8")

    prompt = (
        f"{rules}\n\n"
        f"--- EXISTING SUMMARY ---\n"
        f"{existing}\n\n"
        f"--- EXCHANGES ---\n"
        + "\n".join(lines)
        + "\n\nSUMMARY:"
    )

    llm = model or get_model(light=True)
    summary = llm.invoke(prompt).content.strip()

    set_summary(thread_id, summary)

    return {
        "summarised": True,
        "summary": summary,
        "n_turns": len(old),
    } 