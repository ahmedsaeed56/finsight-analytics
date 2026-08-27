"""
src/memory/rewrite.py
=====================

Turn a follow-up into a question that stands on its own.

    "what about only nano loans?"  ->  "what is the default rate by region
                                        for nano loans?"

WHY THIS RUNS BEFORE THE ROUTER
-------------------------------
The router picks a tool and its parameters from the question alone. It cannot
do that with "what about Sindh?" — there is no metric in it, no grouping, and
no way to know which of the last five answers is being narrowed. Resolving the
reference first is what makes routing a decision rather than a guess.

The alternative — handing the whole conversation to the router and letting it
work out both things at once — costs history tokens on every routing call and
makes the router's job two jobs.

TWO SOURCES OF CONTEXT
----------------------
The recent turns are the primary source and are shown in full. The running
summary covers everything that has fallen out of that window, and exists so a
reference to something fifteen turns back can still resolve.

They are ordered summary first, then turns: oldest to newest, so recency reads
down the page and the model's attention lands on the most recent material last.

ALWAYS CALLS, RATHER THAN GUESSING WHETHER IT NEEDS TO
------------------------------------------------------
A keyword test for ambiguity would be the same shape as the injection list,
against a much larger space: "and for defaulters only?", "same but Punjab",
"break that down" share no vocabulary. Twenty patterns would still miss the
twenty-first. So the model is asked every time and told to return an already
standalone question unchanged — one cheap call beats a rule that silently
fails.

The one exception is having neither turns nor a summary, where there is
nothing to resolve against and no call is worth making.
"""

from pathlib import Path

from src.config import PROMPTS_DIR
from src.llm import get_model
from src.memory.conversation import get_summary, recent_turns

# How many past exchanges to show. Six is enough to resolve a reference two or
# three turns back, and few enough to keep the prompt small — anything older
# belongs in the running summary rather than here.
RECENT_TURNS = 6

# Answers can be long: a Tier 3 result table or a fifty-row ranking would
# dominate the prompt. The rewriter needs to know WHAT was discussed, not every
# figure, so each answer is cut to roughly a sentence or two.
MAX_ANSWER_CHARS = 200

# The rules never change; the conversation does. Same split as freeform.md.
_RULES_PATH = Path(PROMPTS_DIR) / "rewrite.md"


def rewrite_question(question, thread_id, fingerprint=None, model=None):
    """Resolve a follow-up into a standalone question.

    Parameters
    ----------
    question
        What the user typed, in their own words.
    thread_id
        Which conversation to read history from.
    fingerprint
        The dataset currently loaded. Turns recorded against a DIFFERENT
        upload are dropped before the model sees them: resolving "what about
        Sindh?" against numbers from last week's file would produce a question
        about a population that is no longer there.
    model
        An override, for tests. Defaults to the light model.

    Returns
    -------
    dict — question (the resolved one), original, rewritten (bool).

    A dict rather than a bare string because whether a rewrite happened is
    worth tracing: it is the difference between a question the user asked and
    one the system inferred, and if a downstream answer is wrong, that is the
    first thing to check.
    """
    data = recent_turns(thread_id, RECENT_TURNS)

    history = [turn for turn in data if turn["fingerprint"] == fingerprint]

    # The compressed record of everything older than the window. Not filtered
    # by fingerprint — a summary spans whatever the thread has covered, and
    # there is no per-sentence fingerprint to filter on. It is background, and
    # the prompt tells the model to treat it that way.
    summary = get_summary(thread_id)

    # Nothing to resolve against at all. Covers the first question in a thread,
    # and the case where every past turn was recorded against a different
    # upload — those describe numbers that are no longer loaded, so they are
    # worse than no history.
    #
    # Returns without calling the model: there is no reference to resolve, and
    # a model asked to rewrite a standalone question will sometimes "improve"
    # it instead.
    if not history and not summary:
        return {
            "question": question,
            "original": question,
            "rewritten": False,
        }

    # The turns are dicts; the model needs readable text. Two lines each, in
    # the same Q:/A: shape the examples in rewrite.md use — the model resolves
    # references better against the format it was taught.
    lines = []
    for turn in history:
        lines.append(f"Q: {turn['question']}")

        # Trimmed, not dropped. A Tier 3 result table or a fifty-row ranking
        # would dominate the prompt, and the rewriter needs to know WHAT was
        # discussed rather than every figure — enough to resolve "the second
        # one" or "it", not to reproduce the answer.
        answer = turn["answer"][:MAX_ANSWER_CHARS]
        lines.append(f"A: {answer}")

    conversation = "\n".join(lines) or "(no recent turns)"

    # Read per call rather than at import, so editing the prompt does not need
    # a restart. No existence check — Python's FileNotFoundError already names
    # the path, and a wrapper would add nothing.
    rules = _RULES_PATH.read_text(encoding="utf-8")

    # Rules, then the summary, then the recent turns, then the question.
    # Oldest context first so recency reads downward; the question last
    # because it is the thing being acted on.
    parts = [rules, "\n\n"]

    if summary:
        parts.append(f"--- EARLIER IN THIS CONVERSATION ---\n{summary}\n\n")

    parts.append(f"--- RECENT TURNS ---\n{conversation}\n\n")
    parts.append(f"--- QUESTION ---\n{question}\n\n")
    parts.append("REWRITTEN QUESTION:")

    prompt = "".join(parts)

    llm = model or get_model(light=True)
    resolved = llm.invoke(prompt).content.strip()

    return {
        "question": resolved,
        "original": question,
        # Compared rather than assumed. The model is told to return an
        # already-standalone question unchanged, and this is how you find out
        # whether it obeyed — a rewrite that fires on every question is a
        # prompt problem, not a feature.
        "rewritten": resolved != question,
    } 