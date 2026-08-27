"""
src/graph/narrate.py
====================

Turns one tool's structured result into a plain-language answer.

    (question, execute_result)  ->  a sentence a person can read

THE LAST STEP, AND THE ONE PYTHON CANNOT CHECK
----------------------------------------------
Every number has already been computed, validated and caveated by the tool
layer. This node only says what it means. That division is the whole point:
Python produces the figures so they are correct, the narrator phrases them so
they are readable — and the moment the narrator does arithmetic, the guarantee
breaks. A wrong number in a fluent sentence is worse than no answer, because
nobody can see it is wrong. narrate.md carries that rule; this function just
delivers the material to it.

THE ONE NODE THAT DOES NOT CONSTRAIN ITS OUTPUT
-----------------------------------------------
The router returns a schema, the sandbox returns a checked expression. This
returns prose, so there is no Pydantic model and no with_structured_output —
just invoke and read the text. That is why this file is shorter than router.py.

BRANCHING LIVES IN THE PROMPT, NOT HERE
---------------------------------------
execute_result has four shapes (success, ToolError, TypeError, out_of_scope),
and narrate.md tells the model to branch on `ok`. This function does NOT read
`ok` itself — duplicating that branch into Python would put the same logic in
two places that can drift. The dict goes over whole; the prompt decides.
"""

import json
from pathlib import Path

from src.config import PROMPTS_DIR
from src.llm import get_model

# Read per call inside the function, not at import, so editing the prompt does
# not need a restart — the same choice router.py and freeform.py make.
_RULES_PATH = Path(PROMPTS_DIR) / "narrate.md"


def narrate(question, execute_result, label=None, model=None):
    """Phrase one execute result as an answer.

    Parameters
    ----------
    question
        The RESOLVED question, so the narrator knows what was asked. Follow-ups
        are already rewritten to standalone form upstream.
    execute_result
        The five-key dict from execute() — ok, tool, result, error, retryable.
        Passed whole; the prompt branches on `ok`, this function does not.
    label
        The dataset label from describe(), so the narrator can say which file
        an answer describes. Optional — omitted when the caller has no dataset
        identity to hand over.
    model
        An override, for tests. Defaults to the light model, like every other
        LLM call in this project.

    Returns
    -------
    str — the answer text. Plain prose, not a structured object: this is the
    one node whose output is meant for a human to read directly.
    """
    rules = _RULES_PATH.read_text(encoding="utf-8")

    # dumps, not the Python repr: f"{dict}" prints np.float64(...) and single
    # quotes, which a model reads less reliably than clean JSON. default=str is
    # a safety net — any value that is not natively serialisable is stringified
    # rather than raising. The reference dicts are clean, so this only guards
    # against a future tool return.
    result_json = json.dumps(execute_result, default=str)

    # Rules first, where a model reads most carefully; the thing being narrated
    # last. Each section under its own header — without them the model receives
    # one undifferentiated wall of text.
    prompt = (
        f"--- RULES ---\n"
        f"{rules}\n\n"
        f"--- QUESTION ---\n"
        f"{question}\n\n"
        f"--- RESULT ---\n"
        f"{result_json}\n\n"
        f"--- DATASET ---\n"
        f"{label}\n\n"
        f"ANSWER:"
    )

    # Plain invoke, no schema. .content pulls the text out of the message
    # object, the same way answer_freeform reads its generated expression.
    llm = model or get_model(light=True)
    return llm.invoke(prompt).content 

if __name__ == "__main__":
    from src.tools.dataset import load_reference
    from src.graph.router import RouterDecision, Tool
    from src.graph.execute import execute

    load_reference()

    d = RouterDecision(
        tool=Tool.AGGREGATE_METRIC,
        params={"metric": "defaulted"},
        confidence=0.98,
        reason="overall rate",
    )
    print(narrate("what is the default rate?", execute(d), label="reference extract")) 