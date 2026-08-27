"""
src/llm.py
==========

One place the language models are configured.

Every LLM call in this project comes through here — the Tier 3 expression
generator, the Phase 4 router, and the narrator. Configuring them at each call
site is how a model name ends up hardcoded in six files and a temperature
change becomes a six-file edit.

WHY THE NATIVE PATH, NOT THE OPENAI-COMPATIBILITY LAYER
-------------------------------------------------------
Gemini exposes an OpenAI-compatible endpoint, and it works. It also hides
Gemini-specific features behind the translation — and more importantly, JSON
mode there is a REQUEST for a shape, where the native `response_schema` is a
CONSTRAINT on it.

The router decides which tool runs from a structured object (tier, tool,
params, confidence). A malformed response there is not a bad answer, it is a
crash or a wrong tool. Enforced beats requested.

TWO MODELS, TWO JOBS
--------------------
Routing and expression generation are short, structured, mechanical — the
cheap model does them well. Narration is the one place fluency matters, and
it is one call per question rather than one per retry.
"""

import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Structured, mechanical work: routing, generating a pandas expression,
# rewriting a failed one. Short prompts, short outputs, many of them.
LIGHT_MODEL = "gemini-2.5-flash-lite"

# Narration — turning a tool's structured facts into a sentence a person
# reads. One call per question, and the only place wording matters.
MAIN_MODEL = "gemini-2.5-flash"

# Zero everywhere. This project's governing rule is that the LLM routes and
# narrates but never computes, and a temperature above zero makes the same
# question route two different ways on two runs. For an audit trail, that is
# not a style preference — it is the difference between explainable and not.
TEMPERATURE = 0


def get_model(light=True):
    """A configured chat model.

    Parameters
    ----------
    light
        True for routing and code generation, False for narration.

    Raises
    ------
    RuntimeError
        No API key. Raised here rather than surfacing as an auth error
        several layers deep, where the cause is much harder to see.
    """
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Put it in a .env file in the project "
            "root, or export it in the shell."
        )

    return ChatGoogleGenerativeAI(
        model=LIGHT_MODEL if light else MAIN_MODEL,
        temperature=TEMPERATURE,
    )  