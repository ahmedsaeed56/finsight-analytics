"""
src/api/models.py
=================

The shapes that cross the HTTP boundary.

Separate from main.py so that file stays a readable list of endpoints, and so
the contract is inspectable on its own — this is what a client codes against.

WHY DECLARE RESPONSE MODELS AT ALL
----------------------------------
Without one, FastAPI documents an endpoint's return as "string" and a client
has to guess. With one, /docs shows every field and its type, FastAPI validates
what you send back, and a field renamed in a tool cannot silently change the
API's contract — the mismatch surfaces here rather than in someone's parser.

Same reasoning as RouterDecision: the schema is the constraint, not a comment.

AGGREGATES CROSS THE WIRE, NEVER ROWS
-------------------------------------
The chart models below return what a plot NEEDS rather than what it was drawn
from. A histogram is bin edges and counts; a boxplot is five numbers per group.
Sending 15,000 rows so a client can compute a median would move a megabyte to
produce one number, and would put the computation on the side of the wire that
this whole project keeps it off.

The scatter is the one exception, and it is already a SAMPLE — three thousand
points, because a full scatter at this size draws a solid block rather than a
pattern.
"""

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ==========================================================================
#  SYSTEM
# ==========================================================================

class HealthResponse(BaseModel):
    """Is the server up, and can it actually answer anything?

    Two different questions. A process can be alive with no dataset loaded, in
    which case every question fails — so the dataset state belongs in the
    health check rather than only in /dataset.

    `single_tenant` is stated rather than assumed: a client that does not know
    the dataset is shared cannot know that another client's upload changed the
    answers underneath it.
    """

    status: str = Field(description="'ok' when the process is serving.")
    dataset_loaded: bool
    dataset: Optional["DatasetInfo"] = None
    single_tenant: bool = Field(
        description=(
            "Always true. One dataset is loaded at a time and shared by every "
            "client — a second upload replaces the first for everyone."
        )
    )


# ==========================================================================
#  DATASET
# ==========================================================================

class DatasetInfo(BaseModel):
    """What is loaded, and enough of its identity to trust an answer.

    Mirrors dataset.describe(). Every figure this API returns describes ONE
    upload, so a client that cannot name the file cannot caveat its numbers —
    which is why the fingerprint and the as-of anchor travel with the label
    rather than being available only on request.
    """

    label: str = Field(description="What to call this dataset — a filename, or 'reference extract'.")
    fingerprint: Optional[str] = Field(
        default=None,
        description=(
            "Content hash of the three source files. None for the reference "
            "extract, which is read from parquets and has no upload to hash."
        ),
    )
    loaded_at: str = Field(description="When this dataset was loaded, UTC.")
    as_of: Optional[str] = Field(
        default=None,
        description="The freeze date the pipeline derived from the file.",
    )
    rows: dict[str, int] = Field(description="Row count per feature table.")


class PipelineReport(BaseModel):
    """What the upload path did, and whether it worked.

    `ok` false means the file was rejected and `failed_at` names the stage.
    The report is returned either way — a rejected upload still needs to say
    why, and the fingerprint survives a failure so the user knows which upload
    is being described.
    """

    ok: bool
    failed_at: Optional[str] = None
    fingerprint: Optional[str] = None
    label: Optional[str] = None
    report: dict[str, Any] = Field(
        description=(
            "Gate result, derived anchor, reconciliation checks, churn panel "
            "and per-table drift. Passed through as the orchestrator built it."
        )
    )
    dataset: Optional[DatasetInfo] = Field(
        default=None,
        description="The now-loaded dataset, on success.",
    )


# ==========================================================================
#  ASK
# ==========================================================================

class AskRequest(BaseModel):
    """One question.

    `thread_id` is what makes a follow-up work: turns are stored against it,
    and the rewriter resolves "what about Sindh?" using the previous ones. A
    client that sends a new id every time gets a system with no memory — which
    is correct behaviour for a one-shot query and wrong for a conversation.
    """

    question: str = Field(
        min_length=1,
        description="The question, in plain language.",
        examples=["what is the default rate by region?"],
    )
    thread_id: str = Field(
        min_length=1,
        description=(
            "Conversation id. Reuse it for follow-ups; generate a new one to "
            "start fresh. Any string — a UUID is the obvious choice."
        ),
        examples=["7f3c1a90-2b44-4c8e-9f01-5d6e7a8b9c00"],
    )


class AskResponse(BaseModel):
    """The answer, plus everything needed to judge it.

    THE RAW RESULT IS INCLUDED ON PURPOSE. `answer` is prose for a human;
    `result` is the tool's own dict, so a client can draw its own chart from
    the same numbers the sentence was written from. Those two can then never
    disagree — which is exactly how the Streamlit interface avoids a chart
    that contradicts its own caption.

    `tier`, `confidence` and `retries` are the audit trail. Tier 1 measured
    what happened, Tier 2 is a model's estimate, Tier 3 is code written at
    runtime — three different kinds of claim, and a client showing them
    identically is misrepresenting two of them.
    """

    answer: str = Field(description="The narrated answer, for a human to read.")
    tool: Optional[str] = Field(
        default=None,
        description=(
            "Which tool ran. 'out_of_scope' when the question could not be "
            "answered from this data; null when a guardrail blocked it before "
            "routing."
        ),
    )
    tier: Optional[int] = Field(
        default=None,
        description=(
            "1 = measured from the file. 2 = a model's estimate. 3 = pandas "
            "generated at runtime. Null for refusals and blocks."
        ),
    )
    confidence: Optional[float] = Field(
        default=None,
        description="How sure the router was, 0-1. Below 0.70 the system asks rather than answers.",
    )
    result: Optional[Any] = Field(
        default=None,
        description=(
            "The tool's raw return — rates, counts, caveats. Draw charts from "
            "THIS rather than from the prose, so the two cannot diverge."
        ),
    )
    expression: Optional[str] = Field(
        default=None,
        description=(
            "Tier 3 only: the pandas that produced the figure. Present because "
            "nobody validated this tier's code in advance."
        ),
    )
    retries: int = Field(
        default=0,
        description="Reroutes spent recovering from a rejected parameter.",
    )
    thread_id: str = Field(description="Echoed back, so a client can chain follow-ups.")
    latency_s: float = Field(description="Wall-clock seconds for the whole turn.")


# ==========================================================================
#  CONVERSATION
# ==========================================================================

class Turn(BaseModel):
    """One stored exchange.

    `question` is the RESOLVED question, not what the user typed. Storing the
    raw follow-up would make the next rewrite resolve against something already
    ambiguous, and the ambiguity would compound down the thread.
    """

    question: str
    answer: str
    tool: Optional[str] = None
    fingerprint: Optional[str] = None
    created_at: Optional[str] = None


class ThreadResponse(BaseModel):
    thread_id: str
    turns: list[Turn]


class ThreadSummary(BaseModel):
    """One conversation in a list of them.

    `title` is the first question asked — the only label that means anything
    without opening the thread.
    """

    thread_id: str
    title: str
    n_turns: int
    last_active: Optional[str] = None


class ThreadListResponse(BaseModel):
    threads: list[ThreadSummary]


# ==========================================================================
#  EXPLORE — direct charts, no model involved
# ==========================================================================

ChartType = Literal[
    "distribution",
    "rate_by_group",
    "box_by_group",
    "correlation",
    "scatter",
    "counts",
]


class ColumnsResponse(BaseModel):
    """What can be charted, split the way the chart types need it.

    A distribution needs a numeric column; a grouping needs a categorical one.
    Sending the split rather than a flat list means the client's selectboxes do
    not have to re-derive it — and cannot derive it differently.

    `categorical` uses the same rule the interface used when it read the frame
    directly: object and category dtypes, plus any numeric column with twelve
    or fewer distinct values. A 0/1 flag is a grouping even though it is stored
    as an integer.
    """

    table: str
    numeric: list[str]
    categorical: list[str]
    n_rows: int


class ExploreRequest(BaseModel):
    """One chart, described rather than drawn.

    ONE ENDPOINT, SIX SHAPES. The alternative — six endpoints — reads more
    cleanly in /docs but means six near-identical branches in the client, which
    is a single selectbox switching between them. So the type is a field and
    the optional parameters are the ones that type needs.

    Which fields matter per type:

        distribution    column, bins
        rate_by_group   metric, group_by
        box_by_group    metric, group_by
        correlation     columns
        scatter         x, y, hue
        counts          column

    Anything irrelevant to the chosen type is ignored rather than rejected —
    a client switching types can leave stale fields in place.
    """

    table: str = Field(
        default="default",
        description="Which feature table — default, churn or segment.",
    )
    chart_type: ChartType

    # --- distribution, counts ---
    column: Optional[str] = None
    bins: int = Field(
        default=40,
        ge=5,
        le=200,
        description="Histogram bins. Bounded because five is unreadable and "
                    "two hundred is noise.",
    )

    # --- rate_by_group, box_by_group ---
    metric: Optional[str] = None
    group_by: Optional[str] = None

    # --- correlation ---
    columns: Optional[list[str]] = Field(
        default=None,
        description="Numeric columns for the matrix. Two or more.",
    )

    # --- scatter ---
    x: Optional[str] = None
    y: Optional[str] = None
    hue: Optional[str] = Field(
        default=None,
        description="Optional categorical column to colour points by.",
    )
    sample: int = Field(
        default=3000,
        ge=100,
        le=10000,
        description=(
            "Points to return for a scatter. The full frame draws as a solid "
            "block rather than a pattern, so this is a sample by design and "
            "the response says how many rows it came from."
        ),
    )


class BoxStats(BaseModel):
    """The five numbers a boxplot actually draws, plus its outliers.

    Computed server-side because seaborn would otherwise need every row to
    find a median. Whiskers follow the standard 1.5 x IQR rule, and anything
    beyond them is an outlier — the same convention seaborn uses, so the plot
    looks identical.
    """

    group: str
    min: float = Field(description="Lower whisker — the smallest value inside 1.5 x IQR.")
    q1: float
    median: float
    q3: float
    max: float = Field(description="Upper whisker.")
    outliers: list[float] = Field(
        default_factory=list,
        description="Points beyond the whiskers, capped so one wild group cannot dominate the payload.",
    )
    n: int


class ScatterPoint(BaseModel):
    x: float
    y: float
    hue: Optional[str] = None


class ExploreResponse(BaseModel):
    """A chart's numbers. Which fields are populated depends on `chart_type`.

    Deliberately one model rather than six: a client switching chart types
    parses one shape and reads the fields its current type uses. Six response
    models would mean six parsers for what is one panel.

    NOTHING HERE PASSED THROUGH A MODEL. This is pandas over the loaded frame,
    so unlike an /ask result there is no narration to check and nothing that
    could have been hallucinated.
    """

    chart_type: ChartType
    table: str
    n_total: int = Field(description="Rows in the table before any sampling.")

    # --- distribution ---
    bin_edges: Optional[list[float]] = Field(
        default=None,
        description="Length is counts + 1 — every bin has a left and right edge.",
    )
    counts: Optional[list[int]] = None

    # --- rate_by_group, counts ---
    values: Optional[dict[str, float]] = Field(
        default=None,
        description="Label to figure. The aggregate for rate_by_group, the row count for counts.",
    )
    group_sizes: Optional[dict[str, int]] = Field(
        default=None,
        description=(
            "Rows behind each group. A 20% rate on 300 rows is not the same "
            "claim as 20% on 3,000, and a bar chart draws them the same height."
        ),
    )
    shares: Optional[dict[str, float]] = Field(
        default=None,
        description="counts only: each level's proportion of the table.",
    )

    # --- box_by_group ---
    boxes: Optional[list[BoxStats]] = None

    # --- correlation ---
    matrix: Optional[dict[str, dict[str, float]]] = Field(
        default=None,
        description="column -> column -> correlation, ordered as requested.",
    )
    matrix_columns: Optional[list[str]] = Field(
        default=None,
        description="The column order, since a dict does not guarantee one.",
    )

    # --- scatter ---
    points: Optional[list[ScatterPoint]] = None
    n_sampled: Optional[int] = Field(
        default=None,
        description="Points returned. Compare against n_total — this is a sample.",
    )

    # --- shared ---
    label: Optional[str] = Field(
        default=None,
        description="Axis label the server suggests, e.g. 'mean defaulted'.",
    )
    note: Optional[str] = Field(
        default=None,
        description="Anything the client should say alongside the chart — a sampling caveat, a thin group.",
    )


# ==========================================================================
#  OBSERVABILITY
# ==========================================================================

class GuardrailCounts(BaseModel):
    """How often each control point fired, and what it did.

    Nested rather than flat because the structure carries the relationship:
    {"scope": {"blocked": 47}} reads as one fact. A count stuck at zero is how
    you find a guardrail that never worked.
    """

    window_days: Optional[int] = Field(
        default=None,
        description="Null means all time.",
    )
    counts: dict[str, dict[str, int]] = Field(
        description="guardrail -> action -> count",
        examples=[{"scope": {"injection": 12, "empty": 3}, "confidence": {"clarify": 8}}],
    )
    total: int


# ==========================================================================
#  INTROSPECTION
# ==========================================================================

class ToolInfo(BaseModel):
    tool: str
    tier: int
    parameters: list[str]
    description: str


class ToolsResponse(BaseModel):
    """What this system can be asked.

    Returned so a client can build a UI, or a person can see the boundary,
    without reading the source. The eleven tools ARE the boundary — anything
    outside them is refused rather than approximated.
    """

    tools: list[ToolInfo]
    note: str


class VocabularyResponse(BaseModel):
    """The metrics, groupings and allowed values in the LOADED file.

    Generated from the data rather than stored, for the same reason the router
    reads it live: a file with a seventh region must be answerable about that
    region, and a hardcoded list of six never would be.
    """

    vocabulary: str = Field(description="Plain text, as the router receives it.")
    dataset: DatasetInfo


# ==========================================================================
#  ERRORS
# ==========================================================================

class ErrorResponse(BaseModel):
    """A refusal a client can act on.

    ToolError messages name the bad value AND list the valid ones, so they are
    worth passing through intact rather than replacing with a generic 400.
    """

    detail: str


# HealthResponse references DatasetInfo before it is defined — this resolves
# the forward reference now that both exist.
HealthResponse.model_rebuild() 