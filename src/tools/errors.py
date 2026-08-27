"""
src/tools/errors.py
===================

Exceptions shared across the tool layer.

WHY THIS IS ITS OWN FILE
------------------------
ToolError started in analytics.py, where it was first needed. It is raised in
at least three places now — the whitelist checks, the dataset accessor, and
the ID lookups in inference — so leaving it in any one of them would force the
others to import from it. dataset.py raising analytics.py's exception while
analytics.py imports dataset.py's accessor is a circular import: each file
needs the other before it has finished loading, and Python refuses.

One file that imports nothing, imported by everything, breaks that cycle.
"""


class ToolError(Exception):
    """A tool cannot answer the request as asked.

    Carries a message written for the ROUTER, not the user: it names the
    rejected value and, where the list is short enough, the valid ones, so
    the model can correct itself and retry. Turning that into an apology is
    the narration layer's job, adjustable in a prompt rather than hardcoded
    here.

    Raised for bad names, values that don't exist in a column, empty filter
    combinations, requests whose grain cannot be answered, and a missing
    dataset — never for a result that is merely thin or uninteresting. Those
    come back as data with a flag.
    """ 