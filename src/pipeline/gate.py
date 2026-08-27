"""
src/pipeline/gate.py
====================

Structural pre-gate — the first thing an upload meets.

    three raw CSV paths  ->  one report dict

Its only job is turning a crash into a message. Without it, a file missing
`loan_id` dies twelve steps into cleaning with a KeyError and the step-13
validation gates never run. The user sees a stack trace about a dictionary key
instead of "your loans file is missing loan_id".

SCOPE — COLUMN NAMES ONLY.
No dtypes, no values, no parsing. Both arrive dirty by design and deciding
about them is cleaning's job. In particular the 12-month panel check is NOT
here: it needs parsed dates, and at this point `month` is still a mix of
'2024-07' and 'Mar-2025' strings where no span is measurable. That check lives
in validate_transactions, after parse_month.

RETURNS, NEVER RAISES.
All three tables are checked before anything is reported, so the user fixes
every problem in one pass rather than re-uploading once per missing column. A
returned report is also what a Streamlit page can render; catching an
AssertionError to read its message is not.
"""

import pandas as pd

from src.config import (
    REQUIRED_CUSTOMER_COLS,
    OPTIONAL_CUSTOMER_COLS,
    REQUIRED_LOAN_COLS,
    OPTIONAL_LOAN_COLS,
    REQUIRED_TXN_COLS,
    OPTIONAL_TXN_COLS,
    CUSTOMERS_RAW,
    LOANS_RAW,
    TRANSACTIONS_RAW,
)

# How many padded names to show as examples. The finding is "your export pads
# its headers", which one or two cases prove; printing all eighteen buries it.
PADDED_EXAMPLES = 3


def check_table(path, required, optional):
    """Check one raw upload's column names against its contract.

    One function for all three tables — the lists arrive as arguments, so the
    knowledge that customers uses REQUIRED_CUSTOMER_COLS lives in exactly one
    place rather than in three near-identical copies.

    Returns findings rather than raising; see the module docstring.
    """
    df = pd.read_csv(path, nrows=0)
    columns_org = df.columns
    strip_cols = df.columns.str.strip()

    # Compared on the STRIPPED names, because the raw headers are padded on
    # BOTH sides: every column after the first carries the space that follows
    # its comma (' age '), and several carry trailing padding too
    # (' purpose         '). skipinitialspace=True would remove the leading
    # half and hide it — so it is deliberately not used here.
    missing = set(required) - set(strip_cols)

    # Neither required nor known-optional. Reported, never silently dropped:
    # an unrecognised name may be a RENAME of a column the system needs
    # (loan_income_ratio for inflow_to_loan_ratio), and silence hides that.
    extra = set(strip_cols) - set(required) - set(optional)

    # Per-name, not a set operation: does this name change when stripped?
    # Comparing against the other list would misjudge a file carrying both
    # 'age' and 'age ' — the padded one would appear in the stripped list.
    padded = [pad for pad in columns_org if pad != pad.strip()]

    return {
        "path": str(path),
        "n_columns": len(columns_org),
        # sorted() because sets are unordered — the same file would otherwise
        # report its problems in a different order each run.
        "missing_required": sorted(missing),
        "unrecognised": sorted(extra),
        "padded_names": padded,
        # Only a missing required column stops the pipeline. Extras and
        # padding are worth saying out loud and nothing more.
        "ok": not missing,
    }


def check_upload(customers_path, loans_path, transactions_path):
    """Check all three raw uploads against their column contracts.

    Paths are ARGUMENTS, not config reads: the orchestrator hands this whatever
    the user dropped in, which lives in an upload buffer rather than data/raw.
    """
    tables = {
        "customers":    check_table(customers_path, REQUIRED_CUSTOMER_COLS, OPTIONAL_CUSTOMER_COLS),
        "loans":        check_table(loans_path, REQUIRED_LOAN_COLS, OPTIONAL_LOAN_COLS),
        "transactions": check_table(transactions_path, REQUIRED_TXN_COLS, OPTIONAL_TXN_COLS),
    }

    return {
        # The upload passes only if every table does. Each result already
        # carries its own `ok`, so the rule lives in check_table and is not
        # re-derived here — change it there and both functions follow.
        "ok": all(t["ok"] for t in tables.values()),
        "tables": tables,
    }


def format_report(report):
    """Turn the report dict into lines a person can read.

    Separate from check_upload so the same report can feed a terminal, a
    Streamlit page, or a log without the checking logic knowing which.
    """
    lines = []

    for name, result in report["tables"].items():
        status = "PASS" if result["ok"] else "FAIL"
        lines.append(f"{name:<14} {status}   {result['n_columns']} columns")

        # The only fatal finding.
        if result["missing_required"]:
            lines.append(
                f"  missing required: {', '.join(result['missing_required'])}"
            )

        # Not fatal — but an unrecognised name may be a renamed column the
        # system needs, so it is never passed over in silence.
        if result["unrecognised"]:
            lines.append(
                f"  unrecognised: {', '.join(result['unrecognised'])}"
            )

        # Count first, examples second. repr() keeps the quotes, which is the
        # whole point — 'age ' and 'age' are indistinguishable without them.
        if result["padded_names"]:
            padded = result["padded_names"]
            shown = ", ".join(repr(p) for p in padded[:PADDED_EXAMPLES])
            lines.append(
                f"  {len(padded)} of {result['n_columns']} headers padded "
                f"— e.g. {shown}"
            )

    lines.append("")
    lines.append(f"upload: {'PASS' if report['ok'] else 'FAIL'}")
    return lines


def main():
    """Run the gate on the reference files, so the module is testable alone.

    Same pattern as the cleaning modules: the FUNCTION takes paths so it works
    on any upload, and main() supplies the reference ones.
    """
    report = check_upload(CUSTOMERS_RAW, LOANS_RAW, TRANSACTIONS_RAW)
    for line in format_report(report):
        print(line)
    return report


if __name__ == "__main__":
    main()   