"""
src/pipeline/orchestrator.py
============================

The upload path, end to end.

    three raw CSV paths  ->  three feature frames + one report

Everything this calls already existed. What did not exist was a single entry
point that runs them in the right order with the right values threaded through
— and the ORDER is the real content here, not the plumbing.

WHY CUSTOMERS CANNOT GO FIRST
-----------------------------
recover_tenure needs the as-of anchor, and the anchor is derived from the loans
and transactions dates. Those dates are only comparable AFTER parsing: before
that they are mixed-format strings where max() sorts alphabetically and
'30/03/2025' beats '2025-06-01'. So loans and transactions clean first, the
anchor comes out of them, and customers cleans against it.

THE BASELINE IS REQUIRED
------------------------
Band edges come from models/baseline.json, not from the upload. Recomputing
them per file is what made Q1 cover different credit scores in January than in
February, so an upload without a baseline cannot produce comparable bands at
all. Run `python -m src.models.baseline` after training.

EVERY UPLOAD IS FINGERPRINTED
-----------------------------
The three files are hashed before anything else touches them, so the dataset
that comes out has an identity. That is what lets an answer say which file it
describes, and lets a re-upload of an unchanged file be recognised rather than
silently rebuilt.

TRAINING vs SCORING
-------------------
This is the SCORING path. It never splits, never trains. Each module's main()
remains the training entry point and is untouched.

PARQUETS DO GET WRITTEN NOW
---------------------------
Earlier version said "returned, never written — parquets on disk belong to the
training path". That was true until _DATA started needing to survive a
Streamlit re-import. Now every upload also lands three parquets under
data/uploads/<fingerprint>/, and load_dataset() is handed their paths so its
pointer file can find them again on the next import.

Training parquets still belong to the training path. These are different files
in a different folder for a different reason — session persistence, not model
provenance — and they are safe to delete when the upload is no longer wanted.

RETURNS ONE SHAPE ALWAYS
------------------------
Success and failure return the same keys, with the frames None on failure. The
caller writes one branch, and a user is always told what actually happened
rather than receiving a different object with no explanation.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from src.config import (
    CUSTOMERS_RAW, LOANS_RAW, TRANSACTIONS_RAW, AS_OF_DATE,
)
from src.cleaning.customers import clean_customers, validate as validate_customers
from src.cleaning.loans import clean_loans, validate_loans
from src.cleaning.transactions import clean_transactions, validate_transactions
from src.features.churn import build_churn_features
from src.features.default import build_default_features
from src.features.segments import build_segment_features
from src.pipeline.drift import check_drift, load_baseline
from src.pipeline.gate import check_upload
from src.pipeline.reconcile import reconcile

# Files are read in fixed-size pieces rather than whole, so the fingerprint
# does not need the entire upload in memory at once.
_HASH_CHUNK = 65536

# Where per-upload parquets live. One folder per fingerprint so re-uploads of
# the same file map to the same folder and never duplicate.
_UPLOADS_DIR = Path("data") / "uploads"


def fingerprint_sources(customers_path, loans_path, transactions_path):
    """One hash identifying these three files together.

    CONTENT, not filename. A company exporting daily writes to the same three
    names every morning, so the name says nothing about which day it is. The
    bytes do.

    All three go into one hash because they are one upload — a file where only
    the loans changed is a different dataset, and must not be mistaken for the
    one already loaded.

    Returns a short hex string. sha256 truncated to 16 characters: long enough
    that two real uploads will not collide, short enough to show a user.
    """
    digest = hashlib.sha256()

    for path in (customers_path, loans_path, transactions_path):
        # The path itself is not hashed — only the bytes. The same data saved
        # under a new name is the same data.
        with open(Path(path), "rb") as handle:
            while chunk := handle.read(_HASH_CHUNK):
                digest.update(chunk)

    return digest.hexdigest()[:16]


def derive_anchor(loans, transactions):
    """The file's freeze date: the latest date anything in it refers to.

    Both tables, not one — a company could send transactions through June and
    loans through May, and the anchor is the later of the two.

    MUST run after the date parsing in clean_loans and clean_transactions. On
    raw strings max() would sort alphabetically and return nonsense.

    NOTE: on the reference extract this returns 2025-06-01, one month before
    config.AS_OF_DATE. That constant was reverse-engineered from onboarding +
    stored tenure rather than observed, so the two legitimately differ. Pass
    `as_of` explicitly to reproduce the original build.
    """
    return max(loans["disbursed_date"].max(), transactions["month"].max())


def derive_churn_panel(transactions):
    """The three churn window dates, from the months actually present.

    Derived from the sorted month VALUES rather than by date arithmetic, for
    the same reason the panel-length check counts distinct months: a file can
    span a year and still be missing one from the middle.

    The rule, read off the reference constants (Jul 2024 .. Jun 2025):
      panel_start  months[0]   = 2024-07  first month of the panel
      window_end   months[5]   = 2024-12  last month used as a FEATURE, so the
                                          feature window is the first half of
                                          the panel and the label belongs to
                                          the second half
      half_split   months[3]   = 2024-10  first month of the second half OF THE
                                          FEATURE WINDOW — the midpoint of
                                          months 1-6, not of the whole panel

    half_split has to be that midpoint or `difference` compares two unequal
    spans, and a longer half reads as growth that never happened.
    """
    months = sorted(transactions["month"].unique())
    n = len(months)

    panel_start = months[0]
    window_end = months[n // 2 - 1]
    half_split = months[n // 4]

    return panel_start, window_end, half_split


def band_edges_for(baseline, table):
    """This table's saved quartile edges, widened for use.

    MOMENT B. The builders have always accepted band_edges and forwarded them
    to add_bands; what never existed was anything to pass. Now the artifact
    holds them, so Q1 means the same credit-score range in every upload.

    The artifact stores the REAL outer bounds, because infinity is not valid
    JSON. Widening happens here, on use — without it a credit score above the
    training maximum falls outside every bin and becomes NaN, which is exactly
    the case the bands need to survive.
    """
    edges = {}

    for col, cuts in baseline[table]["band_edges"].items():
        widened = list(cuts)
        widened[0], widened[-1] = float("-inf"), float("inf")
        edges[col] = widened

    return edges


def _persist_features(features, fingerprint):
    """Write the three feature frames to disk under this upload's folder.

    One folder per fingerprint — data/uploads/<fp>/{default,churn,segment}.parquet.
    A re-upload of the same file lands the same folder and overwrites cleanly,
    with no duplicate row growth.

    Returned as a paths dict so load_dataset() can persist a pointer to them.
    The paths are absolute strings so a fresh import from any working directory
    still finds them.
    """
    folder = _UPLOADS_DIR / fingerprint
    folder.mkdir(parents=True, exist_ok=True)

    paths = {}
    for name, frame in features.items():
        path = folder / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        paths[name] = str(path.resolve())

    return paths


def _empty_result(stage, report, fingerprint=None, label=None):
    """Failure shape — identical keys to success, frames set to None.

    The fingerprint and label survive a failure on purpose: a user whose
    upload was rejected still needs to know which upload it was.
    """
    return {
        "ok": False,
        "failed_at": stage,
        "report": report,
        "features": None,
        "paths": None,
        "as_of": None,
        "fingerprint": fingerprint,
        "label": label,
    }


def run_pipeline(customers_path, loans_path, transactions_path, as_of=None):
    """Raw uploads to feature tables.

    Parameters
    ----------
    customers_path, loans_path, transactions_path
        Wherever the upload landed. Arguments rather than config reads: a real
        upload lives in Streamlit's buffer or a temp folder, not data/raw.
    as_of
        Override the derived anchor. None means derive it from the file, which
        is what an upload does. Pass a value only to reproduce a build whose
        anchor is known independently of the data.

    Returns
    -------
    dict with the same keys either way — ok, failed_at, report, features,
    paths, as_of, fingerprint, label. `features` holds the three frames on
    success and None on failure; `paths` holds the parquets they were written
    to, for load_dataset() to persist as a pointer.
    """
    report = {}

    # -- 0. identity --------------------------------------------------------
    # Before anything else, because it needs only the paths — and because a
    # file rejected at the gate still has to be identifiable in the report.
    fingerprint = fingerprint_sources(
        customers_path, loans_path, transactions_path
    )
    label = Path(customers_path).name
    report["fingerprint"] = fingerprint
    report["label"] = label

    # -- 1. structural pre-gate -------------------------------------------
    # Column names only. Turns "KeyError: 'loan_id' twelve steps into
    # cleaning" into "your loans file is missing loan_id".
    gate_report = check_upload(customers_path, loans_path, transactions_path)
    report["gate"] = gate_report
    if not gate_report["ok"]:
        return _empty_result("pre-gate", report, fingerprint, label)

    # Loaded once, used twice — band edges for the feature build, and the
    # distributions drift compares against. Raises if it is missing, which is
    # correct: without it the bands are not comparable across uploads.
    baseline = load_baseline()

    # -- 2. clean the two date-bearing tables ------------------------------
    # These must come first: their parsed dates are what the anchor is
    # derived from. Neither uses the anchor itself — only their validate
    # gates do, further down.
    loans = clean_loans(loans_path)
    transactions = clean_transactions(transactions_path)

    # -- 3. derive the anchor ---------------------------------------------
    # Computed ONCE and threaded everywhere. Derived per cleaner instead,
    # loans and customers could disagree about when the file was frozen.
    anchor = pd.Timestamp(as_of) if as_of is not None else derive_anchor(loans, transactions)
    report["as_of"] = str(anchor)

    # -- 4. clean customers, against that anchor ---------------------------
    # Loaded here rather than inside the cleaner: clean_customers takes a
    # FRAME while the other two take paths. That inconsistency is invisible
    # while each module has its own main(); the orchestrator has to absorb it.
    raw_customers = pd.read_csv(customers_path, skipinitialspace=True)
    customers = clean_customers(raw_customers, anchor)

    # -- 5. the step-13 gates ----------------------------------------------
    # Each table against its own invariants. The 12-month panel assert fires
    # inside validate_transactions.
    validate_customers(customers, anchor)
    validate_loans(loans, anchor)
    validate_transactions(transactions, anchor)

    # -- 6. cross-table reconciliation --------------------------------------
    # POLICY: orphan loans and ratio mismatches are KEPT, not dropped. The
    # rows are real and analytics answers correctly on them; what they cannot
    # have is a trustworthy score. Refusing the prediction while keeping the
    # row is the same principle as withholding p_value in compare_groups —
    # one number is unavailable, everything else still works.
    #
    # TODO(tier2): reconcile returns counts and 10-key samples only.
    # score_population needs the FULL unscorable loan_id list.
    recon_report = reconcile(customers, loans, transactions)
    report["reconciliation"] = recon_report

    # -- 7. derive the churn window ----------------------------------------
    panel_start, window_end, half_split = derive_churn_panel(transactions)
    report["churn_panel"] = {
        "panel_start": str(panel_start),
        "window_end": str(window_end),
        "half_split": str(half_split),
    }

    # -- 8. build the three feature tables ----------------------------------
    # Band edges come from the BASELINE now, not from this file. Each table
    # gets its own set, because each was fitted on a different population —
    # 6,394 loans, 11,760 labelled customers, 12,000 customers.
    default_features, _ = build_default_features(
        customers, loans, transactions,
        panel_start=panel_start,
        band_edges=band_edges_for(baseline, "default"),
    )

    # drop_unlabeled=False is what makes a scoring upload survive: a live
    # customer has no churn label, and dropping on its absence would discard
    # every row you wanted to score.
    churn_features, _ = build_churn_features(
        customers, transactions,
        panel_start=panel_start,
        window_end=window_end,
        half_split=half_split,
        drop_unlabeled=False,
        band_edges=band_edges_for(baseline, "churn"),
    )

    segment_features, _ = build_segment_features(
        customers, transactions,
        band_edges=band_edges_for(baseline, "segment"),
    )

    features = {
        "default": default_features,
        "churn": churn_features,
        "segment": segment_features,
    }

    # -- 9. drift ------------------------------------------------------------
    # After the feature build, because the baseline describes FEATURE tables —
    # months_available and active_ratio do not exist before it runs.
    #
    # Reports, never blocks. A high PSI says look at this column; it cannot say
    # whether the cause is a business change, a data bug, or a promotion.
    report["drift"] = {
        name: check_drift(frame, name, baseline)
        for name, frame in features.items()
    }

    # -- 10. persist to disk so the tools survive a re-import ----------------
    # The frames returned in memory work today; the parquets written here work
    # tomorrow, after Streamlit has torn the process down and rebuilt it.
    # load_dataset() gets the paths and writes a pointer alongside them so the
    # next fresh import finds this same upload.
    paths = _persist_features(features, fingerprint)

    return {
        "ok": True,
        "failed_at": None,
        "report": report,
        "features": features,
        "paths": paths,
        "as_of": anchor,
        "fingerprint": fingerprint,
        "label": label,
    }


def main():
    """Run the upload path on the reference raw files, standalone."""
    result = run_pipeline(CUSTOMERS_RAW, LOANS_RAW, TRANSACTIONS_RAW)

    if not result["ok"]:
        print(f"FAILED at: {result['failed_at']}")
        from src.pipeline.gate import format_report
        for line in format_report(result["report"]["gate"]):
            print(line)
        return result

    report = result["report"]
    print(f"dataset:       {result['label']}  [{result['fingerprint']}]")
    print(f"as-of anchor:  {report['as_of']}")
    print(f"churn panel:   {report['churn_panel']['panel_start']} .. "
          f"{report['churn_panel']['window_end']}  "
          f"(split {report['churn_panel']['half_split']})")

    recon = report["reconciliation"]
    print(f"reconciliation: {'PASS' if recon['ok'] else 'FLAGGED'}")
    for name, check in recon["checks"].items():
        if not check["ok"]:
            print(f"  {name}: flagged")

    print("\nfeature tables:")
    for name, frame in result["features"].items():
        print(f"  {name:<10} {frame.shape[0]:,} rows x {frame.shape[1]} cols")

    print("\ndrift:")
    for name, drift_report in report["drift"].items():
        status = "STABLE" if drift_report["ok"] else "FLAGGED"
        print(f"  {name:<10} {status}")
        for col in drift_report["flagged"]:
            entry = drift_report["columns"][col]
            print(f"      {col:<24} PSI {entry['psi']:.4f}  {entry['band']}")

    return result


if __name__ == "__main__":
    main() 