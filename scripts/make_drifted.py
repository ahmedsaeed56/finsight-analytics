"""
scripts/make_drifted.py
=======================

A test file with KNOWN drift in exactly one column.

    customers_raw.csv  ->  customers_drifted.csv   (credit_score + 40)

Not a random second dataset. The point is ground truth: one column moved by a
known amount, everything else untouched, so the detector can be judged rather
than just observed. A test that flags credit_score and leaves the rest stable
proves it works; one that flags six columns proves the binning is broken —
which is exactly what caught the equal-shares bug.

    python -m scripts.make_drifted
"""

import pandas as pd

from src.config import DATA_RAW, CUSTOMERS_RAW

CREDIT_SHIFT = 40
OUT = DATA_RAW / "customers_drifted.csv"


def main():
    df = pd.read_csv(CUSTOMERS_RAW, skipinitialspace=True)
    df.columns = df.columns.str.strip()

    before = df["credit_score"].mean()
    df["credit_score"] = df["credit_score"] + CREDIT_SHIFT
    after = df["credit_score"].mean()

    df.to_csv(OUT, index=False)

    print(f"wrote {OUT}")
    print(f"  credit_score mean {before:.1f} -> {after:.1f}  (+{CREDIT_SHIFT})")
    print(f"  every other column unchanged")


if __name__ == "__main__":
    main() 