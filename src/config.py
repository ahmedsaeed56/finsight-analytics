from pathlib import Path

ROOT= Path(__file__).resolve().parents[1]

DATA_RAW= ROOT / "data"/ "raw"
DATA_CLEAN= ROOT / "data"/ "clean"

CUSTOMERS_RAW    = DATA_RAW / "customers_raw.csv"
LOANS_RAW        = DATA_RAW / "loans_raw.csv"
TRANSACTIONS_RAW = DATA_RAW / "transactions_raw.csv" 