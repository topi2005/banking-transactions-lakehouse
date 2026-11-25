"""
The silver transactions cleaning step re-implemented in pandas, rule for rule, for the benchmark.

It follows src/silver_transforms.py exactly (same parsing, same dedupe, same quarantine reasons). The benchmark
refuses to report timings unless both engines produce the same silver and quarantine row counts,
so the comparison is like for like.
"""
import numpy as np
import pandas as pd

VALID_CHANNELS = ["POS", "ATM", "ONLINE", "APP", "EFT"]
MAX_PLAUSIBLE_AMOUNT = 50_000
NUMERIC = r"^-?\d+(\.\d+)?$"


def _blank_to_null(s: pd.Series) -> pd.Series:
    s = s.str.strip()
    return s.mask(s == "")


def clean_transactions(raw: pd.DataFrame, valid_account_ids: set):
    """Return (silver, quarantine, duplicates_removed) for a DataFrame read with dtype=str."""
    df = raw.copy()
    df["_transaction_id"] = _blank_to_null(df["transaction_id"])
    df["_account_id"] = _blank_to_null(df["account_id"])

    # deduplicate: keep the first row per non-null id; rows with a NULL id are all kept
    has_id = df["_transaction_id"].notna()
    keep = ~(has_id & df.duplicated("_transaction_id", keep="first"))
    n_dups = int((~keep).sum())
    df = df[keep]

    # amount: 'R153,62' -> 153.62 ; anything that is not a number -> NaN
    amount_raw_blank = _blank_to_null(df["amount"]).isna()
    cleaned = (_blank_to_null(df["amount"]).str.replace(r"^R\s*", "", regex=True).str.replace(",", ".", regex=False))
    amount = pd.to_numeric(cleaned.where(cleaned.str.match(NUMERIC, na=False)), errors="coerce")

    # timestamp: two known formats, strict, day-first
    ts_text = _blank_to_null(df["transaction_ts"])
    ts = pd.to_datetime(ts_text, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    ts = ts.fillna(pd.to_datetime(ts_text, format="%d/%m/%Y %H:%M", errors="coerce"))
    ts_raw_blank = ts_text.isna()

    channel = _blank_to_null(df["channel"]).str.upper()
    merchant = _blank_to_null(df["merchant_id"])
    is_refund = _blank_to_null(df["is_refund"]).str.lower().eq("true")

    bad = (
        df["_transaction_id"].isna()
        | amount_raw_blank
        | (~amount_raw_blank & amount.isna())
        | ts_raw_blank
        | (~ts_raw_blank & ts.isna())
        | ~df["_account_id"].isin(valid_account_ids)
        | (amount.notna() & ((amount < 0) != is_refund))
    )

    silver = pd.DataFrame({
        "transaction_id": df["_transaction_id"], "account_id": df["_account_id"],
        "merchant_id": merchant.fillna("UNKNOWN"), "transaction_ts": ts,
        "transaction_date": ts.dt.date, "amount": amount,
        "currency": _blank_to_null(df["currency"]).str.upper(),
        "channel": channel.where(channel.isin(VALID_CHANNELS), "UNKNOWN"),
        "is_refund": is_refund,
        "channel_imputed": ~channel.isin(VALID_CHANNELS),
        "merchant_imputed": merchant.isna(),
        "is_amount_outlier": amount.abs() > MAX_PLAUSIBLE_AMOUNT,
    })[~bad]
    quarantine = raw.loc[df.index[bad]]
    return silver, quarantine, n_dups
