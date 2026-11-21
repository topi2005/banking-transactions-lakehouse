"""
Synthetic retail-banking data generator for the banking-lakehouse project.

Produces four CSV files with deliberately injected data-quality problems so the
silver layer and quality framework have real work to do:

    <out>/customers/customers.csv, <out>/accounts/accounts.csv,
    <out>/merchants/merchants.csv, <out>/transactions/transactions_part{1..N}.csv

It also writes dirty_data_manifest.json: the exact counts of what was injected.
Those counts are approximate ground truth for your before/after quality metrics:
injections can overlap on the same row, so verify with your own checks too.

Usage (terminal):
    python generate_data.py --rows 500000 --out ./data --seed 42
Usage (Databricks notebook, file imported next to the notebook):
    import generate_data
    generate_data.main(["--out", "/Volumes/workspace/banking_lakehouse/landing/raw"])
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROVINCES = ["Gauteng", "Western Cape", "KwaZulu-Natal", "Eastern Cape",
             "Free State", "Limpopo", "Mpumalanga", "North West", "Northern Cape"]
PROVINCE_P = [0.26, 0.12, 0.18, 0.12, 0.05, 0.10, 0.08, 0.05, 0.04]
SEGMENTS = ["Retail", "Premium", "Student", "Business"]
SEGMENT_P = [0.60, 0.15, 0.15, 0.10]
ACCOUNT_TYPES = ["Cheque", "Savings", "Credit Card"]
CHANNELS = ["POS", "ATM", "ONLINE", "APP", "EFT"]
CATEGORIES = ["Groceries", "Fuel", "Dining", "Retail", "Utilities",
              "Travel", "Entertainment", "Health", "Education", "Transfer"]
FIRST = ["Thabo", "Lerato", "Sipho", "Naledi", "Ayanda", "Zanele", "Kagiso",
         "Nomsa", "Tshepo", "Palesa", "Bongani", "Lindiwe", "Pieter", "Aisha",
         "Michael", "Sarah", "Themba", "Refilwe", "Johan", "Priya"]
LAST = ["Dlamini", "Nkosi", "Mokoena", "Khumalo", "Naidoo", "van der Merwe",
        "Botha", "Sithole", "Molefe", "Pillay", "Zulu", "Mahlangu", "Ndlovu",
        "Smith", "Mthembu", "Kumalo", "Venter", "Maseko", "Moyo", "Jacobs"]


def make_customers(rng, n):
    df = pd.DataFrame({
        "customer_id": [f"C{i:07d}" for i in range(1, n + 1)],
        "first_name": rng.choice(FIRST, n),
        "last_name": rng.choice(LAST, n),
        "province": rng.choice(PROVINCES, n, p=PROVINCE_P),
        "segment": rng.choice(SEGMENTS, n, p=SEGMENT_P),
        "date_of_birth": (
            pd.Timestamp("2005-01-01")
            - pd.to_timedelta(rng.integers(0, 365 * 55, n), unit="D")
        ).strftime("%Y-%m-%d"),
        "onboarded_date": (
            pd.Timestamp("2026-06-30")
            - pd.to_timedelta(rng.integers(0, 365 * 12, n), unit="D")
        ).strftime("%Y-%m-%d"),
    })
    df["email"] = (df.first_name.str.lower() + "." +
                   df.last_name.str.lower().str.replace(" ", "") +
                   df.customer_id.str[-4:] + "@example.com")
    return df


def make_accounts(rng, customers):
    rows = []
    n_acc = 0
    for cid in customers.customer_id:
        for _ in range(rng.choice([1, 2, 3], p=[0.55, 0.35, 0.10])):
            n_acc += 1
            rows.append((f"A{n_acc:08d}", cid, rng.choice(ACCOUNT_TYPES, p=[0.5, 0.3, 0.2])))
    acc = pd.DataFrame(rows, columns=["account_id", "customer_id", "account_type"])
    acc["opened_date"] = (
        pd.Timestamp("2026-06-30")
        - pd.to_timedelta(rng.integers(0, 365 * 10, len(acc)), unit="D")
    ).strftime("%Y-%m-%d")
    acc["status"] = rng.choice(["ACTIVE", "DORMANT", "CLOSED"], len(acc), p=[0.88, 0.08, 0.04])
    return acc


def make_merchants(rng, n):
    return pd.DataFrame({
        "merchant_id": [f"M{i:05d}" for i in range(1, n + 1)],
        "merchant_name": [f"Merchant {i}" for i in range(1, n + 1)],
        "category": rng.choice(CATEGORIES[:-1], n),
        "province": rng.choice(PROVINCES, n, p=PROVINCE_P),
    })


def make_transactions(rng, accounts, merchants, n):
    acc_idx = rng.integers(0, len(accounts), n)
    mer_idx = rng.integers(0, len(merchants), n)
    amount = np.round(rng.lognormal(mean=5.0, sigma=1.1, size=n), 2)  # median ~R150
    ts = (pd.Timestamp("2026-01-01")
          + pd.to_timedelta(rng.integers(0, 181 * 86400, n), unit="s"))  # Jan-Jun 2026
    is_refund = rng.random(n) < 0.02
    tx = pd.DataFrame({
        "transaction_id": [f"T{i:010d}" for i in range(1, n + 1)],
        "account_id": accounts.account_id.to_numpy()[acc_idx],
        "merchant_id": merchants.merchant_id.to_numpy()[mer_idx],
        "transaction_ts": ts,
        "amount": np.where(is_refund, -amount, amount),
        "currency": "ZAR",
        "channel": rng.choice(CHANNELS, n, p=[0.40, 0.10, 0.20, 0.20, 0.10]),
        "is_refund": is_refund,
    })
    return tx


def dirty_transactions(rng, tx, accounts, rate_null=0.08, rate_dup=0.02):
    """Inject problems; return dirty frame + manifest of what was injected."""
    n = len(tx)
    m = {}
    tx = tx.copy()
    tx["transaction_ts"] = tx["transaction_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    tx["amount"] = tx["amount"].astype(object)
    tx["is_refund"] = tx["is_refund"].astype(str).str.lower()

    def pick(frac):
        return rng.choice(n, int(n * frac), replace=False)

    # 1. Nulls in key business columns
    for col in ["merchant_id", "channel", "amount"]:
        idx = pick(rate_null / 3)
        tx.loc[tx.index[idx], col] = None
        m[f"null_{col}"] = int(len(idx))

    # 2. Mixed date formats (parsable but inconsistent)
    idx = pick(0.03)
    parsed = pd.to_datetime(tx.loc[tx.index[idx], "transaction_ts"])
    tx.loc[tx.index[idx], "transaction_ts"] = parsed.dt.strftime("%d/%m/%Y %H:%M")
    m["alt_date_format"] = int(len(idx))

    # 3. Unparseable / impossible dates
    idx = pick(0.005)
    tx.loc[tx.index[idx], "transaction_ts"] = rng.choice(
        ["31/02/2026 10:00", "not_a_date", "2026-13-45 99:99:99"], len(idx))
    m["bad_date"] = int(len(idx))

    # 4. Dirty channel strings (case / whitespace)
    idx = pick(0.05)
    tx.loc[tx.index[idx], "channel"] = (
        tx.loc[tx.index[idx], "channel"].fillna("pos").str.lower().radd(" ") + " ")
    m["dirty_channel_text"] = int(len(idx))

    # 5. Amount as text with currency symbol / comma decimal
    idx = pick(0.02)
    sub = tx.loc[tx.index[idx], "amount"]
    tx.loc[tx.index[idx], "amount"] = [
        "R" + str(a).replace(".", ",") if a is not None and a == a else a for a in sub]
    m["amount_text_format"] = int(len(idx))

    # 6. Orphan foreign keys (account that does not exist)
    idx = pick(0.004)
    tx.loc[tx.index[idx], "account_id"] = "A99999999"
    m["orphan_account_id"] = int(len(idx))

    # 7. Outliers (amount x1000)
    idx = pick(0.001)
    for i in idx:
        v = tx.iloc[i]["amount"]
        try:
            tx.iloc[i, tx.columns.get_loc("amount")] = round(float(v) * 1000, 2)
        except (TypeError, ValueError):
            pass
    m["amount_outlier_attempted"] = int(len(idx))

    # 8. Exact duplicate rows (same transaction_id)
    dup = tx.sample(int(n * rate_dup), random_state=int(rng.integers(0, 10_000)))
    tx = pd.concat([tx, dup], ignore_index=True).sample(frac=1, random_state=1).reset_index(drop=True)
    m["duplicate_rows"] = int(len(dup))
    m["total_rows_written"] = int(len(tx))
    m["clean_rows_before_dups"] = int(n)
    return tx, m


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=500_000)
    ap.add_argument("--customers", type=int, default=20_000)
    ap.add_argument("--merchants", type=int, default=1_500)
    ap.add_argument("--out", default="./data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", type=int, default=3,
                    help="split transactions into N files, to simulate daily drops")
    a = ap.parse_args(argv)

    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    customers = make_customers(rng, a.customers)
    accounts = make_accounts(rng, customers)
    merchants = make_merchants(rng, a.merchants)
    tx_clean = make_transactions(rng, accounts, merchants, a.rows)
    tx_dirty, manifest = dirty_transactions(rng, tx_clean, accounts)

    # One folder per table: bronze ingests every table the same way (<out>/<table>/*.csv)
    for name, frame in [("customers", customers), ("accounts", accounts), ("merchants", merchants)]:
        (out / name).mkdir(exist_ok=True)
        frame.to_csv(out / name / f"{name}.csv", index=False)

    # Split transactions into several files: bronze should record source file per row
    tx_dir = out / "transactions"
    tx_dir.mkdir(exist_ok=True)
    bounds = np.linspace(0, len(tx_dirty), a.split + 1, dtype=int)
    for i in range(a.split):
        part = tx_dirty.iloc[bounds[i]:bounds[i + 1]]
        part.to_csv(tx_dir / f"transactions_part{i + 1}.csv", index=False)

    manifest.update({"customers": len(customers), "accounts": len(accounts),
                     "merchants": len(merchants), "seed": a.seed})
    (out / "dirty_data_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
