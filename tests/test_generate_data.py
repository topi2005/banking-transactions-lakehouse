"""The synthetic data generator: reproducibility, injected-problem accounting, incremental drops."""
import json

import pandas as pd
import pytest

import generate_data


def _read_all(folder):
    return pd.concat([pd.read_csv(f, dtype=str) for f in sorted(folder.glob("*.csv"))], ignore_index=True)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp("data")
    generate_data.main(["--rows", "3000", "--customers", "300", "--merchants", "40", "--out", str(out), "--seed", "7"])
    return out


def test_full_load_writes_one_folder_per_table(dataset):
    for name in ["customers", "accounts", "merchants", "transactions"]:
        assert any((dataset / name).glob("*.csv")), name
    assert (dataset / "dirty_data_manifest.json").exists()
    assert len(list((dataset / "transactions").glob("*.csv"))) == 3          # split to mimic daily drops


def test_manifest_matches_what_was_written(dataset):
    m = json.loads((dataset / "dirty_data_manifest.json").read_text())
    tx = _read_all(dataset / "transactions")
    assert len(tx) == m["total_rows_written"] == m["clean_rows_before_dups"] + m["duplicate_rows"]
    assert tx["transaction_id"].duplicated().sum() == m["duplicate_rows"]
    assert m["customers"] == 300 and m["merchants"] == 40


def test_dirty_data_is_actually_injected(dataset):
    tx = _read_all(dataset / "transactions")
    assert tx["amount"].isna().any() and tx["merchant_id"].isna().any() and tx["channel"].isna().any()
    assert tx["amount"].dropna().str.startswith("R").any()                   # text amounts
    assert (tx["account_id"] == "A99999999").any()                           # orphan keys


def test_same_seed_gives_identical_files(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for out in (a, b):
        generate_data.main(["--rows", "500", "--customers", "50", "--merchants", "10", "--out", str(out), "--seed", "3"])
    assert (a / "transactions" / "transactions_part1.csv").read_bytes() == (b / "transactions" / "transactions_part1.csv").read_bytes()


def test_incremental_batch_adds_new_redelivered_and_fixed_rows(dataset):
    before = _read_all(dataset / "transactions")
    generate_data.main(["--incremental", "--out", str(dataset), "--seed", "7", "--new-rows", "400",
                        "--redeliver", "50", "--resubmit-fixed", "10", "--batch-name", "batchX"])
    batch = pd.read_csv(dataset / "transactions" / "transactions_batchX.csv", dtype=str)
    m = json.loads((dataset / "incremental_manifest_batchX.json").read_text())

    old_max = before["transaction_id"].str[1:].astype(int).max()
    seen = batch["transaction_id"].isin(before["transaction_id"])
    assert m["first_new_transaction_id"] == f"T{old_max + 1:010d}"
    assert seen.sum() == m["redelivered_rows"] + m["resubmitted_fixes"]      # only these reuse earlier ids
    new_ids = batch.loc[~seen, "transaction_id"].str[1:].astype(int)
    assert new_ids.min() > old_max                                           # fresh ids continue the sequence

    # fixes: earlier rows that had NO amount and now have one (re-delivered rows keep their original amount)
    original_amount = before.drop_duplicates("transaction_id").set_index("transaction_id")["amount"]
    reused = batch[seen]
    healed = reused[reused["amount"].notna() & reused["transaction_id"].map(original_amount).isna()]
    assert len(healed) == m["resubmitted_fixes"]
