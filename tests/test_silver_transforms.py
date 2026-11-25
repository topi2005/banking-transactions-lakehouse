"""Silver layer: parsing helpers, deduplication, validation rules, quarantine/imputation split."""
from decimal import Decimal

import pytest
from pyspark.sql import functions as F

import silver_transforms as st
from conftest import make_bronze


# ---------------------------------------------------------------- parsing helpers
def _eval(spark, values, fn):
    df = spark.createDataFrame([(v,) for v in values], "v string")
    return [r["out"] for r in df.select(fn(F.col("v")).alias("out")).collect()]


def test_parse_amount_handles_every_known_format(spark):
    out = _eval(spark, ["R153,62", " 99.5 ", "R-20,00", "-7", "150"], st.parse_amount)
    assert out == [Decimal("153.62"), Decimal("99.50"), Decimal("-20.00"), Decimal("-7.00"), Decimal("150.00")]


@pytest.mark.parametrize("bad", ["abc", "", "   ", "R", "1,2,3", "12abc", None])
def test_parse_amount_returns_null_never_raises(spark, bad):
    # ANSI mode is on in the test session: a naive cast would raise here
    assert _eval(spark, [bad], st.parse_amount) == [None]


def _parsed_as_text(spark, values):
    """
    Parse, then format back to text INSIDE Spark. Collecting a timestamp into Python would convert it to the
    machine's local time zone (a UTC+2 laptop would see 12:30 instead of 10:30), so we compare text instead.
    """
    df = spark.createDataFrame([(v,) for v in values], "v string")
    out = df.select(F.date_format(st.parse_timestamp(F.col("v")), "yyyy-MM-dd HH:mm").alias("out"))
    return [r["out"] for r in out.collect()]


def test_parse_timestamp_accepts_both_formats_day_first(spark):
    out = _parsed_as_text(spark, ["2026-03-05 10:30:00", "05/03/2026 10:30"])
    assert out == ["2026-03-05 10:30", "2026-03-05 10:30"]   # dd/MM/yyyy, so 5 March, not 3 May


def test_parse_timestamp_is_day_first_for_ambiguous_dates(spark):
    assert _parsed_as_text(spark, ["03/04/2026 08:00"]) == ["2026-04-03 08:00"]   # 3 April, not 4 March


@pytest.mark.parametrize("bad", ["31/02/2026 10:00", "not_a_date", "2026-13-45 99:99:99", "", None])
def test_parse_timestamp_returns_null_for_impossible_values(spark, bad):
    assert _eval(spark, [bad], st.parse_timestamp) == [None]


def test_blank_to_null_trims_and_nulls_empty_strings(spark):
    assert _eval(spark, ["  x  ", "", "   ", None], st.blank_to_null) == ["x", None, None, None]


# ---------------------------------------------------------------- standardise and deduplicate
def test_standardise_normalises_channel_and_refund_flag(spark):
    df = make_bronze(spark, [dict(channel="  pos ", is_refund=" TRUE "), dict(channel="", is_refund="false")])
    rows = st.standardise_transactions(df).orderBy("transaction_id").collect()
    assert rows[0]["_channel"] == "POS" and rows[0]["_is_refund"] is True
    assert rows[1]["_channel"] is None and rows[1]["_is_refund"] is False


def test_standardise_keeps_raw_columns_untouched(spark):
    df = make_bronze(spark, [dict(amount="R153,62", channel=" pos ")])
    row = st.standardise_transactions(df).first()
    assert row["amount"] == "R153,62" and row["channel"] == " pos "


def test_deduplicate_keeps_first_ingested_row(spark):
    df = make_bronze(spark, [
        dict(transaction_id="T1", amount="1.00", _ingested_at="2026-11-02 00:00:00", _source_file="late.csv"),
        dict(transaction_id="T1", amount="2.00", _ingested_at="2026-11-01 00:00:00", _source_file="early.csv"),
        dict(transaction_id="T2"),
    ])
    out = {r["transaction_id"]: r["amount"] for r in st.deduplicate(st.standardise_transactions(df)).collect()}
    assert out == {"T1": "2.00", "T2": "150.00"}


def test_deduplicate_does_not_collapse_rows_with_null_ids(spark):
    df = make_bronze(spark, [dict(transaction_id=None), dict(transaction_id=None), dict(transaction_id="T1")])
    assert st.deduplicate(st.standardise_transactions(df)).count() == 3


# ---------------------------------------------------------------- validation rules
def _reasons(spark, accounts, **overrides):
    df = make_bronze(spark, [overrides])
    checked = st.apply_transaction_rules(st.standardise_transactions(df), accounts)
    return sorted(checked.first()["reject_reasons"])


def test_valid_row_has_no_reject_reasons(spark, accounts):
    assert _reasons(spark, accounts) == []


@pytest.mark.parametrize("overrides, expected", [
    (dict(transaction_id=None), ["MISSING_TRANSACTION_ID"]),
    (dict(amount=None), ["MISSING_AMOUNT"]),
    (dict(amount="   "), ["MISSING_AMOUNT"]),
    (dict(amount="abc"), ["INVALID_AMOUNT"]),
    (dict(transaction_ts=None), ["MISSING_TIMESTAMP"]),
    (dict(transaction_ts="31/02/2026 10:00"), ["INVALID_TIMESTAMP"]),
    (dict(account_id="A99"), ["ORPHAN_ACCOUNT_ID"]),
    (dict(account_id=None), ["ORPHAN_ACCOUNT_ID"]),
    (dict(amount="-5.00", is_refund="false"), ["REFUND_SIGN_MISMATCH"]),
    (dict(amount="5.00", is_refund="true"), ["REFUND_SIGN_MISMATCH"]),
])
def test_each_rule_fires_with_its_own_reason(spark, accounts, overrides, expected):
    assert _reasons(spark, accounts, **overrides) == expected


def test_legitimate_refund_passes(spark, accounts):
    assert _reasons(spark, accounts, amount="-25.50", is_refund="true") == []


def test_a_row_can_fail_several_rules_at_once(spark, accounts):
    assert _reasons(spark, accounts, amount=None, account_id="A99") == ["MISSING_AMOUNT", "ORPHAN_ACCOUNT_ID"]


def test_repaired_formats_pass_validation(spark, accounts):
    # text amount and day-first date are repaired, not rejected
    assert _reasons(spark, accounts, amount="R153,62", transaction_ts="05/03/2026 10:30") == []


# ---------------------------------------------------------------- split into silver / quarantine
def _split(spark, accounts, rows, **kw):
    checked = st.apply_transaction_rules(st.standardise_transactions(make_bronze(spark, rows)), accounts)
    return st.split_valid_and_quarantine(checked, **kw)


def test_split_row_counts_add_up(spark, accounts):
    rows = [dict(transaction_id="T1"), dict(transaction_id="T2", amount=None), dict(transaction_id="T3", account_id="A9")]
    silver, quarantine = _split(spark, accounts, rows)
    assert silver.count() == 1 and quarantine.count() == 2


def test_quarantine_keeps_raw_values_and_reasons(spark, accounts):
    _, quarantine = _split(spark, accounts, [dict(transaction_id="T2", amount=None, channel=" pos ")])
    row = quarantine.first()
    assert row["amount"] is None and row["channel"] == " pos "
    assert list(row["reject_reasons"]) == ["MISSING_AMOUNT"]
    assert "_batch_id" in quarantine.columns and "_quarantined_at" in quarantine.columns


def test_imputation_is_explicit_and_flagged(spark, accounts):
    rows = [dict(transaction_id="T1", merchant_id=None, channel=None), dict(transaction_id="T2", channel="XYZ")]
    silver, _ = _split(spark, accounts, rows)
    out = {r["transaction_id"]: r for r in silver.collect()}
    assert out["T1"]["merchant_id"] == "UNKNOWN" and out["T1"]["merchant_imputed"] is True
    assert out["T1"]["channel"] == "UNKNOWN" and out["T1"]["channel_imputed"] is True
    assert out["T2"]["channel"] == "UNKNOWN" and out["T2"]["channel_imputed"] is True   # unrecognised value
    assert out["T2"]["merchant_imputed"] is False


def test_valid_channel_is_not_flagged_as_imputed(spark, accounts):
    silver, _ = _split(spark, accounts, [dict(channel="pos")])
    row = silver.first()
    assert row["channel"] == "POS" and row["channel_imputed"] is False


def test_outlier_flag_applies_only_strictly_above_the_limit(spark, accounts):
    rows = [dict(transaction_id="T1", amount="50000.00"), dict(transaction_id="T2", amount="50000.01"),
            dict(transaction_id="T3", amount="-50000.01", is_refund="true")]
    silver, _ = _split(spark, accounts, rows)
    flags = {r["transaction_id"]: r["is_amount_outlier"] for r in silver.collect()}
    assert flags == {"T1": False, "T2": True, "T3": True}      # flagged, but kept in silver


def test_silver_output_is_typed(spark, accounts):
    silver, _ = _split(spark, accounts, [dict()])
    types = dict(silver.dtypes)
    assert types["amount"] == "decimal(18,2)" and types["transaction_ts"] == "timestamp"
    assert types["transaction_date"] == "date" and types["is_refund"] == "boolean"


# ---------------------------------------------------------------- dimensions
def test_clean_dimension_trims_types_uppercases_and_dedupes(spark):
    df = spark.createDataFrame(
        [("A1", " active ", "2020-01-31", "2026-11-01 00:00:00"),
         ("A1", "closed", "2020-01-31", "2026-11-02 00:00:00"),
         (None, "active", "2020-01-31", "2026-11-01 00:00:00")],
        "account_id string, status string, opened_date string, _ingested_at string",
    ).withColumn("_ingested_at", F.col("_ingested_at").cast("timestamp"))
    out = st.clean_dimension(df, "account_id", date_cols=["opened_date"], upper_cols=["status"])
    rows = out.collect()
    assert len(rows) == 1                                   # duplicate collapsed, null key dropped
    assert rows[0]["status"] == "ACTIVE"                    # first ingested wins, trimmed, upper-cased
    assert dict(out.dtypes)["opened_date"] == "date"