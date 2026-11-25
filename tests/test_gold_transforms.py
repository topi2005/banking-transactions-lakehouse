"""Gold layer: surrogate keys, unknown members, dimensions, fact grain, aggregates."""
from datetime import date, datetime
from decimal import Decimal

import pytest
from pyspark.sql import functions as F

import gold_transforms as gt


# ---------------------------------------------------------------- keys
def test_key_or_unknown_maps_null_and_sentinel_to_minus_one(spark):
    df = spark.createDataFrame([("UNKNOWN",), (None,), ("M1",), ("M2",), ("M1",)], "v string")
    keys = [r["k"] for r in df.select(gt.key_or_unknown("v").alias("k")).collect()]
    assert keys[0] == gt.UNKNOWN_KEY and keys[1] == gt.UNKNOWN_KEY
    assert keys[2] == keys[4]                       # deterministic: same input, same key
    assert keys[2] != keys[3]                       # different inputs, different keys
    assert keys[2] != gt.UNKNOWN_KEY


# ---------------------------------------------------------------- dimensions
def test_dim_date_is_inclusive_and_derives_calendar_attributes(spark):
    d = gt.build_dim_date(spark, date(2026, 1, 1), date(2026, 1, 31))
    rows = {r["date"]: r for r in d.collect()}
    assert len(rows) == 31
    jan1, jan3, jan31 = rows[date(2026, 1, 1)], rows[date(2026, 1, 3)], rows[date(2026, 1, 31)]
    assert jan1["date_key"] == 20260101 and jan1["day_name"] == "Thursday" and jan1["iso_day_of_week"] == 4
    assert jan1["is_weekend"] is False and jan3["is_weekend"] is True      # 3 Jan 2026 is a Saturday
    assert jan31["is_month_end"] is True and jan1["is_month_end"] is False
    assert jan1["quarter"] == 1 and jan1["month_name"] == "January"


def test_dim_date_accepts_string_bounds(spark):
    assert gt.build_dim_date(spark, "2026-06-29", "2026-07-02").count() == 4


def test_dim_customer_age_bands_and_pii_minimisation(spark):
    customers = spark.createDataFrame(
        [("C1", "Thabo", "Dlamini", "t@x.com", "Gauteng", "Retail", date(2001, 7, 1), date(2020, 1, 1)),   # 24y 11m
         ("C2", "Aisha", "Naidoo", "a@x.com", "Gauteng", "Retail", date(2001, 6, 30), date(2020, 1, 1)),   # exactly 25
         ("C3", "Piet", "Botha", "p@x.com", "Gauteng", "Retail", date(1970, 1, 1), date(2020, 1, 1)),
         ("C4", "Zed", "Moyo", "z@x.com", "Gauteng", "Retail", None, date(2020, 1, 1))],
        "customer_id string, first_name string, last_name string, email string, province string, "
        "segment string, date_of_birth date, onboarded_date date")
    out = gt.build_dim_customer(customers)
    bands = {r["customer_id"]: r["age_band"] for r in out.collect()}
    assert bands == {"C1": "18-24", "C2": "25-34", "C3": "55+", "C4": "Unknown"}
    assert not {"first_name", "last_name", "email", "date_of_birth"} & set(out.columns)   # no PII in gold


def test_dim_merchant_adds_exactly_one_unknown_member(spark):
    merchants = spark.createDataFrame([("M1", "Shop 1", "Fuel", "Gauteng")],
                                      "merchant_id string, merchant_name string, category string, province string")
    out = gt.build_dim_merchant(spark, merchants)
    assert out.count() == 2
    unknown = out.filter(F.col("merchant_key") == gt.UNKNOWN_KEY).collect()
    assert len(unknown) == 1 and unknown[0]["merchant_id"] == "UNKNOWN" and unknown[0]["category"] == "Unknown"


def test_dim_channel_groups_and_unknown_member(spark):
    out = {r["channel_code"]: r for r in gt.build_dim_channel(spark).collect()}
    assert set(out) == {"POS", "ATM", "ONLINE", "APP", "EFT", "UNKNOWN"}
    assert out["APP"]["channel_group"] == "Digital" and out["ATM"]["channel_group"] == "Physical"
    assert out["EFT"]["channel_group"] == "Interbank"
    assert out["UNKNOWN"]["channel_key"] == gt.UNKNOWN_KEY
    assert len({r["channel_key"] for r in out.values()}) == 6                # keys are unique


# ---------------------------------------------------------------- fact and aggregates
SILVER_COLS = ("transaction_id string, account_id string, merchant_id string, transaction_ts timestamp, "
               "transaction_date date, amount decimal(18,2), channel string, is_refund boolean, "
               "channel_imputed boolean, merchant_imputed boolean, is_amount_outlier boolean, _batch_id string")


def _silver(spark, rows):
    return spark.createDataFrame(rows, SILVER_COLS)


def _row(tid, acc="A1", mer="M1", amount="100.00", channel="POS", refund=False, outlier=False,
         day=date(2026, 3, 5), m_imp=False):
    return (tid, acc, mer, datetime(day.year, day.month, day.day, 10, 0), day, Decimal(amount), channel,
            refund, False, m_imp, outlier, "b1")


@pytest.fixture
def accounts(spark):
    return spark.createDataFrame([("A1", "C1")], "account_id string, customer_id string")


def test_fact_has_one_row_per_silver_row_with_expected_keys(spark, accounts):
    silver = _silver(spark, [_row("T1"), _row("T2", mer="UNKNOWN", m_imp=True, channel="UNKNOWN")])
    fact = {r["transaction_id"]: r for r in gt.build_fact_transactions(silver, accounts).collect()}
    assert len(fact) == 2
    assert fact["T1"]["date_key"] == 20260305
    assert fact["T1"]["merchant_key"] != gt.UNKNOWN_KEY
    assert fact["T2"]["merchant_key"] == gt.UNKNOWN_KEY and fact["T2"]["channel_key"] == gt.UNKNOWN_KEY


def test_fact_uses_unknown_customer_when_account_is_missing_instead_of_dropping_the_row(spark, accounts):
    silver = _silver(spark, [_row("T1", acc="A1"), _row("T2", acc="A_MISSING")])
    fact = {r["transaction_id"]: r["customer_key"] for r in gt.build_fact_transactions(silver, accounts).collect()}
    assert fact["T1"] != gt.UNKNOWN_KEY and fact["T2"] == gt.UNKNOWN_KEY


def test_fact_keys_match_dimension_keys(spark, accounts):
    silver = _silver(spark, [_row("T1")])
    fact = gt.build_fact_transactions(silver, accounts).first()
    cust = gt.build_dim_customer(spark.createDataFrame(
        [("C1", "p", "Retail", date(1990, 1, 1), date(2020, 1, 1))],
        "customer_id string, province string, segment string, date_of_birth date, onboarded_date date")).first()
    assert fact["customer_key"] == cust["customer_key"]


def test_aggregates_exclude_outliers_but_count_them(spark, accounts):
    silver = _silver(spark, [
        _row("T1", amount="100.00"),
        _row("T2", amount="50.00"),
        _row("T3", amount="-30.00", refund=True),
        _row("T4", amount="90000.00", outlier=True),
    ])
    fact = gt.build_fact_transactions(silver, accounts)
    customers = gt.build_dim_customer(spark.createDataFrame(
        [("C1", "Gauteng", "Retail", date(1990, 1, 1), date(2020, 1, 1))],
        "customer_id string, province string, segment string, date_of_birth date, onboarded_date date"))
    dates = gt.build_dim_date(spark, date(2026, 3, 5), date(2026, 3, 5))

    agg = gt.build_agg_daily_segment(fact, customers, dates).first()
    assert agg["txn_count"] == 3 and agg["outlier_txn_count"] == 1
    assert agg["txn_count"] + agg["outlier_txn_count"] == fact.count()      # nothing disappears
    assert agg["purchase_amount"] == Decimal("150.00")
    assert agg["refund_amount"] == Decimal("-30.00")
    assert agg["net_amount"] == Decimal("120.00")
    assert agg["avg_purchase_value"] == Decimal("75.00")
    assert agg["active_customers"] == 1


def test_monthly_category_aggregate_accounts_for_every_fact_row(spark, accounts):
    silver = _silver(spark, [_row("T1"), _row("T2", mer="UNKNOWN", m_imp=True), _row("T3", outlier=True, amount="99999.00")])
    fact = gt.build_fact_transactions(silver, accounts)
    merchants = gt.build_dim_merchant(spark, spark.createDataFrame(
        [("M1", "Shop", "Fuel", "Gauteng")], "merchant_id string, merchant_name string, category string, province string"))
    dates = gt.build_dim_date(spark, date(2026, 3, 1), date(2026, 3, 31))
    agg = gt.build_agg_monthly_category(fact, merchants, dates)
    total = agg.agg((F.sum("txn_count") + F.sum("outlier_txn_count")).alias("n")).first()["n"]
    assert total == 3
    assert "Unknown" in {r["category"] for r in agg.collect()}              # imputed merchant lands in Unknown
