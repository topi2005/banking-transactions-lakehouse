"""Shared fixtures: one local SparkSession for the whole run, and helpers to build tiny test DataFrames."""
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session")
def spark():
    from pyspark.sql import SparkSession

    session = (
        SparkSession.builder.master("local[2]").appName("banking-lakehouse-tests")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.sql.session.timeZone", "UTC")
        # Databricks serverless runs with ANSI mode on: bad casts raise instead of returning NULL.
        # Testing with it on proves the parsing helpers are genuinely safe.
        .config("spark.sql.ansi.enabled", "true")
        .getOrCreate()
    )
    yield session
    session.stop()


BRONZE_SCHEMA = (
    "transaction_id string, account_id string, merchant_id string, transaction_ts string, "
    "amount string, currency string, channel string, is_refund string, "
    "_ingested_at string, _source_file string, _batch_id string"
)

GOOD_ROW = dict(
    transaction_id="T1", account_id="A1", merchant_id="M1", transaction_ts="2026-03-05 10:30:00",
    amount="150.00", currency="ZAR", channel="POS", is_refund="false",
    _ingested_at="2026-11-01 00:00:00", _source_file="f1.csv", _batch_id="b1",
)


def make_bronze(spark, rows):
    """Build a bronze-shaped DataFrame from dicts. Missing keys fall back to a valid row's values."""
    from pyspark.sql import functions as F

    data = [tuple({**GOOD_ROW, **r}[c.split()[0]] for c in BRONZE_SCHEMA.split(", ")) for r in rows]
    return spark.createDataFrame(data, BRONZE_SCHEMA).withColumn("_ingested_at", F.col("_ingested_at").cast("timestamp"))


@pytest.fixture
def bronze():
    """Factory fixture: bronze(rows) -> DataFrame."""
    return make_bronze


@pytest.fixture
def accounts(spark):
    return spark.createDataFrame([("A1", "C1"), ("A2", "C2")], "account_id string, customer_id string")
