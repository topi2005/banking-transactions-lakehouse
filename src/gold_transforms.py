"""
Gold-layer transformations: a star schema for transaction analytics.

Grain of the fact table: ONE ROW PER TRANSACTION.

    dim_date ─────┐
    dim_customer ─┤
    dim_account ──┼── fact_transactions
    dim_merchant ─┤
    dim_channel ──┘

Design decisions (each is a talking point):
* Surrogate keys are deterministic hashes of the natural key (xxhash64). The same input always gives
  the same key, so reruns and incremental loads (Day 5) never reshuffle keys, and the fact table can
  compute its keys without joining to every dimension. Referential integrity is then *verified* by the
  data-quality rules instead of assumed.
* Unknown members: a fact row whose merchant or channel is unknown points at key -1, a real dimension
  row labelled 'Unknown'. No NULL foreign keys, no dropped rows.
* PII minimisation: dim_customer drops names, e-mail and date of birth. Analysts get an age band,
  province and segment, which is all a spending mart needs.
* Aggregates exclude rows flagged as suspected outliers, and count them separately so nothing is hidden.

Pure functions only (DataFrames in, DataFrames out) so they can be unit-tested on Day 6.
"""
from pyspark.sql import DataFrame, SparkSession, functions as F

UNKNOWN_KEY = -1
AS_OF_DATE = "2026-06-30"   # fixed reference date for age bands (end of the data window)

CHANNEL_GROUPS = {
    "POS": "Physical", "ATM": "Physical",
    "ONLINE": "Digital", "APP": "Digital",
    "EFT": "Interbank",
    "UNKNOWN": "Unknown",
}


# --------------------------------------------------------------------------- keys
def key_or_unknown(col):
    """Hash of the natural key, or -1 when the key is NULL or the 'UNKNOWN' sentinel."""
    c = F.col(col) if isinstance(col, str) else col
    return F.when(c.isNull() | (c == "UNKNOWN"), F.lit(UNKNOWN_KEY)).otherwise(F.xxhash64(c))


# --------------------------------------------------------------------------- dimensions
def build_dim_date(spark: SparkSession, start, end) -> DataFrame:
    """One row per calendar day between `start` and `end` (python dates or 'yyyy-MM-dd' strings)."""
    days = spark.range(1).select(
        F.explode(F.sequence(F.lit(start).cast("date"), F.lit(end).cast("date"),
                             F.expr("interval 1 day"))).alias("date"))
    iso_dow = ((F.dayofweek("date") + 5) % 7) + 1   # Monday=1 ... Sunday=7
    return days.select(
        F.date_format("date", "yyyyMMdd").cast("int").alias("date_key"),
        "date",
        F.year("date").alias("year"),
        F.quarter("date").alias("quarter"),
        F.month("date").alias("month"),
        F.date_format("date", "MMMM").alias("month_name"),
        F.dayofmonth("date").alias("day_of_month"),
        iso_dow.alias("iso_day_of_week"),
        F.date_format("date", "EEEE").alias("day_name"),
        F.weekofyear("date").alias("week_of_year"),
        (iso_dow >= 6).alias("is_weekend"),
        (F.col("date") == F.last_day("date")).alias("is_month_end"),
    )


def build_dim_customer(silver_customers: DataFrame) -> DataFrame:
    age = F.floor(F.months_between(F.lit(AS_OF_DATE).cast("date"), F.col("date_of_birth")) / 12)
    age_band = (F.when(age.isNull(), "Unknown")
                 .when(age < 25, "18-24").when(age < 35, "25-34")
                 .when(age < 45, "35-44").when(age < 55, "45-54")
                 .otherwise("55+"))
    return silver_customers.select(
        F.xxhash64("customer_id").alias("customer_key"),
        "customer_id", "province", "segment",
        age_band.alias("age_band"),
        "onboarded_date",
    )


def build_dim_account(silver_accounts: DataFrame) -> DataFrame:
    return silver_accounts.select(
        F.xxhash64("account_id").alias("account_key"),
        "account_id", "account_type", "status", "opened_date",
    )


def build_dim_merchant(spark: SparkSession, silver_merchants: DataFrame) -> DataFrame:
    real = silver_merchants.select(
        F.xxhash64("merchant_id").alias("merchant_key"),
        "merchant_id", "merchant_name", "category", "province")
    unknown = spark.createDataFrame(
        [(UNKNOWN_KEY, "UNKNOWN", "Unknown merchant", "Unknown", "Unknown")],
        "merchant_key long, merchant_id string, merchant_name string, category string, province string")
    return real.unionByName(unknown)


def build_dim_channel(spark: SparkSession) -> DataFrame:
    rows = [(c, g) for c, g in CHANNEL_GROUPS.items()]
    return (spark.createDataFrame(rows, "channel_code string, channel_group string")
            .select(key_or_unknown("channel_code").alias("channel_key"), "channel_code", "channel_group"))


# --------------------------------------------------------------------------- fact
def build_fact_transactions(silver_tx: DataFrame, silver_accounts: DataFrame) -> DataFrame:
    """Keys are computed from natural keys; the only join is account -> customer."""
    acc = silver_accounts.select("account_id", "customer_id")
    f = silver_tx.join(F.broadcast(acc), "account_id", "left")
    return f.select(
        "transaction_id",                                              # degenerate dimension
        F.date_format("transaction_date", "yyyyMMdd").cast("int").alias("date_key"),
        "transaction_ts",
        F.xxhash64("account_id").alias("account_key"),
        key_or_unknown("customer_id").alias("customer_key"),
        key_or_unknown("merchant_id").alias("merchant_key"),
        key_or_unknown("channel").alias("channel_key"),
        "amount", "is_refund", "is_amount_outlier", "merchant_imputed", "channel_imputed",
        "_batch_id",
    )


# --------------------------------------------------------------------------- aggregates
def _money(cond, amount):
    return F.coalesce(F.sum(F.when(cond, amount)), F.lit(0)).cast("decimal(18,2)")


def build_agg_daily_segment(fact: DataFrame, dim_customer: DataFrame, dim_date: DataFrame) -> DataFrame:
    """Daily spend by customer segment. Suspected outliers excluded and counted separately."""
    core = ~F.col("is_amount_outlier")
    purchase = core & ~F.col("is_refund")
    refund = core & F.col("is_refund")
    joined = (fact.join(dim_customer.select("customer_key", "segment"), "customer_key")
                  .join(dim_date.select("date_key", "date"), "date_key"))
    return joined.groupBy("date", "segment").agg(
        F.sum(F.when(core, 1).otherwise(0)).alias("txn_count"),
        _money(purchase, F.col("amount")).alias("purchase_amount"),
        _money(refund, F.col("amount")).alias("refund_amount"),
        _money(core, F.col("amount")).alias("net_amount"),
        F.round(F.avg(F.when(purchase, F.col("amount"))), 2).alias("avg_purchase_value"),
        F.countDistinct(F.when(core, F.col("customer_key"))).alias("active_customers"),
        F.sum(F.when(~core, 1).otherwise(0)).alias("outlier_txn_count"),
    )


def build_agg_monthly_category(fact: DataFrame, dim_merchant: DataFrame, dim_date: DataFrame) -> DataFrame:
    """Monthly spend by merchant category and merchant province."""
    core = ~F.col("is_amount_outlier")
    purchase = core & ~F.col("is_refund")
    refund = core & F.col("is_refund")
    joined = (fact.join(dim_merchant.select("merchant_key", "category",
                                            F.col("province").alias("merchant_province")), "merchant_key")
                  .join(dim_date.select("date_key", "year", "month"), "date_key"))
    return joined.groupBy("year", "month", "category", "merchant_province").agg(
        F.sum(F.when(core, 1).otherwise(0)).alias("txn_count"),
        _money(purchase, F.col("amount")).alias("purchase_amount"),
        _money(refund, F.col("amount")).alias("refund_amount"),
        _money(core, F.col("amount")).alias("net_amount"),
        F.countDistinct(F.when(core, F.col("customer_key"))).alias("active_customers"),
        F.sum(F.when(~core, 1).otherwise(0)).alias("outlier_txn_count"),
    )
