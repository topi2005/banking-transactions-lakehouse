"""
Silver-layer transformations for the banking-lakehouse project.

Every function takes a DataFrame and returns a DataFrame: no reads, no writes, no globals.
That keeps the logic unit-testable (pytest + a local SparkSession, see Day 6) and lets the
notebook stay a thin orchestration layer.

Design rules
------------
* Raw bronze columns are never modified, so quarantined rows still show exactly what arrived.
* Parsing is *safe*: bad input becomes NULL (then a named rejection reason), never an exception.
  This matters on Databricks serverless, where ANSI mode is on and bad casts raise errors.
* Every rejected row gets one or more machine-readable reasons.
* Imputation is explicit: filled values are 'UNKNOWN' plus a boolean flag, never silent.
"""
from pyspark.sql import DataFrame, Window, functions as F

VALID_CHANNELS = ["POS", "ATM", "ONLINE", "APP", "EFT"]

# South African convention: day first. 'dd/MM/yyyy' is assumed, never 'MM/dd/yyyy'.
TS_FORMATS = ["yyyy-MM-dd HH:mm:ss", "dd/MM/yyyy HH:mm"]

MAX_PLAUSIBLE_AMOUNT = 50_000  # ZAR; above this we flag (not reject) as a suspected outlier

TX_RAW_COLS = ["transaction_id", "account_id", "merchant_id", "transaction_ts",
               "amount", "currency", "channel", "is_refund"]
LINEAGE_COLS = ["_ingested_at", "_source_file", "_batch_id"]


# --------------------------------------------------------------------------- helpers
def blank_to_null(col):
    """Trim a string column and turn empty strings into NULL."""
    trimmed = F.trim(col)
    return F.when(trimmed == "", F.lit(None)).otherwise(trimmed)


def parse_amount(col):
    """'R153,62' -> 153.62, ' 99.5 ' -> 99.50, 'R-20,00' -> -20.00. Invalid -> NULL (never an error)."""
    cleaned = F.regexp_replace(blank_to_null(col), r"^R\s*", "")
    cleaned = F.regexp_replace(cleaned, ",", ".")
    is_number = cleaned.rlike(r"^-?\d+(\.\d+)?$")
    return F.when(is_number, cleaned.cast("decimal(18,2)"))


def parse_timestamp(col):
    """Try each known format in turn. Impossible dates (31/02/2026) and junk -> NULL."""
    cleaned = blank_to_null(col)
    return F.coalesce(*[F.try_to_timestamp(cleaned, F.lit(fmt)) for fmt in TS_FORMATS])


# --------------------------------------------------------------------------- transactions
def standardise_transactions(bronze: DataFrame) -> DataFrame:
    """Add parsed/cleaned helper columns (prefixed with _) next to the untouched raw columns."""
    return (
        bronze
        .withColumn("_amount", parse_amount(F.col("amount")))
        .withColumn("_ts", parse_timestamp(F.col("transaction_ts")))
        .withColumn("_channel", F.upper(blank_to_null(F.col("channel"))))
        .withColumn("_merchant_id", blank_to_null(F.col("merchant_id")))
        .withColumn("_transaction_id", blank_to_null(F.col("transaction_id")))
        .withColumn("_account_id", blank_to_null(F.col("account_id")))
        .withColumn("_is_refund", F.lower(blank_to_null(F.col("is_refund"))) == F.lit("true"))
    )


def deduplicate(df: DataFrame, key: str = "_transaction_id") -> DataFrame:
    """
    Keep the first-ingested row per key. Rows with a NULL key are kept (not collapsed together)
    so the validation step can quarantine them individually.
    """
    w = Window.partitionBy(key).orderBy(F.col("_ingested_at").asc(), F.col("_source_file").asc())
    keyed = (df.filter(F.col(key).isNotNull())
               .withColumn("_rn", F.row_number().over(w))
               .filter(F.col("_rn") == 1)
               .drop("_rn"))
    unkeyed = df.filter(F.col(key).isNull())
    return keyed.unionByName(unkeyed)


def apply_transaction_rules(std: DataFrame, valid_accounts: DataFrame) -> DataFrame:
    """Attach `reject_reasons` (array<string>); empty array means the row is valid."""
    accounts = (valid_accounts.select(F.col("account_id").alias("_acct_key"))
                .distinct().withColumn("_acct_found", F.lit(True)))
    joined = std.join(F.broadcast(accounts), std["_account_id"] == accounts["_acct_key"], "left")

    raw_amount_blank = blank_to_null(F.col("amount")).isNull()
    raw_ts_blank = blank_to_null(F.col("transaction_ts")).isNull()

    candidate_reasons = F.array(
        F.when(F.col("_transaction_id").isNull(), F.lit("MISSING_TRANSACTION_ID")),
        F.when(raw_amount_blank, F.lit("MISSING_AMOUNT")),
        F.when(~raw_amount_blank & F.col("_amount").isNull(), F.lit("INVALID_AMOUNT")),
        F.when(raw_ts_blank, F.lit("MISSING_TIMESTAMP")),
        F.when(~raw_ts_blank & F.col("_ts").isNull(), F.lit("INVALID_TIMESTAMP")),
        F.when(F.col("_acct_found").isNull(), F.lit("ORPHAN_ACCOUNT_ID")),
        F.when(
            F.col("_amount").isNotNull()
            & ((F.col("_amount") < 0) != F.col("_is_refund")),
            F.lit("REFUND_SIGN_MISMATCH"),
        ),
    )
    return (joined
            .withColumn("reject_reasons", F.filter(candidate_reasons, lambda r: r.isNotNull()))
            .drop("_acct_key", "_acct_found"))


def split_valid_and_quarantine(checked: DataFrame, max_amount: float = MAX_PLAUSIBLE_AMOUNT):
    """Return (silver_transactions, quarantine). Quarantine keeps the RAW values plus reasons."""
    is_bad = F.size("reject_reasons") > 0

    quarantine = (checked.filter(is_bad)
                  .select(*TX_RAW_COLS, "reject_reasons", *LINEAGE_COLS)
                  .withColumn("_quarantined_at", F.current_timestamp()))

    silver = (
        checked.filter(~is_bad)
        .withColumn("channel_imputed",
                    F.col("_channel").isNull() | ~F.col("_channel").isin(VALID_CHANNELS))
        .withColumn("merchant_imputed", F.col("_merchant_id").isNull())
        .select(
            F.col("_transaction_id").alias("transaction_id"),
            F.col("_account_id").alias("account_id"),
            F.coalesce(F.col("_merchant_id"), F.lit("UNKNOWN")).alias("merchant_id"),
            F.col("_ts").alias("transaction_ts"),
            F.to_date(F.col("_ts")).alias("transaction_date"),
            F.col("_amount").alias("amount"),
            F.upper(blank_to_null(F.col("currency"))).alias("currency"),
            F.when(F.col("_channel").isin(VALID_CHANNELS), F.col("_channel"))
             .otherwise(F.lit("UNKNOWN")).alias("channel"),
            F.col("_is_refund").alias("is_refund"),
            "channel_imputed",
            "merchant_imputed",
            (F.abs(F.col("_amount")) > F.lit(max_amount)).alias("is_amount_outlier"),
            *LINEAGE_COLS,
        )
        .withColumn("_processed_at", F.current_timestamp())
    )
    return silver, quarantine


# --------------------------------------------------------------------------- dimensions
def clean_dimension(df: DataFrame, key: str, date_cols=(), upper_cols=()) -> DataFrame:
    """Trim all strings, type the date columns, upper-case selected columns, dedupe on the key."""
    out = df
    for c, t in df.dtypes:
        if t == "string" and c not in LINEAGE_COLS:
            out = out.withColumn(c, blank_to_null(F.col(c)))
    for c in date_cols:
        out = out.withColumn(c, F.try_to_timestamp(F.col(c), F.lit("yyyy-MM-dd")).cast("date"))
    for c in upper_cols:
        out = out.withColumn(c, F.upper(F.col(c)))
    w = Window.partitionBy(key).orderBy(F.col("_ingested_at").asc())
    return (out.filter(F.col(key).isNotNull())
               .withColumn("_rn", F.row_number().over(w))
               .filter("_rn = 1").drop("_rn")
               .withColumn("_processed_at", F.current_timestamp()))
