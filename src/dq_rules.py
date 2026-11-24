"""
Rule definitions for each layer. Rules are plain data: add, remove or tighten one by editing this file,
with no framework changes.

Bronze rules are *expected to fail*: they measure how dirty the raw data is (the "before").
Silver rules are expected to pass: they prove the cleaning worked (the "after") and gate the pipeline.
"""
from dq_framework import Rule
from silver_transforms import MAX_PLAUSIBLE_AMOUNT, VALID_CHANNELS

NUMERIC = r"^-?\d+(\.\d+)?$"

# --------------------------------------------------------------------------- bronze (raw strings)
BRONZE_RULES = [
    # completeness
    *[Rule(f"{c}_not_null", "transactions", "not_null", c, threshold_pct=99.9)
      for c in ["transaction_id", "account_id", "merchant_id", "transaction_ts", "amount", "channel"]],
    # uniqueness
    Rule("transaction_id_unique", "transactions", "unique", "transaction_id"),
    # validity
    Rule("channel_valid_value", "transactions", "accepted_values", "channel",
         {"values": VALID_CHANNELS}, threshold_pct=99.9),
    Rule("amount_is_numeric", "transactions", "regex", "amount", {"pattern": NUMERIC}, threshold_pct=99.9),
    Rule("transaction_ts_iso_parseable", "transactions", "sql_expression", "transaction_ts",
         {"fail_sql": "transaction_ts IS NOT NULL AND "
                      "try_to_timestamp(transaction_ts, 'yyyy-MM-dd HH:mm:ss') IS NULL"},
         threshold_pct=99.9),
    # referential integrity
    Rule("account_id_exists", "transactions", "foreign_key", "account_id",
         {"ref_table": "accounts", "ref_column": "account_id"}, threshold_pct=99.9),
]

# --------------------------------------------------------------------------- silver (typed, clean)
SILVER_RULES = [
    # transactions: completeness
    *[Rule(f"{c}_not_null", "transactions", "not_null", c)
      for c in ["transaction_id", "account_id", "merchant_id", "transaction_ts",
                "amount", "channel", "currency", "is_refund"]],
    # transactions: uniqueness
    Rule("transaction_id_unique", "transactions", "unique", "transaction_id"),
    # transactions: validity
    Rule("channel_valid_value", "transactions", "accepted_values", "channel",
         {"values": VALID_CHANNELS + ["UNKNOWN"]}),
    Rule("currency_is_zar", "transactions", "accepted_values", "currency", {"values": ["ZAR"]}),
    Rule("transaction_date_in_window", "transactions", "range", "transaction_date",
         {"min": "2026-01-01", "max": "2026-12-31"}),   # plausible window for this dataset (the year)
    Rule("refund_sign_consistent", "transactions", "sql_expression", None,
         {"fail_sql": "(amount < 0) <> is_refund"}),
    # outliers are flagged and kept on purpose, so they warn rather than block
    Rule("amount_within_plausible_range", "transactions", "range", "amount",
         {"min": -MAX_PLAUSIBLE_AMOUNT, "max": MAX_PLAUSIBLE_AMOUNT},
         threshold_pct=99.9, severity="warn"),
    # transactions: referential integrity ('UNKNOWN' is the documented imputation sentinel)
    Rule("account_id_exists", "transactions", "foreign_key", "account_id",
         {"ref_table": "accounts", "ref_column": "account_id"}),
    Rule("merchant_id_exists", "transactions", "foreign_key", "merchant_id",
         {"ref_table": "merchants", "ref_column": "merchant_id", "allow_values": ["UNKNOWN"]}),

    # dimensions
    Rule("customer_id_not_null", "customers", "not_null", "customer_id"),
    Rule("customer_id_unique", "customers", "unique", "customer_id"),
    Rule("account_id_not_null", "accounts", "not_null", "account_id"),
    Rule("account_id_unique", "accounts", "unique", "account_id"),
    Rule("account_customer_exists", "accounts", "foreign_key", "customer_id",
         {"ref_table": "customers", "ref_column": "customer_id"}),
    Rule("account_status_valid", "accounts", "accepted_values", "status",
         {"values": ["ACTIVE", "DORMANT", "CLOSED"]}),
    Rule("account_type_valid", "accounts", "accepted_values", "account_type",
         {"values": ["CHEQUE", "SAVINGS", "CREDIT CARD"]}),
    Rule("merchant_id_unique", "merchants", "unique", "merchant_id"),
    Rule("merchant_category_not_null", "merchants", "not_null", "category"),
]

# --------------------------------------------------------------------------- gold (star schema)
# Logical table names: fact, dim_customer, dim_account, dim_merchant, dim_channel, dim_date
GOLD_RULES = [
    # fact: keys present and unique grain
    Rule("transaction_id_not_null", "fact", "not_null", "transaction_id"),
    Rule("transaction_id_unique", "fact", "unique", "transaction_id"),
    *[Rule(f"{k}_not_null", "fact", "not_null", k)
      for k in ["date_key", "account_key", "customer_key", "merchant_key", "channel_key", "amount"]],
    # fact -> dimension referential integrity (keys are hashes, so this is VERIFIED, not assumed)
    Rule("date_key_exists", "fact", "foreign_key", "date_key",
         {"ref_table": "dim_date", "ref_column": "date_key"}),
    Rule("account_key_exists", "fact", "foreign_key", "account_key",
         {"ref_table": "dim_account", "ref_column": "account_key"}),
    Rule("customer_key_exists", "fact", "foreign_key", "customer_key",
         {"ref_table": "dim_customer", "ref_column": "customer_key"}),
    Rule("merchant_key_exists", "fact", "foreign_key", "merchant_key",
         {"ref_table": "dim_merchant", "ref_column": "merchant_key"}),
    Rule("channel_key_exists", "fact", "foreign_key", "channel_key",
         {"ref_table": "dim_channel", "ref_column": "channel_key"}),
    Rule("refund_sign_consistent", "fact", "sql_expression", None,
         {"fail_sql": "(amount < 0) <> is_refund"}),
    # dimensions: surrogate keys must be unique (guards against hash collisions)
    Rule("customer_key_unique", "dim_customer", "unique", "customer_key"),
    Rule("customer_key_not_null", "dim_customer", "not_null", "customer_key"),
    Rule("customer_age_band_valid", "dim_customer", "accepted_values", "age_band",
         {"values": ["18-24", "25-34", "35-44", "45-54", "55+", "Unknown"]}),
    Rule("account_key_unique", "dim_account", "unique", "account_key"),
    Rule("merchant_key_unique", "dim_merchant", "unique", "merchant_key"),
    Rule("channel_key_unique", "dim_channel", "unique", "channel_key"),
    Rule("date_key_unique", "dim_date", "unique", "date_key"),
]
