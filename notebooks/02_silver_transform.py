# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Silver transformation
# MAGIC **Goal:** turn raw bronze strings into typed, deduplicated, validated tables, and *quarantine*
# MAGIC (never silently drop) every row that fails a rule.
# MAGIC
# MAGIC | Output table | Contents |
# MAGIC |---|---|
# MAGIC | `silver_customers`, `silver_accounts`, `silver_merchants` | typed, trimmed, deduplicated dimensions |
# MAGIC | `silver_transactions` | valid rows only, typed and standardised, with imputation flags |
# MAGIC | `silver_transactions_quarantine` | rejected rows, original raw values + `reject_reasons` |
# MAGIC | `silver_run_summary` | headline metrics for this run (feeds the README and Day 3) |
# MAGIC
# MAGIC The logic lives in `src/silver_transforms.py` (pure functions, unit-tested on Day 6).
# MAGIC This notebook only orchestrates: read → transform → write → reconcile.

# COMMAND ----------

# MAGIC %md ## 0 · Config

# COMMAND ----------

import sys
sys.path.append("../src")   # repo layout: notebooks/ and src/ are siblings

from datetime import datetime, timezone
from pyspark.sql import functions as F

import silver_transforms as st

CATALOG = "workspace"
SCHEMA = "banking_lakehouse"
T = lambda name: f"{CATALOG}.{SCHEMA}.{name}"   # fully-qualified table name

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
print("Run:", RUN_ID)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · Dimensions first
# MAGIC Accounts must be cleaned *before* transactions, because the orphan check (`account_id` must exist) joins to it.

# COMMAND ----------

dim_specs = {
    "customers": dict(key="customer_id", date_cols=["date_of_birth", "onboarded_date"]),
    "accounts":  dict(key="account_id",  date_cols=["opened_date"], upper_cols=["status", "account_type"]),
    "merchants": dict(key="merchant_id"),
}

dim_counts = {}
for name, spec in dim_specs.items():
    bronze_df = spark.table(T(f"bronze_{name}"))
    silver_df = st.clean_dimension(bronze_df, **spec)
    silver_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(T(f"silver_{name}"))
    dim_counts[name] = (bronze_df.count(), spark.table(T(f"silver_{name}")).count())
    print(f"silver_{name:10s} bronze={dim_counts[name][0]:>8,}  silver={dim_counts[name][1]:>8,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · Transactions: standardise → deduplicate → validate → split
# MAGIC Order matters. We dedupe *before* validating so a duplicated bad row isn't counted as two rejections.

# COMMAND ----------

bronze_tx = spark.table(T("bronze_transactions"))
silver_accounts = spark.table(T("silver_accounts"))

standardised = st.standardise_transactions(bronze_tx)
deduped = st.deduplicate(standardised)
checked = st.apply_transaction_rules(deduped, silver_accounts)
silver_tx, quarantine = st.split_valid_and_quarantine(checked)

# COMMAND ----------

(silver_tx.write.format("delta").mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(T("silver_transactions")))
(quarantine.write.format("delta").mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(T("silver_transactions_quarantine")))

silver_tx = spark.table(T("silver_transactions"))
quarantine = spark.table(T("silver_transactions_quarantine"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 · Reconciliation: the check a bank cares about most
# MAGIC Every bronze row must be accounted for: **bronze = silver + quarantine + duplicates removed**.
# MAGIC If this fails, rows are being lost silently, and the pipeline must stop.

# COMMAND ----------

n_bronze = bronze_tx.count()
n_after_dedupe = deduped.count()
n_dups = n_bronze - n_after_dedupe
n_silver = silver_tx.count()
n_quarantine = quarantine.count()

print(f"bronze rows            {n_bronze:>10,}")
print(f"- duplicates removed   {n_dups:>10,}")
print(f"- quarantined          {n_quarantine:>10,}")
print(f"= silver rows          {n_silver:>10,}")

assert n_bronze == n_silver + n_quarantine + n_dups, "RECONCILIATION FAILED: rows are unaccounted for"
assert silver_tx.select("transaction_id").distinct().count() == n_silver, "Duplicate transaction_ids in silver"
print("\nPASS: every bronze row is accounted for, and silver transaction_id is unique")

# COMMAND ----------

# MAGIC %md ## 4 · What got rejected, and why

# COMMAND ----------

reasons = (quarantine.select(F.explode("reject_reasons").alias("reason"))
           .groupBy("reason").count().orderBy(F.desc("count")))
display(reasons)

# Look at actual rejected rows: raw values preserved exactly as they arrived
display(quarantine.limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 · Before vs after (the headline numbers)
# MAGIC "Before" is measured on the **raw bronze** data; "after" on **silver**.
# MAGIC Be precise about what changed: nulls in `amount` and `transaction_ts` are fixed by *rejecting* rows
# MAGIC (we won't guess a monetary value). Nulls in `merchant_id` / `channel` are fixed by *explicit imputation*
# MAGIC (`'UNKNOWN'` + a flag column). Say both in your README.

# COMMAND ----------

KEY_COLS = ["merchant_id", "amount", "channel", "transaction_ts"]

def null_cells(df, cols):
    row = df.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in cols]).first()
    return sum(row[c] or 0 for c in cols)

cells_before = null_cells(bronze_tx, KEY_COLS)
cells_after = null_cells(silver_tx, KEY_COLS)
null_reduction_pct = round(100 * (cells_before - cells_after) / cells_before, 2) if cells_before else 100.0

# Fixes applied (measured on the deduped, standardised frame)
fixes = deduped.agg(
    F.sum(F.col("transaction_ts").rlike(r"^\d{2}/\d{2}/\d{4}").cast("int")).alias("alt_date_format_parsed"),
    F.sum(F.col("amount").rlike(r"^R").cast("int")).alias("text_amounts_parsed"),
    F.sum((F.col("channel").isNotNull() & (F.col("channel") != F.col("_channel"))).cast("int")).alias("channel_values_normalised"),
).first()

imputed = silver_tx.agg(
    F.sum(F.col("merchant_imputed").cast("int")).alias("merchant_imputed"),
    F.sum(F.col("channel_imputed").cast("int")).alias("channel_imputed"),
    F.sum(F.col("is_amount_outlier").cast("int")).alias("amount_outliers_flagged"),
).first()

metrics = {
    "bronze_rows": n_bronze,
    "duplicates_removed": n_dups,
    "quarantined_rows": n_quarantine,
    "silver_rows": n_silver,
    "quarantine_rate_pct": round(100 * n_quarantine / n_bronze, 2),
    "null_cells_key_cols_before": cells_before,
    "null_cells_key_cols_after": cells_after,
    "null_reduction_pct": null_reduction_pct,
    "alt_date_format_parsed": fixes["alt_date_format_parsed"],
    "text_amounts_parsed": fixes["text_amounts_parsed"],
    "channel_values_normalised": fixes["channel_values_normalised"],
    "merchant_imputed_unknown": imputed["merchant_imputed"],
    "channel_imputed_unknown": imputed["channel_imputed"],
    "amount_outliers_flagged": imputed["amount_outliers_flagged"],
}
for k, v in metrics.items():
    print(f"{k:30s} {v:>12,}" if isinstance(v, int) else f"{k:30s} {v:>12}")

# COMMAND ----------

summary = spark.createDataFrame(
    [(RUN_ID, k, float(v)) for k, v in metrics.items()],
    "run_id string, metric string, value double",
).withColumn("recorded_at", F.current_timestamp())

summary.write.format("delta").mode("append").saveAsTable(T("silver_run_summary"))
display(spark.table(T("silver_run_summary")).orderBy(F.desc("recorded_at")).limit(len(metrics)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Done: Day 2 checklist
# MAGIC - [ ] 5 silver tables + `silver_transactions_quarantine` + `silver_run_summary` exist in Catalog Explorer
# MAGIC - [ ] Reconciliation prints PASS
# MAGIC - [ ] Screenshot: quarantine reasons table
# MAGIC - [ ] Screenshot: the before/after metrics printout
# MAGIC - [ ] Copy the metrics into the README "Results" table
# MAGIC - [ ] Commit and push
