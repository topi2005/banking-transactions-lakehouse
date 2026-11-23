# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Gold layer: star schema and reporting marts
# MAGIC **Goal:** model the clean silver data for analytics. One fact table (one row per transaction), five
# MAGIC conformed dimensions, and two pre-aggregated reporting tables, all verified before anyone queries them.
# MAGIC
# MAGIC | Table | Grain |
# MAGIC |---|---|
# MAGIC | `fact_transactions` | one row per transaction |
# MAGIC | `dim_date`, `dim_customer`, `dim_account`, `dim_merchant`, `dim_channel` | one row per member |
# MAGIC | `agg_daily_segment` | day × customer segment |
# MAGIC | `agg_monthly_category` | month × merchant category × merchant province |
# MAGIC
# MAGIC Logic lives in `src/gold_transforms.py`. Verification reuses the Day 3 framework (`GOLD_RULES`) plus
# MAGIC **control-total reconciliation** against silver, which is the check finance teams actually ask for.

# COMMAND ----------

# MAGIC %md ## 0 · Config

# COMMAND ----------

import sys
sys.path.append("../src")   # repo layout: notebooks/ and src/ are siblings

from datetime import datetime, timezone
from pyspark.sql import functions as F

import dq_framework as dq
import gold_transforms as gt
from dq_rules import GOLD_RULES

CATALOG = "workspace"
SCHEMA = "banking_lakehouse"
T = lambda name: f"{CATALOG}.{SCHEMA}.{name}"

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
print("Run:", RUN_ID)

def write_table(df, name):
    (df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(T(name)))
    n = spark.table(T(name)).count()
    print(f"{name:24s} {n:>10,} rows")
    return n

# COMMAND ----------

# MAGIC %md ## 1 · Read silver

# COMMAND ----------

silver_tx = spark.table(T("silver_transactions"))
silver_accounts = spark.table(T("silver_accounts"))
silver_customers = spark.table(T("silver_customers"))
silver_merchants = spark.table(T("silver_merchants"))

bounds = silver_tx.agg(F.min("transaction_date").alias("lo"), F.max("transaction_date").alias("hi")).first()
print("Transaction date range:", bounds["lo"], "→", bounds["hi"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · Dimensions
# MAGIC Two details worth knowing for an interview:
# MAGIC * **Deterministic surrogate keys** (hash of the natural key), so reruns and `MERGE` loads never reshuffle keys.
# MAGIC * **Unknown member (`-1`)** in `dim_merchant` and `dim_channel`, so imputed rows keep a valid foreign key.
# MAGIC * `dim_customer` is **PII-minimised**: no names, e-mail or date of birth, only an age band.

# COMMAND ----------

dim_date = gt.build_dim_date(spark, bounds["lo"], bounds["hi"])
dim_customer = gt.build_dim_customer(silver_customers)
dim_account = gt.build_dim_account(silver_accounts)
dim_merchant = gt.build_dim_merchant(spark, silver_merchants)
dim_channel = gt.build_dim_channel(spark)

counts = {
    "dim_date": write_table(dim_date, "dim_date"),
    "dim_customer": write_table(dim_customer, "dim_customer"),
    "dim_account": write_table(dim_account, "dim_account"),
    "dim_merchant": write_table(dim_merchant, "dim_merchant"),
    "dim_channel": write_table(dim_channel, "dim_channel"),
}

# COMMAND ----------

# MAGIC %md ## 3 · Fact table

# COMMAND ----------

fact = gt.build_fact_transactions(silver_tx, silver_accounts)
counts["fact_transactions"] = write_table(fact, "fact_transactions")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 · Verify the model before building on it
# MAGIC Foreign keys are hashes, so nothing in the engine guarantees they match a dimension row. We **verify** it.

# COMMAND ----------

gold_tables = {
    "fact": spark.table(T("fact_transactions")),
    "dim_customer": spark.table(T("dim_customer")),
    "dim_account": spark.table(T("dim_account")),
    "dim_merchant": spark.table(T("dim_merchant")),
    "dim_channel": spark.table(T("dim_channel")),
    "dim_date": spark.table(T("dim_date")),
}
gold_results = dq.run_rules(gold_tables, GOLD_RULES, RUN_ID, "gold")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Control-total reconciliation
# MAGIC Row counts *and* money must tie back to silver. Amounts are compared in cents (integers) to avoid float noise.

# COMMAND ----------

def cents(df, col="amount"):
    return int(df.agg(F.sum(col)).first()[0] * 100)

fact_df = gold_tables["fact"]
recon = [
    dq.equality_check(spark, RUN_ID, "gold", "fact", "fact_rows_equal_silver_rows",
                      expected=silver_tx.count(), actual=fact_df.count()),
    dq.equality_check(spark, RUN_ID, "gold", "fact", "fact_amount_cents_equal_silver_amount_cents",
                      expected=cents(silver_tx), actual=cents(fact_df)),
    dq.equality_check(spark, RUN_ID, "gold", "fact", "unknown_merchant_rows_equal_imputed_rows",
                      expected=silver_tx.filter("merchant_imputed").count(),
                      actual=fact_df.filter(F.col("merchant_key") == gt.UNKNOWN_KEY).count()),
]
for r in recon:
    gold_results = gold_results.unionByName(r)
gold_results = gold_results.cache()

display(gold_results.select("table_name", "rule_name", "kind", "failed_rows", "total_rows",
                            "pass_rate_pct", "severity", "status")
        .orderBy(F.desc("failed_rows"), "table_name"))

# COMMAND ----------

gold_results.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(T("dq_results"))
dq.assert_no_failures(gold_results)
print(f"PASS: all {gold_results.count()} gold checks met their thresholds, and control totals reconcile with silver")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 · Reporting aggregates
# MAGIC Suspected outliers are **excluded from the aggregates and counted separately** (`outlier_txn_count`), so
# MAGIC `txn_count + outlier_txn_count` always equals the fact row count. Nothing disappears silently.

# COMMAND ----------

agg_daily = gt.build_agg_daily_segment(fact_df, gold_tables["dim_customer"], gold_tables["dim_date"])
agg_monthly = gt.build_agg_monthly_category(fact_df, gold_tables["dim_merchant"], gold_tables["dim_date"])
write_table(agg_daily, "agg_daily_segment")
write_table(agg_monthly, "agg_monthly_category")

# Aggregates must account for every fact row
for name in ["agg_daily_segment", "agg_monthly_category"]:
    total = spark.table(T(name)).agg((F.sum("txn_count") + F.sum("outlier_txn_count")).alias("n")).first()["n"]
    assert total == counts["fact_transactions"], f"{name} does not account for every fact row ({total:,})"
print("PASS: both aggregates account for every fact row")

# COMMAND ----------

# MAGIC %md ## 6 · Document the tables (shows up in Catalog Explorer)

# COMMAND ----------

comments = {
    "fact_transactions": "Grain: one row per transaction. Keys are deterministic hashes; -1 = unknown member. Source: silver_transactions.",
    "dim_date": "One row per calendar day in the data window.",
    "dim_customer": "PII-minimised customer attributes: province, segment, age band. No names, e-mail or date of birth.",
    "dim_account": "Account attributes: type, status, opened date.",
    "dim_merchant": "Merchant attributes. Row with key -1 is the Unknown member for imputed merchants.",
    "dim_channel": "Channel codes grouped as Physical / Digital / Interbank. Key -1 = Unknown.",
    "agg_daily_segment": "Daily spend by customer segment. Suspected outliers excluded and counted in outlier_txn_count.",
    "agg_monthly_category": "Monthly spend by merchant category and merchant province. Suspected outliers excluded.",
}
for table, text in comments.items():
    safe = text.replace("\\", "\\\\").replace("'", "\\'")   # escape backslashes and single quotes for the SQL literal
    spark.sql(f"COMMENT ON TABLE {T(table)} IS '{safe}'")
print("Table comments set")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7 · The star schema in action
# MAGIC A typical business question: *which merchant categories drive the most net spend, by customer segment?*
# MAGIC The `fact` joins to `dim_customer` and `dim_merchant` on surrogate keys.

# COMMAND ----------

display(spark.sql(f"""
    SELECT c.segment,
           m.category,
           COUNT(*)                          AS txns,
           ROUND(SUM(f.amount), 2)           AS net_amount
    FROM {T('fact_transactions')} f
    JOIN {T('dim_customer')} c ON f.customer_key = c.customer_key
    JOIN {T('dim_merchant')} m ON f.merchant_key = m.merchant_key
    WHERE NOT f.is_amount_outlier
    GROUP BY c.segment, m.category
    ORDER BY net_amount DESC
    LIMIT 15
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Done: Day 4 checklist
# MAGIC - [ ] 8 gold tables exist in Catalog Explorer, with comments
# MAGIC - [ ] All gold checks PASS, including both control-total reconciliations
# MAGIC - [ ] Screenshot: gold results table
# MAGIC - [ ] Screenshot: the segment × category query
# MAGIC - [ ] Screenshot: *Catalog → fact_transactions → Lineage* (should show silver tables feeding it)
# MAGIC - [ ] Copy the numbers into the README, commit and push