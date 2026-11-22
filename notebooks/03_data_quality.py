# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Data-quality framework
# MAGIC **Goal:** measure quality with explicit, versioned rules instead of eyeballing tables, store every
# MAGIC result in a Delta table, and use it as a gate that can stop the pipeline.
# MAGIC
# MAGIC * Rules live in `src/dq_rules.py` as **data**; the engine is `src/dq_framework.py`.
# MAGIC * The **same engine** runs on bronze (expected to fail, which is the "before") and silver (expected to pass, which is the "after").
# MAGIC * Results are appended to `dq_results`, so quality becomes a trend you can query, not a one-off printout.
# MAGIC * `error` rules block the pipeline; `warn` rules only report.

# COMMAND ----------

# MAGIC %md ## 0 · Config

# COMMAND ----------

import sys
sys.path.append("../src")   # repo layout: notebooks/ and src/ are siblings

from datetime import datetime, timezone
from pyspark.sql import functions as F

import dq_framework as dq
from dq_rules import BRONZE_RULES, SILVER_RULES

CATALOG = "workspace"
SCHEMA = "banking_lakehouse"
T = lambda name: f"{CATALOG}.{SCHEMA}.{name}"

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
print("Run:", RUN_ID, "| bronze rules:", len(BRONZE_RULES), "| silver rules:", len(SILVER_RULES))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · Bronze: measure how dirty the raw data is
# MAGIC These are *expected* to fail. Nothing is gated here. This is the baseline for every "before" number.

# COMMAND ----------

bronze_tables = {n: spark.table(T(f"bronze_{n}")) for n in ["transactions", "accounts", "customers", "merchants"]}
bronze_results = dq.run_rules(bronze_tables, BRONZE_RULES, RUN_ID, "bronze").cache()

display(bronze_results.select("table_name", "rule_name", "kind", "failed_rows", "total_rows",
                              "pass_rate_pct", "threshold_pct", "status").orderBy("pass_rate_pct"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · Silver: prove the cleaning worked

# COMMAND ----------

silver_tables = {n: spark.table(T(f"silver_{n}")) for n in ["transactions", "accounts", "customers", "merchants"]}
silver_results = dq.run_rules(silver_tables, SILVER_RULES, RUN_ID, "silver")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Reconciliation as a quality rule
# MAGIC The Day 2 invariant, now recorded in the results table: `bronze = silver + quarantine + duplicates removed`.

# COMMAND ----------

n_bronze = spark.table(T("bronze_transactions")).count()
n_silver = spark.table(T("silver_transactions")).count()
n_quarantine = spark.table(T("silver_transactions_quarantine")).count()

# duplicates removed = rows in bronze minus distinct transaction_ids (every duplicate row shares an id)
n_dups = n_bronze - spark.table(T("bronze_transactions")).select("transaction_id").distinct().count()

recon = dq.equality_check(
    spark, RUN_ID, "silver", "transactions",
    "bronze_equals_silver_plus_quarantine_plus_duplicates",
    expected=n_bronze, actual=n_silver + n_quarantine + n_dups,
)
silver_results = silver_results.unionByName(recon).cache()

display(silver_results.select("table_name", "rule_name", "kind", "failed_rows", "total_rows",
                              "pass_rate_pct", "threshold_pct", "severity", "status")
        .orderBy(F.desc("failed_rows"), "table_name"))

# COMMAND ----------

# MAGIC %md ## 3 · Persist results (append-only history)

# COMMAND ----------

all_results = bronze_results.unionByName(silver_results)
all_results.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(T("dq_results"))

# Convenience view: the most recent run only
spark.sql(f"""
    CREATE OR REPLACE VIEW {T('dq_latest')} AS
    SELECT * FROM {T('dq_results')}
    WHERE run_id = (SELECT MAX(run_id) FROM {T('dq_results')})
""")
print("Saved to", T("dq_results"), "and refreshed view", T("dq_latest"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 · Headline numbers: bronze vs silver
# MAGIC "Reliability" here = the share of individual row-checks that passed across *all* rules in a layer.
# MAGIC It's only as meaningful as the rules behind it, so quote the rule count next to it.

# COMMAND ----------

def layer_summary(res, label):
    counts = {r["status"]: r["n"] for r in res.groupBy("status").agg(F.count("*").alias("n")).collect()}
    return {
        "layer": label,
        "rules": res.count(),
        "pass": counts.get("PASS", 0),
        "warn": counts.get("WARN", 0),
        "fail": counts.get("FAIL", 0),
        "reliability_pct": dq.reliability_pct(res),
    }

summary = [layer_summary(bronze_results, "bronze"), layer_summary(silver_results, "silver")]
display(spark.createDataFrame(summary))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 · The gate
# MAGIC Raises if any `error` rule fails on silver. In Day 5 this cell becomes a Workflow task, so a bad load
# MAGIC stops before it reaches gold.

# COMMAND ----------

dq.assert_no_failures(silver_results)
print("PASS: silver meets every error-level rule")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Done: Day 3 checklist
# MAGIC - [ ] `dq_results` table and `dq_latest` view exist
# MAGIC - [ ] Bronze shows failing rules; silver shows the gate passing
# MAGIC - [ ] Screenshot: bronze results (sorted by pass rate)
# MAGIC - [ ] Screenshot: silver results + the layer summary
# MAGIC - [ ] Copy the layer summary into the README
# MAGIC - [ ] Commit and push
