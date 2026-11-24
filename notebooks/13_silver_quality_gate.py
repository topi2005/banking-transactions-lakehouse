# Databricks notebook source
# MAGIC %md
# MAGIC # 13 · Silver quality gate
# MAGIC **Task 3 of the workflow.** Runs the Day 3 silver rules against the *whole* silver layer and fails the job if any
# MAGIC `error`-level rule is below its threshold. Because the gold task depends on this one, **bad data stops here
# MAGIC and never reaches the star schema.** Results are appended to `dq_results` either way.

# COMMAND ----------

import sys
sys.path.append("../src")   # repo layout: notebooks/ and src/ are siblings

import dq_framework as dq
import pipeline_ops as ops
from dq_rules import SILVER_RULES

CATALOG = "workspace"
SCHEMA = "banking_lakehouse"
T = lambda name: f"{CATALOG}.{SCHEMA}.{name}"

RUN_ID = ops.new_batch_id()
tables = {n: spark.table(T(f"silver_{n}")) for n in ["transactions", "accounts", "customers", "merchants"]}

results = dq.run_rules(tables, SILVER_RULES, RUN_ID, "silver").cache()
results.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(T("dq_results"))

# COMMAND ----------

display(results.select("table_name", "rule_name", "kind", "failed_rows", "total_rows",
                       "pass_rate_pct", "threshold_pct", "severity", "status").orderBy("pass_rate_pct"))
print(f"{results.count()} rules, reliability {dq.reliability_pct(results)}%")

# COMMAND ----------

dq.assert_no_failures(results)   # raises, and the task (and job) fail, if any error-level rule fails
print("PASS: silver meets every error-level rule")
