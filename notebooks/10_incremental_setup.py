# Databricks notebook source
# MAGIC %md
# MAGIC # 10 · Incremental pipeline: one-time setup
# MAGIC Days 1 to 4 built every layer with a **full load**. Before the incremental pipeline takes over, this notebook:
# MAGIC 1. creates the batch log (`pipeline_batch_log`), and
# MAGIC 2. records the existing data as **already processed** at every layer, so the first incremental run
# MAGIC    doesn't reprocess 510,000 rows it has already handled.
# MAGIC
# MAGIC Run it **once**, *before* the first incremental run. It is safe to rerun: it only adds log rows that are missing.

# COMMAND ----------

import sys
sys.path.append("../src")   # repo layout: notebooks/ and src/ are siblings

from pyspark.sql import functions as F
import pipeline_ops as ops

CATALOG = "workspace"
SCHEMA = "banking_lakehouse"
T = lambda name: f"{CATALOG}.{SCHEMA}.{name}"
LOG = T("pipeline_batch_log")

# COMMAND ----------

ops.ensure_log_table(spark, LOG)

bronze = spark.table(T("bronze_transactions"))
existing_batches = sorted(r["_batch_id"] for r in bronze.select("_batch_id").distinct().collect())
print("Batches already in bronze:", existing_batches)

# COMMAND ----------

log = spark.table(LOG)
for layer in ["bronze", "silver", "gold"]:
    logged = {r["batch_id"] for r in log.filter((F.col("layer") == layer) & (F.col("status") == "SUCCESS"))
                                        .select("batch_id").collect()}
    for b in existing_batches:
        if b not in logged:
            rows = bronze.filter(F.col("_batch_id") == b).count() if layer == "bronze" else None
            ops.log_event(spark, LOG, b, layer, "SUCCESS", rows_in=rows, rows_out=rows,
                          message="baseline: built by the full load in Days 1-4")
            print(f"baselined {layer:6s} {b}")

# COMMAND ----------

display(spark.table(LOG).orderBy("batch_id", "layer"))

# Sanity check: these are the numbers the incremental pipeline starts from
print("silver_transactions", f"{spark.table(T('silver_transactions')).count():,}")
print("fact_transactions  ", f"{spark.table(T('fact_transactions')).count():,}")
