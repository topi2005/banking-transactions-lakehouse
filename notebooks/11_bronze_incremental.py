# Databricks notebook source
# MAGIC %md
# MAGIC # 11 · Bronze (incremental)
# MAGIC **Task 1 of the workflow.** Finds transaction files in the landing folder that haven't been loaded yet and
# MAGIC **appends** them to bronze as one new batch. Nothing is cleaned here, as before.
# MAGIC
# MAGIC "Not loaded yet" = the file name isn't in `bronze_transactions._source_file`. If there is nothing new, the task
# MAGIC succeeds and says so, so a daily schedule with no new file is *not* a failure.
# MAGIC
# MAGIC *Production alternatives to this hand-rolled file tracking: Auto Loader (`cloudFiles`) or `COPY INTO`.*

# COMMAND ----------

import sys
sys.path.append("../src")   # repo layout: notebooks/ and src/ are siblings

from pyspark.sql import functions as F
import pipeline_ops as ops

CATALOG = "workspace"
SCHEMA = "banking_lakehouse"
T = lambda name: f"{CATALOG}.{SCHEMA}.{name}"
LOG = T("pipeline_batch_log")
BRONZE = T("bronze_transactions")
LANDING = f"/Volumes/{CATALOG}/{SCHEMA}/landing/raw/transactions"

ops.ensure_log_table(spark, LOG)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Recovery
# MAGIC If an earlier run appended to bronze but crashed before logging, the batch is in the table but not in the log.
# MAGIC A Delta append is atomic, so the batch is complete: log it so silver can pick it up.

# COMMAND ----------

logged = {r["batch_id"] for r in spark.table(LOG).filter((F.col("layer") == "bronze") & (F.col("status") == "SUCCESS"))
                                                .select("batch_id").collect()}
present = {r["_batch_id"] for r in spark.table(BRONZE).select("_batch_id").distinct().collect()}
for b in sorted(present - logged):
    ops.log_event(spark, LOG, b, "bronze", "SUCCESS", message="recovered: written to bronze but not logged")
    print("recovered batch", b)

# COMMAND ----------

# MAGIC %md ## Find new files

# COMMAND ----------

listed = [f.path for f in dbutils.fs.ls(LANDING)]
loaded = {r["n"] for r in spark.table(BRONZE)
          .select(F.regexp_extract("_source_file", r"[^/]+$", 0).alias("n")).distinct().collect()}
new_files = ops.select_new_files(listed, loaded)

print(f"{len(listed)} file(s) in landing, {len(loaded)} already loaded, {len(new_files)} new")
for f in new_files:
    print("  new:", ops.basename(f))

if not new_files:
    dbutils.notebook.exit("NO_NEW_FILES")

# COMMAND ----------

# MAGIC %md ## Append the new files as one batch

# COMMAND ----------

batch_id = ops.new_batch_id()
print("Batch:", batch_id)

with ops.tracked_run(spark, LOG, batch_id, "bronze") as stats:
    raw = (spark.read.format("csv").option("header", "true").option("inferSchema", "false")
           .load(new_files))
    n_read = raw.count()

    ops.add_bronze_metadata(raw, batch_id).write.format("delta").mode("append").saveAsTable(BRONZE)

    n_written = spark.table(BRONZE).filter(F.col("_batch_id") == batch_id).count()
    assert n_written == n_read, f"Bronze append lost rows: read {n_read:,}, written {n_written:,}"
    stats.update(rows_in=n_read, rows_out=n_written,
                 message="files=" + ", ".join(ops.basename(f) for f in new_files))

print(f"Appended {n_written:,} rows to bronze (batch {batch_id})")
dbutils.notebook.exit(batch_id)
