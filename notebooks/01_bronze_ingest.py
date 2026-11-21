# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 01 · Bronze ingestion
# MAGIC **Goal:** land raw CSVs in a Unity Catalog Volume and load them *unchanged* into Delta tables,
# MAGIC adding only lineage metadata (when, from which file, which batch).
# MAGIC
# MAGIC Bronze rules: **no cleaning, no type casting, no dropping rows.** Everything is read as strings so
# MAGIC nothing is lost or silently coerced. Fixing the data is the silver layer's job.

# COMMAND ----------

# MAGIC %md ## 0 · Config

# COMMAND ----------

import json
import uuid
from datetime import datetime, timezone

from pyspark.sql import DataFrame, functions as F

CATALOG = "workspace"            # change if your workspace uses another catalog
SCHEMA = "banking_lakehouse"
VOLUME = "landing"

BASE = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
RAW = f"{BASE}/raw"

# source folder (under RAW)  ->  bronze table
SOURCES = {
    "customers": "bronze_customers",
    "accounts": "bronze_accounts",
    "merchants": "bronze_merchants",
    "transactions": "bronze_transactions",
}

BATCH_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
print("Batch:", BATCH_ID)

# COMMAND ----------

# MAGIC %md ## 1 · Create schema and landing volume (idempotent)

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")
dbutils.fs.mkdirs(RAW)
print("Ready:", BASE)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · Put the data in the volume
# MAGIC **Option A: generate it here (easiest).** Import `generate_data.py` into the same workspace folder as
# MAGIC this notebook, then run the cell below.
# MAGIC
# MAGIC **Option B: generate locally and upload.** Run `python generate_data.py --out ./data`, then in
# MAGIC *Catalog → workspace → banking_lakehouse → landing* create the `raw/<table>/` folders and upload
# MAGIC each CSV, plus `dirty_data_manifest.json` into `raw/`.

# COMMAND ----------

# Option A. Skip this cell if you uploaded the files yourself.
import generate_data

generate_data.main(["--rows", "500000", "--customers", "20000", "--merchants", "1500", "--out", RAW])

# COMMAND ----------

# Confirm the files landed where bronze expects them
for name in SOURCES:
    files = dbutils.fs.ls(f"{RAW}/{name}")
    print(f"{name:13s} {len(files)} file(s):", [f.name for f in files])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 · Ingest to bronze
# MAGIC Two design choices worth being able to explain in an interview:
# MAGIC * `inferSchema=false` → every column is a string, so bad values (`"R1,50"`, `"not_a_date"`) survive intact.
# MAGIC * `_metadata.file_path` records the source file per row. (`input_file_name()` is not supported on Unity Catalog.)
# MAGIC
# MAGIC Day 1 uses `overwrite` so reruns are safe. On day 5 this becomes an incremental load.

# COMMAND ----------

def add_bronze_metadata(df: DataFrame, batch_id: str) -> DataFrame:
    """Append lineage columns. Never touches the source columns."""
    return (
        df.withColumn("_ingested_at", F.current_timestamp())
          .withColumn("_source_file", F.col("_metadata.file_path"))
          .withColumn("_batch_id", F.lit(batch_id))
    )


def ingest_to_bronze(source: str, table: str, batch_id: str) -> int:
    raw_df = (
        spark.read.format("csv")
        .option("header", "true")
        .option("inferSchema", "false")   # keep everything as string
        .load(f"{RAW}/{source}/")
    )
    bronze_df = add_bronze_metadata(raw_df, batch_id)
    (
        bronze_df.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(f"{CATALOG}.{SCHEMA}.{table}")
    )
    return spark.table(f"{CATALOG}.{SCHEMA}.{table}").count()


bronze_counts = {}
for source, table in SOURCES.items():
    bronze_counts[table] = ingest_to_bronze(source, table, BATCH_ID)
    print(f"{table:22s} {bronze_counts[table]:>10,} rows")

# COMMAND ----------

# MAGIC %md ## 4 · Validate bronze (your first reconciliation check)

# COMMAND ----------

manifest = json.load(open(f"{RAW}/dirty_data_manifest.json"))

checks = [
    ("transactions row count == rows written by generator",
     bronze_counts["bronze_transactions"], manifest["total_rows_written"]),
    ("customers row count",  bronze_counts["bronze_customers"], manifest["customers"]),
    ("accounts row count",   bronze_counts["bronze_accounts"],  manifest["accounts"]),
    ("merchants row count",  bronze_counts["bronze_merchants"], manifest["merchants"]),
]
all_ok = True
for label, got, expected in checks:
    ok = got == expected
    all_ok &= ok
    print(("PASS " if ok else "FAIL ") + f"{label}: got {got:,}, expected {expected:,}")

assert all_ok, "Bronze row counts do not reconcile with the source. Investigate before moving on."

# COMMAND ----------

# Every row should be traceable to a source file
display(
    spark.table(f"{CATALOG}.{SCHEMA}.bronze_transactions")
    .groupBy("_source_file", "_batch_id").count().orderBy("_source_file")
)

# COMMAND ----------

# Profile the raw mess: this is what silver has to fix. Save these numbers for your README "before" figures.
tx = spark.table(f"{CATALOG}.{SCHEMA}.bronze_transactions")
total = tx.count()

null_profile = tx.select([
    F.round(F.avg(F.col(c).isNull().cast("int")) * 100, 2).alias(c)
    for c in ["transaction_id", "account_id", "merchant_id", "transaction_ts", "amount", "channel"]
])
print("% NULL by column (bronze):")
display(null_profile)

dup_ids = total - tx.select("transaction_id").distinct().count()
print(f"Duplicate transaction_id rows: {dup_ids:,} ({dup_ids / total:.2%})")

print("Channel values (note the case/whitespace variants):")
display(tx.groupBy("channel").count().orderBy(F.desc("count")))

# COMMAND ----------

# Delta gives you free audit history: have a look at it, and screenshot it for the README
display(spark.sql(f"DESCRIBE HISTORY {CATALOG}.{SCHEMA}.bronze_transactions"))
display(spark.sql(f"DESCRIBE DETAIL {CATALOG}.{SCHEMA}.bronze_transactions"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Done: Day 1 checklist
# MAGIC - [ ] 4 bronze tables exist in Catalog Explorer
# MAGIC - [ ] Reconciliation checks print PASS
# MAGIC - [ ] Saved the "before" numbers (null %, duplicates, channel variants)
# MAGIC - [ ] Screenshot of `DESCRIBE HISTORY`
# MAGIC - [ ] Code committed to a new GitHub repo (connect via *Workspace → Git folder* or export the notebook)