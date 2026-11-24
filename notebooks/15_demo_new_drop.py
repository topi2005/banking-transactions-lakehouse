# Databricks notebook source
# MAGIC %md
# MAGIC # 15 · Demo: simulate a new daily file drop
# MAGIC Writes **one new file** into the landing volume, as if the source system had delivered tomorrow's transactions.
# MAGIC It is not part of the pipeline; run it by hand between two job runs to demonstrate incremental loading.
# MAGIC
# MAGIC The file contains four kinds of rows (counts are the defaults and approximate after dirty-data injection):
# MAGIC | Rows | Count | What the pipeline should do |
# MAGIC |---|---|---|
# MAGIC | New transactions, dated 1 to 3 July 2026 | 20,000 (+ about 2% in-file duplicates) | clean, quarantine the bad ones, insert the rest |
# MAGIC | Re-delivered rows (ids already loaded) | 1,000 | **ignore** (MERGE matches, nothing inserted) |
# MAGIC | Resubmitted fixes (earlier rows that lacked an amount, now with one) | 200 | **insert** (self-healing) |
# MAGIC
# MAGIC Run it **once per demo**. Running it again with the same `--batch-name` would overwrite the file with new random rows.

# COMMAND ----------

import sys
sys.path.append("../src")   # repo layout: notebooks/ and src/ are siblings

import generate_data

RAW = "/Volumes/workspace/banking_lakehouse/landing/raw"
BATCH_NAME = "batch2"       # use batch3, batch4... for later drops

generate_data.main(["--incremental", "--out", RAW, "--batch-name", BATCH_NAME])

# COMMAND ----------

display(spark.createDataFrame([(f.name, f.size) for f in dbutils.fs.ls(f"{RAW}/transactions")], "file string, bytes long"))
