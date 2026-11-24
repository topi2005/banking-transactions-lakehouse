# Databricks notebook source
# MAGIC %md
# MAGIC # 12 · Silver (incremental MERGE)
# MAGIC **Task 2 of the workflow.** For every bronze batch that silver hasn't finished, run the Day 2 cleaning logic on
# MAGIC *that batch only* and **MERGE** the valid rows into `silver_transactions` (insert-only, keyed on `transaction_id`).
# MAGIC
# MAGIC Why insert-only: a ledger transaction is immutable, so a re-delivered row is not an update. It is a duplicate
# MAGIC to ignore. That makes the step **idempotent**: reprocessing a batch can never create duplicates.
# MAGIC
# MAGIC Every batch must reconcile:
# MAGIC `bronze rows = duplicates in batch + quarantined + re-delivered (already in silver) + newly inserted`
# MAGIC
# MAGIC Side effect worth noticing: a row quarantined earlier (e.g. missing amount) and **re-sent with a fix** now passes
# MAGIC validation and is inserted. The pipeline heals itself.

# COMMAND ----------

import sys
sys.path.append("../src")   # repo layout: notebooks/ and src/ are siblings

from pyspark.sql import functions as F
import pipeline_ops as ops
import silver_transforms as st

CATALOG = "workspace"
SCHEMA = "banking_lakehouse"
T = lambda name: f"{CATALOG}.{SCHEMA}.{name}"
LOG = T("pipeline_batch_log")
BRONZE, SILVER_TX, QUAR = T("bronze_transactions"), T("silver_transactions"), T("silver_transactions_quarantine")

pending = ops.pending_batches(spark, LOG, "bronze", "silver")
print("Pending batches for silver:", pending)
if not pending:
    dbutils.notebook.exit("NOTHING_PENDING")

# COMMAND ----------

silver_accounts = spark.table(T("silver_accounts"))   # transactions-only pipeline: dimensions don't change

for batch in pending:                                  # oldest first: order matters for "first ingested wins"
    with ops.tracked_run(spark, LOG, batch, "silver") as stats:
        bronze_batch = spark.table(BRONZE).filter(F.col("_batch_id") == batch)
        n_bronze = bronze_batch.count()

        deduped = st.deduplicate(st.standardise_transactions(bronze_batch))
        n_dups = n_bronze - deduped.count()

        checked = st.apply_transaction_rules(deduped, silver_accounts)
        silver_new, quarantine = st.split_valid_and_quarantine(checked)
        n_quarantine = quarantine.count()
        n_valid = silver_new.count()

        # rows whose transaction_id is already in silver: re-deliveries, to be ignored
        existing_ids = spark.table(SILVER_TX).select("transaction_id")
        n_redelivered = silver_new.join(existing_ids, "transaction_id", "left_semi").count()
        before = spark.table(SILVER_TX).count()

        # quarantine: replace this batch's rows so a rerun can't double-count them
        spark.sql(f"DELETE FROM {QUAR} WHERE _batch_id = '{batch}'")
        quarantine.write.format("delta").mode("append").saveAsTable(QUAR)

        # the incremental load itself: insert-only MERGE
        silver_new.createOrReplaceTempView("silver_batch")
        spark.sql(f"""
            MERGE INTO {SILVER_TX} AS t
            USING silver_batch AS s
            ON t.transaction_id = s.transaction_id
            WHEN NOT MATCHED THEN INSERT *
        """)
        n_inserted = spark.table(SILVER_TX).count() - before

        # reconciliation: every bronze row in this batch is accounted for
        assert n_inserted == n_valid - n_redelivered, \
            f"MERGE inserted {n_inserted:,}, expected {n_valid - n_redelivered:,}"
        assert n_bronze == n_dups + n_quarantine + n_redelivered + n_inserted, \
            "RECONCILIATION FAILED: bronze rows are unaccounted for"

        stats.update(rows_in=n_bronze, rows_out=n_inserted, rows_rejected=n_quarantine,
                     message=f"dups_in_batch={n_dups}, redelivered_ignored={n_redelivered}")

    print(f"batch {batch}")
    print(f"  bronze {n_bronze:>9,} = dups {n_dups:,} + quarantined {n_quarantine:,} "
          f"+ re-delivered {n_redelivered:,} + inserted {n_inserted:,}   PASS")

# COMMAND ----------

display(spark.table(LOG).filter(F.col("layer") == "silver").orderBy(F.desc("finished_at")).limit(5))
