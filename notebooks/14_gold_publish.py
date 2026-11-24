# Databricks notebook source
# MAGIC %md
# MAGIC # 14 · Gold (incremental publish)
# MAGIC **Task 4 of the workflow.** For every silver batch gold hasn't published:
# MAGIC 1. **MERGE** the batch's fact rows into `fact_transactions` (insert-only, keyed on `transaction_id`)
# MAGIC 2. rebuild `dim_date` so it covers the new dates (a fact row for 1 July needs a 1 July dimension row)
# MAGIC 3. verify the star schema, including the control totals against silver
# MAGIC 4. rebuild the two aggregates from the fact table
# MAGIC
# MAGIC If verification fails, the task fails and the batch is logged FAILED, so it is retried on the next run.

# COMMAND ----------

import sys
sys.path.append("../src")   # repo layout: notebooks/ and src/ are siblings

from pyspark.sql import functions as F
import dq_framework as dq
import gold_transforms as gt
import pipeline_ops as ops
from dq_rules import GOLD_RULES

CATALOG = "workspace"
SCHEMA = "banking_lakehouse"
T = lambda name: f"{CATALOG}.{SCHEMA}.{name}"
LOG = T("pipeline_batch_log")
SILVER_TX, FACT = T("silver_transactions"), T("fact_transactions")

pending = ops.pending_batches(spark, LOG, "silver", "gold")
print("Pending batches for gold:", pending)
if not pending:
    dbutils.notebook.exit("NOTHING_PENDING")

def overwrite(df, name):
    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(T(name))

# COMMAND ----------

with ops.tracked_run(spark, LOG, pending, "gold") as stats:
    silver_tx = spark.table(SILVER_TX)
    silver_accounts = spark.table(T("silver_accounts"))
    fact_before = spark.table(FACT).count()

    # 1. fact: insert-only MERGE of each pending batch
    n_silver_rows = 0
    for batch in pending:
        silver_batch = silver_tx.filter(F.col("_batch_id") == batch)
        n_silver_rows += silver_batch.count()
        gt.build_fact_transactions(silver_batch, silver_accounts).createOrReplaceTempView("fact_batch")
        spark.sql(f"""
            MERGE INTO {FACT} AS t
            USING fact_batch AS s
            ON t.transaction_id = s.transaction_id
            WHEN NOT MATCHED THEN INSERT *
        """)
    n_inserted = spark.table(FACT).count() - fact_before
    print(f"fact: {n_inserted:,} rows inserted from {n_silver_rows:,} silver rows in {len(pending)} batch(es)")

    # 2. dim_date must cover the new dates
    bounds = silver_tx.agg(F.min("transaction_date").alias("lo"), F.max("transaction_date").alias("hi")).first()
    overwrite(gt.build_dim_date(spark, bounds["lo"], bounds["hi"]), "dim_date")
    print("dim_date covers", bounds["lo"], "to", bounds["hi"])

    # 3. verify the model (same rules and control totals as Day 4)
    gold_tables = {
        "fact": spark.table(FACT),
        "dim_customer": spark.table(T("dim_customer")), "dim_account": spark.table(T("dim_account")),
        "dim_merchant": spark.table(T("dim_merchant")), "dim_channel": spark.table(T("dim_channel")),
        "dim_date": spark.table(T("dim_date")),
    }
    run_id = ops.new_batch_id()
    results = dq.run_rules(gold_tables, GOLD_RULES, run_id, "gold")

    cents = lambda df: int(df.agg(F.sum("amount")).first()[0] * 100)
    fact_df, silver_df = gold_tables["fact"], spark.table(SILVER_TX)
    for r in [
        dq.equality_check(spark, run_id, "gold", "fact", "fact_rows_equal_silver_rows",
                          expected=silver_df.count(), actual=fact_df.count()),
        dq.equality_check(spark, run_id, "gold", "fact", "fact_amount_cents_equal_silver_amount_cents",
                          expected=cents(silver_df), actual=cents(fact_df)),
        dq.equality_check(spark, run_id, "gold", "fact", "unknown_merchant_rows_equal_imputed_rows",
                          expected=silver_df.filter("merchant_imputed").count(),
                          actual=fact_df.filter(F.col("merchant_key") == gt.UNKNOWN_KEY).count()),
    ]:
        results = results.unionByName(r)
    results = results.cache()
    results.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(T("dq_results"))
    dq.assert_no_failures(results)
    print(f"PASS: {results.count()} gold checks, control totals reconcile with silver")

    # 4. aggregates (cheap to rebuild from the fact table)
    overwrite(gt.build_agg_daily_segment(fact_df, gold_tables["dim_customer"], gold_tables["dim_date"]),
              "agg_daily_segment")
    overwrite(gt.build_agg_monthly_category(fact_df, gold_tables["dim_merchant"], gold_tables["dim_date"]),
              "agg_monthly_category")
    for name in ["agg_daily_segment", "agg_monthly_category"]:
        total = spark.table(T(name)).agg((F.sum("txn_count") + F.sum("outlier_txn_count")).alias("n")).first()["n"]
        assert total == fact_df.count(), f"{name} does not account for every fact row"

    stats.update(rows_in=n_silver_rows, rows_out=n_inserted,
                 message=f"fact_total={fact_df.count()}, gold_checks={results.count()}")

# COMMAND ----------

display(spark.table(LOG).orderBy(F.desc("finished_at")).limit(10))
