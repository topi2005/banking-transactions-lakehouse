# Databricks notebook source
# MAGIC %md
# MAGIC # 16 · Benchmark on Databricks: pandas vs PySpark
# MAGIC Runs the same silver cleaning benchmark as `benchmarks/benchmark_silver.py`, but on Databricks instead of your laptop.
# MAGIC
# MAGIC **What this compares:** pandas running on the **single driver node** vs PySpark running on **Databricks serverless compute**.
# MAGIC That is a different experiment from the local-mode one: you don't control the core count, serverless adds scheduling
# MAGIC latency, and timings vary between runs. Quote the results as *"on Databricks Free Edition serverless"*, not as a general claim.
# MAGIC
# MAGIC Safety net: timings are reported only if both engines produce **identical silver and quarantine row counts**.
# MAGIC If the notebook runs out of memory at the largest size, remove that size from `SIZES` and run again.

# COMMAND ----------

import os
import platform
import sys
from pathlib import Path

import pandas as pd
import pyspark

sys.path.append("../benchmarks")   # repo layout: notebooks/ and benchmarks/ are siblings
import benchmark_silver as bm       # reuses the exact functions the local script uses

DATA_ROOT = Path("/Volumes/workspace/banking_lakehouse/landing/bench")   # generated test data and outputs live here
SIZES = [50_000, 500_000, 1_000_000]
RUNS = 3

# COMMAND ----------

# MAGIC %md ## Warm up Spark (the first query pays one-off start-up costs, which would unfairly penalise Spark)

# COMMAND ----------

warm_data = bm.make_data(min(SIZES), DATA_ROOT)
bm.run_spark(spark, warm_data, DATA_ROOT / "out_warm")
print("warmed up")

# COMMAND ----------

# MAGIC %md ## Run

# COMMAND ----------

results = []
for n in SIZES:
    print(f"\n== {n:,} rows", flush=True)
    data = bm.make_data(n, DATA_ROOT)
    t_pd, res_pd = bm.timed(lambda: bm.run_pandas(data, DATA_ROOT / f"out_pd_{n}"), RUNS)
    t_sp, res_sp = bm.timed(lambda: bm.run_spark(spark, data, DATA_ROOT / f"out_sp_{n}"), RUNS)
    assert res_pd == res_sp, (f"Engines disagree at {n:,} rows: pandas (silver, quarantine) = {res_pd}, "
                              f"PySpark = {res_sp}. Timings would not be like for like.")
    print(f"  outputs identical: silver={res_pd[0]:,} quarantine={res_pd[1]:,}")
    print(f"  pandas {t_pd:7.1f}s | PySpark {t_sp:7.1f}s")
    results.append((n, t_pd, t_sp, res_pd))

# COMMAND ----------

# MAGIC %md ## Result table (copy this into the README)

# COMMAND ----------

env = (f"Databricks Free Edition serverless | pandas on the driver ({os.cpu_count()} logical cores) vs "
       f"PySpark {pyspark.__version__} on serverless compute | Python {platform.python_version()} | pandas {pd.__version__}")
report = bm.build_report(results, env, RUNS,
                         note=" Not a local-mode benchmark: serverless scheduling latency is included in the PySpark times.")
print(report)

(DATA_ROOT / "results_databricks.md").write_text(report)
print("Saved to", DATA_ROOT / "results_databricks.md")
