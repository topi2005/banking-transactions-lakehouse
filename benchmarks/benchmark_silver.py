"""
PySpark vs pandas on the silver transactions cleaning step (CSV in, cleaned Parquet out).

    pip install pyspark pandas numpy pyarrow
    python benchmarks/benchmark_silver.py                       # sizes 50k, 500k, 2M rows
    python benchmarks/benchmark_silver.py --sizes 100000,1000000 --runs 3

What is measured: wall-clock time for read CSV -> clean/validate/quarantine -> write Parquet, per engine.
What is NOT measured: JVM start-up (Spark is warmed up first) or data generation.
Correctness gate: timings are only reported if both engines produce identical silver and quarantine row counts.

Out of memory? Lower --sizes and --spark-memory, or run notebooks/16_benchmark_databricks.py, which reuses
the functions below on Databricks instead of your machine.

IMPORTANT: PySpark here runs in *local mode on one machine*. This says nothing about a Databricks cluster. It
shows where Spark's overhead stops mattering on a single machine, and it depends heavily on core count.
"""
import argparse
import os
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

import generate_data                     # noqa: E402
import pandas_silver                     # noqa: E402


def make_data(rows: int, cache: Path) -> Path:
    out = cache / f"rows_{rows}"
    if not (out / "dirty_data_manifest.json").exists():
        print(f"  generating {rows:,} rows ...", flush=True)
        generate_data.main(["--rows", str(rows), "--customers", "20000", "--merchants", "1500",
                            "--out", str(out), "--seed", "42", "--split", "3"])
    return out


def run_pandas(data: Path, out: Path):
    raw = pd.concat([pd.read_csv(f, dtype=str) for f in sorted((data / "transactions").glob("*.csv"))], ignore_index=True)
    accounts = set(pd.read_csv(data / "accounts" / "accounts.csv", dtype=str)["account_id"])
    silver, quarantine, _ = pandas_silver.clean_transactions(raw, accounts)
    out.mkdir(parents=True, exist_ok=True)
    silver.to_parquet(out / "silver.parquet")
    quarantine.to_parquet(out / "quarantine.parquet")
    return len(silver), len(quarantine)


def run_spark(spark, data: Path, out: Path):
    import pipeline_ops as ops
    import silver_transforms as st

    raw = (spark.read.format("csv").option("header", "true").option("inferSchema", "false")
           .load(str(data / "transactions")))
    accounts = spark.read.option("header", "true").csv(str(data / "accounts")).select("account_id").distinct()
    deduped = st.deduplicate(st.standardise_transactions(ops.add_bronze_metadata(raw, "bench")))
    silver, quarantine = st.split_valid_and_quarantine(st.apply_transaction_rules(deduped, accounts))
    silver.write.mode("overwrite").parquet(str(out / "silver"))
    quarantine.write.mode("overwrite").parquet(str(out / "quarantine"))
    return (spark.read.parquet(str(out / "silver")).count(),
            spark.read.parquet(str(out / "quarantine")).count())


def timed(fn, runs):
    times, result = [], None
    for _ in range(runs):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times), result


def build_report(rows_out, env: str, runs: int, note: str = "") -> str:
    """Markdown table from [(rows, pandas_seconds, spark_seconds, result), ...]. Shared by the script and the notebook."""
    lines = ["| Rows in | pandas (s) | PySpark (s) | Faster | Ratio |", "|---|---|---|---|---|"]
    for n, tp, ts, _ in rows_out:
        faster, ratio = ("pandas", ts / tp) if tp < ts else ("PySpark", tp / ts)
        lines.append(f"| {n:,} | {tp:.1f} | {ts:.1f} | {faster} | {ratio:.1f}x |")
    return (f"**Environment:** {env}\n\n**Method:** median of {runs} runs; CSV read -> clean, validate, quarantine -> "
            f"Parquet write; Spark warmed up first; outputs verified identical.{note}\n\n" + "\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="50000,500000,2000000")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--cache", default=str(Path(tempfile.gettempdir()) / "lakehouse_bench"))
    ap.add_argument("--spark-memory", default="4g")
    a = ap.parse_args()

    import pyspark
    from pyspark.sql import SparkSession
    spark = (SparkSession.builder.master("local[*]").appName("bench")
             .config("spark.driver.memory", a.spark_memory).config("spark.ui.enabled", "false")
             .config("spark.sql.shuffle.partitions", str(max(2, (os.cpu_count() or 2) * 2)))
             .config("spark.sql.session.timeZone", "UTC").config("spark.sql.ansi.enabled", "true").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")

    sizes = [int(s) for s in a.sizes.split(",")]
    cache = Path(a.cache)
    work = Path(tempfile.mkdtemp(prefix="bench_out_"))

    print("Warming up Spark ...", flush=True)
    warm = make_data(min(sizes), cache)
    run_spark(spark, warm, work / "warm")

    rows_out = []
    for n in sizes:
        print(f"\n== {n:,} rows", flush=True)
        data = make_data(n, cache)
        t_pd, res_pd = timed(lambda: run_pandas(data, work / f"pd_{n}"), a.runs)
        t_sp, res_sp = timed(lambda: run_spark(spark, data, work / f"sp_{n}"), a.runs)
        if res_pd != res_sp:
            sys.exit(f"ABORT: engines disagree at {n:,} rows: pandas (silver, quarantine) = {res_pd}, "
                     f"PySpark = {res_sp}. Timings would not be like for like.")
        print(f"  outputs identical: silver={res_pd[0]:,} quarantine={res_pd[1]:,}")
        print(f"  pandas {t_pd:7.1f}s | PySpark {t_sp:7.1f}s", flush=True)
        rows_out.append((n, t_pd, t_sp, res_pd))

    env = (f"{platform.platform()} | {os.cpu_count()} logical cores | Python {platform.python_version()} | "
           f"pandas {pd.__version__} | PySpark {pyspark.__version__} (local[*], driver memory {a.spark_memory})")
    report = build_report(rows_out, env, a.runs)
    (ROOT / "benchmarks" / "results.md").write_text(report)
    print("\n" + report)
    spark.stop()


if __name__ == "__main__":
    main()