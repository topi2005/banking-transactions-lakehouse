"""
Operational helpers for the incremental pipeline (transactions only).

The idea
--------
Every load is a numbered BATCH (one batch = the new files found in one run). A small append-only Delta table,
`pipeline_batch_log`, records the outcome of each batch at each layer:

    batch_id | layer  | status  | rows_in | rows_out | rows_rejected | message | started_at | finished_at

Each layer asks the log "which batches did the layer before me finish that I have not?" and processes only
those. That gives the pipeline three properties without any external orchestrator state:

* Incremental: only new data is touched.
* Restartable: if silver fails halfway, the batch has no silver SUCCESS row, so the next run picks it up again.
* Observable: the log doubles as the operational report for first-line support (what ran, what failed, how many rows).

Idempotency comes from the writes themselves (insert-only MERGE on transaction_id), so reprocessing a
batch can never create duplicates.
"""
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

from pyspark.sql import DataFrame, functions as F

LOG_SCHEMA = (
    "batch_id string, layer string, status string, rows_in long, rows_out long, "
    "rows_rejected long, message string, started_at timestamp, finished_at timestamp"
)


# --------------------------------------------------------------------------- batches and files
def new_batch_id() -> str:
    """Sortable batch id: UTC timestamp + short random suffix (sorting ids sorts batches in time order)."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]


def basename(path: str) -> str:
    return path.rstrip("/").split("/")[-1]


def select_new_files(listed_paths, loaded_names) -> list:
    """
    Files in the landing folder that have not been loaded yet. Compared by FILE NAME, because the same file can
    appear as 'dbfs:/Volumes/...' in a directory listing and '/Volumes/...' in `_metadata.file_path`.
    """
    seen = set(loaded_names)
    return [p for p in sorted(listed_paths) if p.lower().endswith(".csv") and basename(p) not in seen]


def add_bronze_metadata(df: DataFrame, batch_id: str) -> DataFrame:
    """Lineage columns for bronze (same contract as Day 1): never touches the source columns."""
    return (df.withColumn("_ingested_at", F.current_timestamp())
              .withColumn("_source_file", F.col("_metadata.file_path"))
              .withColumn("_batch_id", F.lit(batch_id)))


# --------------------------------------------------------------------------- batch log
def ensure_log_table(spark, table: str) -> None:
    spark.sql(f"CREATE TABLE IF NOT EXISTS {table} ({LOG_SCHEMA}) USING DELTA")


def log_event(spark, table, batch_id, layer, status, rows_in=None, rows_out=None,
              rows_rejected=None, message=None, started_at=None) -> None:
    row = [(batch_id, layer, status, rows_in, rows_out, rows_rejected,
            (message or "")[:1000] or None, started_at, datetime.now(timezone.utc))]
    spark.createDataFrame(row, LOG_SCHEMA).write.format("delta").mode("append").saveAsTable(table)


def pending_batches(spark, table: str, upstream_layer: str, layer: str) -> list:
    """Batch ids the upstream layer finished successfully and `layer` has not, oldest first."""
    log = spark.table(table)
    upstream = (log.filter((F.col("layer") == upstream_layer) & (F.col("status") == "SUCCESS"))
                   .select("batch_id").distinct())
    done = (log.filter((F.col("layer") == layer) & (F.col("status") == "SUCCESS"))
               .select("batch_id").distinct())
    return sorted(r["batch_id"] for r in upstream.join(done, "batch_id", "left_anti").collect())


@contextmanager
def tracked_run(spark, table: str, batch_ids, layer: str):
    """
    Wrap a unit of work. Yields a dict to fill with rows_in / rows_out / rows_rejected / message.
    Logs SUCCESS for every batch on a clean exit, or FAILED (with the error) and re-raises on any exception,
    so the job still fails loudly while the log records why.
    """
    batch_ids = [batch_ids] if isinstance(batch_ids, str) else list(batch_ids)
    started = datetime.now(timezone.utc)
    stats = {}
    try:
        yield stats
    except Exception as exc:
        for b in batch_ids:
            log_event(spark, table, b, layer, "FAILED", message=f"{type(exc).__name__}: {exc}", started_at=started)
        raise
    for b in batch_ids:
        log_event(spark, table, b, layer, "SUCCESS", started_at=started, **stats)
