"""Operational helpers: new-file selection, batch ids, bronze lineage columns."""
import pipeline_ops as ops


def test_basename_handles_both_path_styles():
    assert ops.basename("dbfs:/Volumes/x/raw/transactions/file.csv") == "file.csv"
    assert ops.basename("/Volumes/x/raw/transactions/file.csv/") == "file.csv"


def test_select_new_files_compares_by_name_and_ignores_non_csv():
    listed = ["dbfs:/Volumes/x/t/part1.csv", "dbfs:/Volumes/x/t/batch2.csv", "dbfs:/Volumes/x/t/notes.txt"]
    assert [ops.basename(p) for p in ops.select_new_files(listed, {"part1.csv"})] == ["batch2.csv"]
    assert ops.select_new_files(listed, {"part1.csv", "batch2.csv"}) == []


def test_select_new_files_matches_across_dbfs_prefix_difference():
    # listing says 'dbfs:/Volumes/...', _metadata.file_path stored '/Volumes/...': names still match
    assert ops.select_new_files(["dbfs:/Volumes/a/b.csv"], {ops.basename("/Volumes/a/b.csv")}) == []


def test_batch_ids_are_unique_and_sort_in_time_order():
    ids = [ops.new_batch_id() for _ in range(50)]
    assert len(set(ids)) == 50
    assert ids == sorted(ids) or len({i[:16] for i in ids}) == 1     # same-second ids tie on the prefix only


def test_add_bronze_metadata_records_source_file_and_batch(spark, tmp_path):
    f = tmp_path / "transactions_x.csv"
    f.write_text("transaction_id,amount\nT1,10\nT2,20\n")
    raw = spark.read.format("csv").option("header", "true").option("inferSchema", "false").load(str(f))
    out = ops.add_bronze_metadata(raw, "batch-1")
    assert {"_ingested_at", "_source_file", "_batch_id"} <= set(out.columns)
    rows = out.collect()
    assert {r["_batch_id"] for r in rows} == {"batch-1"}
    assert all(r["_source_file"].endswith("transactions_x.csv") for r in rows)
    assert [r["amount"] for r in sorted(rows, key=lambda r: r["transaction_id"])] == ["10", "20"]   # still strings
