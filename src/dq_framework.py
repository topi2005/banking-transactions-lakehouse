"""
Small, config-driven data-quality framework for the banking-lakehouse project.

A *rule* is data (a `Rule` object), not code. The runner turns a list of rules into one results
DataFrame with a row per rule: how many rows were checked, how many failed, the pass rate, the
threshold, and a PASS / WARN / FAIL status. That DataFrame is appended to the `dq_results` Delta table.

Rule kinds
----------
not_null          column must not be NULL
unique            non-null values must not repeat
accepted_values   non-null values must be in params["values"]
range             non-null values must satisfy params["min"] <= value <= params["max"] (either bound optional)
regex             non-null values must match params["pattern"]
foreign_key       non-null values must exist in params["ref_table"].params["ref_column"]
                  (params["allow_values"] lists sentinel values such as 'UNKNOWN' that are exempt)
sql_expression    params["fail_sql"] is a SQL boolean that is TRUE for FAILING rows

Convention: only `not_null` cares about NULLs. Every other kind ignores NULLs, so one problem is
reported by one rule instead of by five.

Severity: "error" rules block the pipeline (see `assert_no_failures`); "warn" rules are reported only.
Tables are referred to by *logical* name (e.g. "transactions"); the caller maps each layer's real
tables to those names, so the same rules can run against bronze, silver or gold.
"""
from dataclasses import dataclass, field
from typing import Optional

from pyspark.sql import DataFrame, functions as F

ROW_LEVEL_KINDS = {"not_null", "accepted_values", "range", "regex", "sql_expression"}

RESULT_SCHEMA = (
    "run_id string, layer string, table_name string, rule_name string, kind string, "
    "column_name string, total_rows long, failed_rows long, pass_rate_pct double, "
    "threshold_pct double, severity string, status string"
)


@dataclass(frozen=True)
class Rule:
    name: str
    table: str
    kind: str
    column: Optional[str] = None
    params: dict = field(default_factory=dict)
    threshold_pct: float = 100.0   # minimum acceptable pass rate
    severity: str = "error"        # "error" blocks the pipeline, "warn" only reports


def _fail_expr(rule: Rule):
    """Boolean Column that is TRUE for rows that FAIL a row-level rule."""
    p = rule.params
    if rule.kind == "sql_expression":
        return F.coalesce(F.expr(p["fail_sql"]), F.lit(False))
    c = F.col(rule.column)
    if rule.kind == "not_null":
        return c.isNull()
    if rule.kind == "accepted_values":
        return c.isNotNull() & ~c.isin(*p["values"])
    if rule.kind == "regex":
        return c.isNotNull() & ~c.rlike(p["pattern"])
    if rule.kind == "range":
        out_of_range = F.lit(False)
        if "min" in p:
            out_of_range = out_of_range | (c < F.lit(p["min"]))
        if "max" in p:
            out_of_range = out_of_range | (c > F.lit(p["max"]))
        return c.isNotNull() & out_of_range
    raise ValueError(f"Unknown row-level rule kind: {rule.kind}")


def _status(pass_rate: float, rule: Rule) -> str:
    if pass_rate >= rule.threshold_pct:
        return "PASS"
    return "WARN" if rule.severity == "warn" else "FAIL"


def _row(run_id, layer, rule: Rule, total: int, failed: int):
    pass_rate = round(100.0 * (total - failed) / total, 4) if total else 100.0
    return (run_id, layer, rule.table, rule.name, rule.kind, rule.column,
            int(total), int(failed), float(pass_rate), float(rule.threshold_pct),
            rule.severity, _status(pass_rate, rule))


def run_rules(tables: dict, rules: list, run_id: str, layer: str) -> DataFrame:
    """
    Evaluate `rules` against `tables` (logical name -> DataFrame) and return a results DataFrame.
    All row-level rules for one table are evaluated in a single pass over that table.
    """
    spark = next(iter(tables.values())).sparkSession
    by_table = {}
    for r in rules:
        by_table.setdefault(r.table, []).append(r)

    rows = []
    for tname, trules in by_table.items():
        df = tables[tname]
        row_rules = [r for r in trules if r.kind in ROW_LEVEL_KINDS]

        aggs = [F.count(F.lit(1)).alias("_total")]
        aggs += [F.sum(F.when(_fail_expr(r), 1).otherwise(0)).alias(f"f{i}")
                 for i, r in enumerate(row_rules)]
        agg_row = df.agg(*aggs).first()
        total = agg_row["_total"]

        for i, r in enumerate(row_rules):
            rows.append(_row(run_id, layer, r, total, agg_row[f"f{i}"] or 0))

        for r in trules:
            if r.kind == "unique":
                non_null = df.filter(F.col(r.column).isNotNull())
                failed = non_null.count() - non_null.select(r.column).distinct().count()
                rows.append(_row(run_id, layer, r, total, failed))
            elif r.kind == "foreign_key":
                p = r.params
                ref = tables[p["ref_table"]].select(F.col(p["ref_column"]).alias("_ref")).distinct()
                child = df.filter(F.col(r.column).isNotNull())
                if p.get("allow_values"):
                    child = child.filter(~F.col(r.column).isin(*p["allow_values"]))
                failed = child.join(ref, child[r.column] == ref["_ref"], "left_anti").count()
                rows.append(_row(run_id, layer, r, total, failed))
            elif r.kind not in ROW_LEVEL_KINDS:
                raise ValueError(f"Unknown rule kind: {r.kind}")

    return spark.createDataFrame(rows, RESULT_SCHEMA).withColumn("executed_at", F.current_timestamp())


def equality_check(spark, run_id: str, layer: str, table: str, name: str,
                   expected: int, actual: int, severity: str = "error") -> DataFrame:
    """Cross-table reconciliation as a result row (e.g. bronze = silver + quarantine + duplicates)."""
    rule = Rule(name=name, table=table, kind="reconciliation", severity=severity)
    row = _row(run_id, layer, rule, expected, abs(expected - actual))
    return spark.createDataFrame([row], RESULT_SCHEMA).withColumn("executed_at", F.current_timestamp())


def reliability_pct(results: DataFrame) -> float:
    """Share of individual row-checks that passed, across every rule in `results`."""
    r = results.agg(F.sum("total_rows").alias("t"), F.sum("failed_rows").alias("f")).first()
    return round(100.0 * (r["t"] - r["f"]) / r["t"], 3) if r["t"] else 100.0


def assert_no_failures(results: DataFrame) -> None:
    """Raise if any severity='error' rule is below its threshold. Use as a pipeline gate."""
    failed = results.filter(F.col("status") == "FAIL").collect()
    if failed:
        lines = [f"  {r.table_name}.{r.rule_name}: {r.failed_rows:,} failed ({r.pass_rate_pct}% < {r.threshold_pct}%)"
                 for r in failed]
        raise AssertionError("Data quality gate failed:\n" + "\n".join(lines))
