"""The data-quality engine: every rule kind, NULL conventions, thresholds, severity, gate, reliability."""
import pytest

import dq_framework as dq
from dq_framework import Rule


@pytest.fixture
def tx(spark):
    return spark.createDataFrame(
        [("T1", "A1", "POS", 10.0, "2026-01-05"),
         ("T2", "A1", "pos", 20.0, "2026-01-06"),       # bad channel value
         ("T3", "A9", "ATM", -5.0, "2026-01-07"),       # orphan account, negative amount
         ("T3", "A2", None, 1e6, "bad-date"),           # duplicate id, NULL channel, out-of-range amount
         (None, "A1", "APP", 30.0, "2026-01-09")],      # NULL id
        "transaction_id string, account_id string, channel string, amount double, d string")


@pytest.fixture
def accounts(spark):
    return spark.createDataFrame([("A1",), ("A2",)], "account_id string")


def _run(spark, tables, rules):
    res = dq.run_rules(tables, rules, "run1", "test")
    return {r["rule_name"]: r for r in res.collect()}


def test_each_rule_kind_counts_failures_correctly(spark, tx, accounts):
    rules = [
        Rule("id_not_null", "tx", "not_null", "transaction_id"),
        Rule("id_unique", "tx", "unique", "transaction_id"),
        Rule("channel_valid", "tx", "accepted_values", "channel", {"values": ["POS", "ATM", "APP"]}),
        Rule("amount_range", "tx", "range", "amount", {"min": 0, "max": 1000}),
        Rule("d_format", "tx", "regex", "d", {"pattern": r"^\d{4}-\d{2}-\d{2}$"}),
        Rule("account_fk", "tx", "foreign_key", "account_id", {"ref_table": "accounts", "ref_column": "account_id"}),
        Rule("no_negative", "tx", "sql_expression", None, {"fail_sql": "amount < 0"}),
    ]
    out = _run(spark, {"tx": tx, "accounts": accounts}, rules)
    assert out["id_not_null"]["failed_rows"] == 1
    assert out["id_unique"]["failed_rows"] == 1            # T3 appears twice: one extra row
    assert out["channel_valid"]["failed_rows"] == 1        # 'pos' only: the NULL channel is ignored
    assert out["amount_range"]["failed_rows"] == 2         # -5 and 1e6
    assert out["d_format"]["failed_rows"] == 1
    assert out["account_fk"]["failed_rows"] == 1           # A9
    assert out["no_negative"]["failed_rows"] == 1
    assert all(r["total_rows"] == 5 for r in out.values())


def test_only_not_null_cares_about_nulls(spark, tx):
    rules = [Rule("ch_valid", "tx", "accepted_values", "channel", {"values": ["POS"]}),
             Rule("ch_not_null", "tx", "not_null", "channel")]
    out = _run(spark, {"tx": tx}, rules)
    assert out["ch_not_null"]["failed_rows"] == 1
    assert out["ch_valid"]["failed_rows"] == 3             # 'pos', ATM, APP fail; the NULL is not double counted


def test_foreign_key_allow_values_exempts_sentinels(spark):
    child = spark.createDataFrame([("M1",), ("UNKNOWN",), ("M9",)], "merchant_id string")
    parent = spark.createDataFrame([("M1",)], "merchant_id string")
    strict = Rule("strict", "c", "foreign_key", "merchant_id", {"ref_table": "p", "ref_column": "merchant_id"})
    lenient = Rule("lenient", "c", "foreign_key", "merchant_id",
                   {"ref_table": "p", "ref_column": "merchant_id", "allow_values": ["UNKNOWN"]})
    out = _run(spark, {"c": child, "p": parent}, [strict, lenient])
    assert out["strict"]["failed_rows"] == 2 and out["lenient"]["failed_rows"] == 1


@pytest.mark.parametrize("failed_of_total, threshold, severity, expected", [
    (0, 100.0, "error", "PASS"),
    (1, 100.0, "error", "FAIL"),
    (1, 100.0, "warn", "WARN"),
    (1, 75.0, "error", "PASS"),        # 80% pass rate >= 75% threshold
    (3, 75.0, "error", "FAIL"),        # 40% < 75%
])
def test_status_depends_on_threshold_and_severity(spark, failed_of_total, threshold, severity, expected):
    df = spark.createDataFrame([(i, i < failed_of_total) for i in range(5)], "id int, bad boolean")
    rule = Rule("r", "t", "sql_expression", None, {"fail_sql": "bad"}, threshold_pct=threshold, severity=severity)
    assert _run(spark, {"t": df}, [rule])["r"]["status"] == expected


def test_pass_rate_is_computed_from_failures(spark):
    df = spark.createDataFrame([(1,), (None,), (None,), (4,)], "x int")
    out = _run(spark, {"t": df}, [Rule("nn", "t", "not_null", "x")])
    assert out["nn"]["pass_rate_pct"] == 50.0


def test_empty_table_passes_rather_than_dividing_by_zero(spark):
    df = spark.createDataFrame([], "x int")
    out = _run(spark, {"t": df}, [Rule("nn", "t", "not_null", "x")])
    assert out["nn"]["pass_rate_pct"] == 100.0 and out["nn"]["status"] == "PASS"


def test_unknown_rule_kind_is_rejected(spark, tx):
    with pytest.raises(ValueError):
        dq.run_rules({"tx": tx}, [Rule("bad", "tx", "telepathy", "amount")], "r", "t")


def test_equality_check_passes_and_fails(spark):
    ok = dq.equality_check(spark, "r", "gold", "fact", "rows_match", expected=100, actual=100).first()
    bad = dq.equality_check(spark, "r", "gold", "fact", "rows_match", expected=100, actual=97).first()
    assert ok["status"] == "PASS" and ok["failed_rows"] == 0
    assert bad["status"] == "FAIL" and bad["failed_rows"] == 3


def test_reliability_is_share_of_row_checks_passed(spark):
    df = spark.createDataFrame([(1,), (None,), (3,), (4,)], "x int")
    res = dq.run_rules({"t": df}, [Rule("a", "t", "not_null", "x"), Rule("b", "t", "range", "x", {"min": 0})], "r", "t")
    assert dq.reliability_pct(res) == 87.5                  # 7 of 8 individual checks passed


def test_gate_raises_on_error_failures_but_not_on_warnings(spark):
    df = spark.createDataFrame([(1,), (None,)], "x int")
    fail = dq.run_rules({"t": df}, [Rule("nn", "t", "not_null", "x")], "r", "t")
    warn = dq.run_rules({"t": df}, [Rule("nn", "t", "not_null", "x", severity="warn")], "r", "t")
    with pytest.raises(AssertionError, match="nn"):
        dq.assert_no_failures(fail)
    dq.assert_no_failures(warn)                             # no exception


def test_shipped_rule_sets_only_use_known_kinds_and_have_unique_names():
    import dq_rules
    known = dq.ROW_LEVEL_KINDS | {"unique", "foreign_key"}
    for name in ("BRONZE_RULES", "SILVER_RULES", "GOLD_RULES"):
        rules = getattr(dq_rules, name)
        assert all(r.kind in known for r in rules), name
        keys = [(r.table, r.name) for r in rules]
        assert len(keys) == len(set(keys)), f"duplicate rule names in {name}"
