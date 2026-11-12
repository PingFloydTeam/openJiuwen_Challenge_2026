"""Regression checks for adversarial specifications and output requirements."""
import json
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import httpx

from bench import policy_checks
from bench.policy_checks import write_task
from ping_floyd.coding import Coding
from ping_floyd.contracts import required_inputs, universal_contract_conflicts
from ping_floyd.domain import NoUniqueAnswer, atomic_write
from ping_floyd.tools import db_query, kb_lookup
from ping_floyd.structured import net_ledger


class ContractChecks(unittest.IsolatedAsyncioTestCase):
    execute = policy_checks.OfflinePolicyChecks.execute

    async def test_output_failure_stops_and_snapshot_recovers_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / "answers.txt"
            write_task(root, "a", "Calculate 1 + 2.", 3)
            write_task(root, "b", "Calculate 3 + 4.", 2)

            def handler(_):
                self.fail("local tasks must not spend tokens")

            def failing_snapshot(path, value):
                if path.resolve() == out.resolve() and value:
                    raise OSError("simulated disk full")
                atomic_write(path, value)

            with patch("ping_floyd.domain.atomic_write", side_effect=failing_snapshot):
                events = await self.execute(root, handler, extra=("--out", str(out)))
            self.assertEqual(events[-1]["stop_reason"], "out_write_error")
            self.assertFalse(events[-1]["out_complete"])
            self.assertEqual((root / "a/answer.txt").read_text(), "3\n")
            self.assertFalse((root / "b/answer.txt").exists())
            events = await self.execute(root, handler, extra=("--out", str(out)))
            self.assertTrue(events[-1]["out_complete"])
            self.assertEqual(out.read_text(), "a\t3\nb\t7\n")

    async def test_unwritable_output_fails_before_any_task(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_task(root, "a", "Calculate 1 + 2.", 3)
            with self.assertRaises(SystemExit):
                await self.execute(root, lambda _: self.fail("no API call expected"),
                                   extra=("--out", str(root / "missing/answers.txt")))
            self.assertFalse((root / "a/answer.txt").exists())

    async def test_empty_json_content_stops_with_usage_counted_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(4):
                write_task(root, f"job{index}", "Calculate the sum of two and three.", 4-index)
            calls = []

            def handler(request):
                calls.append(request)
                return httpx.Response(200, json={
                    "choices": [{"message": {"content": "  "}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 3}})

            events = await self.execute(root, handler, budget=4000)
            self.assertEqual(len(calls), 2)
            self.assertEqual(events[-1]["spent_tenths"], 260)
            self.assertEqual(events[-1]["unknown_reserved_tenths"], 0)
            self.assertEqual(events[-1]["deferred_unattempted"], 2)
            self.assertFalse(list(root.glob("*/answer.txt")))

    async def test_initial_green_tests_need_contract_review(self):
        for valid in (False, True):
            with self.subTest(valid=valid), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                prompt = ("Return the exact missing value from policy.txt." if not valid else
                          "Return 0 from value().")
                write_task(root, "task", prompt, 3, "coding", "unittest")
                workspace = root / "task/workspace"
                workspace.mkdir()
                (workspace / "app.py").write_text("def value():\n    return 0\n")
                (workspace / "test_app.py").write_text(
                    "import unittest\nfrom app import value\nclass T(unittest.TestCase):\n"
                    "    def test_value(self):\n        self.assertEqual(value(), 0)\n"
                    "if __name__ == '__main__': unittest.main()\n")
                calls = []

                def handler(request):
                    calls.append(request)
                    action = ({"answer": "DONE", "review": "The contract requires zero and value returns zero."}
                              if valid else {"skip": "policy.txt is missing and the expected value is unknown"})
                    return httpx.Response(200, json={
                        "choices": [{"message": {"content": json.dumps(action)}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 100, "completion_tokens": 20}})

                await self.execute(root, handler)
                self.assertEqual(len(calls), int(valid))
                self.assertEqual((root / "task/answer.txt").exists(), valid)

    def test_named_inputs_are_required_but_output_files_are_not(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for question in ("Implement fees(x) under the table in rates.csv.",
                             "Return the identifier from `release.txt`."):
                with self.assertRaises(NoUniqueAnswer):
                    required_inputs(root, question)
            required_inputs(root, "Write the generated report to report.txt.")
            (root / "rates.csv").write_text("x,y\n1,2\n")
            required_inputs(root, "Implement fees(x) under the table in rates.csv.")

    def test_universal_contracts_reject_only_concrete_contradictions(self):
        universal_contract_conflicts("For every integer k, recover(3*k+1) must equal k exactly.")
        universal_contract_conflicts("For every integer k, positive(k*k) must equal k*k exactly.")
        with self.assertRaises(NoUniqueAnswer):
            universal_contract_conflicts("For every integer k, recover(k*k) must equal k exactly.")
        with self.assertRaises(NoUniqueAnswer):
            universal_contract_conflicts("For each integer x, constant(0) must equal x+2.")
        with self.assertRaises(NoUniqueAnswer):
            universal_contract_conflicts("f() must return a built-in dict and must return a built-in tuple for the same call. Both exact type requirements are mandatory.")

    async def test_impossible_universal_contract_never_reaches_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_task(root, "inverse", "For every integer n, decode(n*n) must equal n exactly.", 5, "coding", "unittest")
            events = await self.execute(root, lambda _: self.fail("contradiction must be caught without a model"))
            self.assertEqual(events[-1]["spent_tenths"], 0)
            self.assertFalse((root / "inverse/answer.txt").exists())

    def test_contract_probe_prevents_false_positive_and_keeps_tests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, stage = root / "source", root / "stage"
            source.mkdir()
            stage.mkdir()
            (source / "app.py").write_text("def first_n(values, n):\n    return values[:n]\n")
            test = ("import unittest\nfrom app import first_n\nclass T(unittest.TestCase):\n"
                    "    def test_public(self):\n        self.assertEqual(first_n([1, 2], 1), [1])\n"
                    "if __name__ == '__main__': unittest.main()\n")
            (source / "test_app.py").write_text(test)
            coding = Coding(source, stage, "first_n(values,n) returns an empty list for zero or negative n.")
            self.assertFalse(coding.test()["passed"])
            with self.assertRaises(ValueError):
                coding.commit()
            result = coding.patch("app.py", [["values[:n]", "values[:max(0, n)]"]])
            self.assertTrue(result["passed"])
            coding.commit()
            self.assertEqual((source / "test_app.py").read_text(), test)
            self.assertFalse(list(source.glob("pf_contract*")))
            self.assertFalse(list(stage.glob("pf_contract*")))

    def test_final_fact_uniqueness_and_exact_entity_spelling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, text in {
                "a": "# Ada\nEmployer: A\n", "b": "# Ada\nEmployer: B\n",
                "c": "# A\nCity: Rome\n", "d": "# B\nCity: Rome\n",
                "e": "# ID-X\nBorn: 1990\n", "f": "# id-x\nBorn: 2000\n",
            }.items():
                (root / f"{name}.md").write_text(text)
            self.assertEqual(kb_lookup(root, "Ada", ["Employer", "City"], True), "Rome")
            self.assertEqual(kb_lookup(root, "ID-X", ["Born"], True), "1990")
            (root / "d.md").write_text("# B\nCity: Lima\n")
            with self.assertRaises(NoUniqueAnswer):
                kb_lookup(root, "Ada", ["Employer", "City"], True)
            (root / "d.md").write_text("# B\nOther: unknown\n")
            with self.assertRaises(NoUniqueAnswer):
                kb_lookup(root, "Ada", ["Employer", "City"], True)

    def test_sql_null_and_unknown_csv_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with closing(sqlite3.connect(root / "scores.db")) as db, db:
                db.execute("CREATE TABLE scores(team TEXT, points INTEGER)")
                db.executemany("INSERT INTO scores VALUES (?,?)", [("a", None), ("a", 9), ("b", 8), ("c", None)])
            self.assertEqual(db_query(root, "scores.db", "argmax_sum", "points", table="scores", group="team"), "a")
            (root / "states.csv").write_text("state\nred\nblue\n\"\"\n")
            with self.assertRaises(NoUniqueAnswer):
                db_query(root, "states.csv", "count_distinct", "state")

    def test_latest_revisions_compare_numeric_amounts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "accounts.csv").write_text("account,region\n001,North\n")
            (root / "events.csv").write_text(
                "order,account,revision,event,cents\na,001,2,sale,00042\na,001,2,sale,42\n")
            self.assertEqual(net_ledger(root, "accounts.csv", "events.csv", "North"), 42)
