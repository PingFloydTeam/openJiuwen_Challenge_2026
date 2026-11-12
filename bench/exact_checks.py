"""Independent small exhaustive oracles for exact tools and real workflow wiring."""
from fractions import Fraction
from itertools import combinations, product
import json
from pathlib import Path
import random
from types import SimpleNamespace
import tempfile
import unittest

import httpx

from bench import policy_checks
from bench.policy_checks import write_task
from ping_floyd.domain import NoUniqueAnswer
from ping_floyd.exact_tasks import elapsed_time, integer_inverse, interval_coverage, urn_probability
from ping_floyd.fast import fast_math
from ping_floyd.structured import dated_fact, linked_fact
from ping_floyd.planning import Planner
from ping_floyd.tools import db_query


class ExactChecks(unittest.TestCase):
    def test_ample_budget_does_not_truncate_expensive_structured_arguments(self):
        task = SimpleNamespace(kind="math", instruction="A long exact calculation. " * 10)
        planner = Planner(32000)
        planner.costs["math"] = 25
        self.assertEqual(planner.limit(task, 1000000, 30), 32000)
        self.assertLessEqual(planner.limit(task, 1000, 30), 1000)

    def test_linked_csv_checks_all_alternatives_and_preserves_quoted_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "index.csv").write_text('key,state,path\n"Depot ""A"", north",release,one.txt\n"Depot ""A"", north",release,two.txt\n"Depot ""A"", north\nbackup",release,decoy.txt\n')
            (root / "one.txt").write_text("Service code: SAME\n")
            (root / "two.txt").write_text("Service code: SAME\n")
            (root / "decoy.txt").write_text("Service code: WRONG\n")
            args = dict(file="index.csv", filters={"key": 'Depot "A", north', "state": "release"}, path_column="path", field="Service code")
            self.assertEqual(linked_fact(root, **args), "SAME")
            (root / "two.txt").write_text("Service code: CONFLICT\n")
            with self.assertRaises(NoUniqueAnswer):
                linked_fact(root, **args)

    def test_interval_sweep_against_independent_cell_count(self):
        rng = random.Random(7342)
        for _ in range(120):
            intervals = [sorted(rng.sample(range(-8, 9), 2)) for _ in range(rng.randrange(1, 9))]
            for minimum in range(1, 5):
                cells = sum(sum(a <= tick < b for a, b in intervals) >= minimum for tick in range(-8, 9))
                supplied = [[str(Fraction(a, 3)), str(Fraction(b, 3))] for a, b in intervals]
                self.assertEqual(Fraction(interval_coverage(supplied, minimum)), Fraction(cells, 3))
        self.assertEqual(interval_coverage([[0, 1], [1, 2]], 2), "0")
        self.assertEqual(interval_coverage([[0.25, 0.75], [0.5, 1]], 2), "1/4")
        with self.assertRaises(NoUniqueAnswer):
            interval_coverage([[3, 2]])

    def test_probability_against_enumeration_of_labelled_objects(self):
        for counts in product(range(1, 4), repeat=3):
            labels = [label for label, count in enumerate(counts) for _ in range(count)]
            for draw in range(min(4, len(labels)) + 1):
                samples = list(combinations(range(len(labels)), draw))
                for distinct in range(4):
                    hits = sum(len({labels[i] for i in sample}) == distinct for sample in samples)
                    self.assertEqual(Fraction(urn_probability(list(counts), draw, distinct)), Fraction(hits, len(samples)))
        self.assertEqual(urn_probability([0, 0], 0, 0), "1")
        self.assertEqual(urn_probability({"red": 2, "blue": 1}, 2, 2), "2/3")
        with self.assertRaises(NoUniqueAnswer):
            urn_probability([1, 2], 4, 2)

    def test_integer_inverse_against_forward_enumeration(self):
        rng = random.Random(546)
        options = [("add", -3), ("add", 2), ("mul", -2), ("mul", 3), ("floor_div", 2), ("floor_div", 3)]
        for _ in range(180):
            operations = rng.choices(options, k=3)
            target, modulus, residue = rng.randrange(-4, 5), rng.randrange(1, 5), rng.randrange(-2, 4)
            candidates = []
            for original in range(-200, 201):
                value = original
                for op, arg in operations:
                    value = value + arg if op == "add" else value * arg if op == "mul" else value // arg
                if value == target and (original - residue) % modulus == 0:
                    candidates.append(original)
            if len(candidates) == 1:
                self.assertEqual(integer_inverse(operations, target, modulus, residue), candidates[0])
            else:
                with self.assertRaises(NoUniqueAnswer):
                    integer_inverse(operations, target, modulus, residue)

    def test_time_offsets_leap_day_and_exact_subseconds(self):
        self.assertEqual(elapsed_time("2024-02-28T23:30:00-03:00", "2024-02-29T04:10:00Z"), "100")
        self.assertEqual(elapsed_time("2024-03-01T00:00:00+02:00", "2024-02-29T23:00:00+00:00", "hours"), "1")
        self.assertEqual(elapsed_time("2020-01-01T00:00:00.125Z", "2020-01-01T00:00:00.375Z", "seconds"), "1/4")
        with self.assertRaises(ValueError):
            elapsed_time("2024-01-01T00:00:00", "2024-01-02T00:00:00Z")
        with self.assertRaises(NoUniqueAnswer):
            elapsed_time("2024-01-02T00:00:00Z", "2024-01-01T00:00:00Z")

    def test_as_of_date_ignores_issued_and_rejects_tied_conflicts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            notices = root / "docs"
            notices.mkdir()
            (notices / "a.txt").write_text("Project: X\nIssued: 2030-01-02\nEffective: 2028-03-01\nCode: old\n")
            (notices / "b.txt").write_text("Project: X\nIssued: 2027-01-01\nEffective: 2028-03-02\nCode: new\n")
            (notices / "c.txt").write_text("Project: X\nIssued: 2027-01-01\nEffective: 2028-03-03\nCode: future\n")
            args = dict(directory="docs", filters={"Project": "X"}, date_field="Effective", field="Code", cutoff="2028-03-02")
            self.assertEqual(dated_fact(root, **args), "new")
            (notices / "d.txt").write_text("Project: X\nEffective: 2028-03-02\nCode: conflicting\n")
            with self.assertRaises(NoUniqueAnswer):
                dated_fact(root, **args)
            (root / "notices.csv").write_text("project,effective,code\nX,2028-03-02,A\nX,2028-03-02,B\nX,2028-03-03,C\n")
            args.update(directory="", file="notices.csv")
            with self.assertRaises(NoUniqueAnswer):
                dated_fact(root, **args)

    def test_ratio_of_sums_and_indivisible_people(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "rates.csv").write_text("group,hits,tries\na,1,2\na,2,7\nb,0,0\nc,1,\n")
            args = dict(file="rates.csv", op="ratio_sums", column="hits", factor="tries")
            self.assertEqual(db_query(root, filters=[["group", "=", "a"]], **args), "1/3")
            for group in ["b", "c"]:
                with self.assertRaises(NoUniqueAnswer):
                    db_query(root, filters=[["group", "=", group]], **args)
        template = ("A {}-seat bus is full. There are three times as many adults as children, "
                    "and every seat contains one adult or one child. How many children are seated?")
        self.assertEqual(fast_math(template.format(20)), 5)
        with self.assertRaises(NoUniqueAnswer):
            fast_math(template.format(21))


class ExactWorkflowChecks(unittest.IsolatedAsyncioTestCase):
    execute = policy_checks.OfflinePolicyChecks.execute

    async def test_linked_selection_cannot_submit_first_document_without_index_check(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_task(root, "linked", "Follow the sole released route for key X from index.csv and give its service code. Multiple released rows have equal priority.", 3, "search", "exact_str")
            workspace = root / "linked/workspace"
            workspace.mkdir()
            (workspace / "index.csv").write_text("key,path\nX,a.txt\nX,b.txt\n")
            (workspace / "a.txt").write_text("Service code: A\n")
            (workspace / "b.txt").write_text("Service code: B\n")
            actions = iter([
                {"tool": "read", "args": {"file": "a.txt"}},
                {"answer": "A"},
                {"tool": "linked_fact", "args": {"file": "index.csv", "filters": {"key": "X"}, "path_column": "path", "field": "Service code"}, "final": True},
            ])

            def handler(_):
                return httpx.Response(200, json={
                    "choices": [{"message": {"content": json.dumps(next(actions))}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 30}})

            events = await self.execute(root, handler, budget=5000)
            self.assertFalse((root / "linked/answer.txt").exists())
            self.assertTrue(any(e.get("kind") == "NoUniqueAnswer" for e in events))

    async def test_effective_selection_cannot_submit_arbitrary_matching_code(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_task(root, "notices", "Find the code effective on 2028-03-02 for project X. Use the latest effective date.", 3, "search", "exact_str")
            workspace = root / "notices/workspace"
            workspace.mkdir()
            (workspace / "ledger.csv").write_text("project,effective,code\nX,2028-03-02,A\nX,2028-03-02,B\n")
            actions = iter([
                {"tool": "read", "args": {"file": "ledger.csv"}},
                {"answer": "A"},
                {"tool": "dated_fact", "args": {"file": "ledger.csv", "filters": {"project": "X"}, "date_field": "effective", "field": "code", "cutoff": "2028-03-02"}, "final": True},
                {"skip": "Latest effective records disagree and no tie-break is given."},
            ])

            def handler(_):
                return httpx.Response(200, json={
                    "choices": [{"message": {"content": json.dumps(next(actions))}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 30}})

            events = await self.execute(root, handler, budget=5000)
            self.assertFalse((root / "notices/answer.txt").exists())
            self.assertTrue(any(e.get("kind") == "NoUniqueAnswer" for e in events))

    async def test_dedicated_math_tools_through_real_openjiuwen(self):
        cases = [
            ("interval", "Give the length covered by overlapping intervals.", "interval_coverage", {"intervals": [[0, 4], [2, 6]], "minimum": 2}, "2"),
            ("clock", "Find the elapsed time with UTC offsets.", "elapsed_time", {"start": "2024-01-01T00:00Z", "end": "2024-01-01T01:00Z"}, "60"),
            ("urn", "Draw uniformly without replacement.", "urn_probability", {"counts": [2, 1], "draw": 2, "distinct": 2}, "2/3"),
            ("inverse", "A counter begins with an even number of tokens. Recover the original number after rounding down.", "integer_inverse", {"operations": [["add", 1], ["floor_div", 2]], "target": 3}, "6"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, question, tool, args, expected in cases:
                write_task(root, name, question, 2, "math", "exact_str")

            def handler(request):
                message = json.loads(request.content)["messages"][-1]["content"]
                _, _, tool, args, _ = next(case for case in cases if case[1] in message)
                return httpx.Response(200, json={
                    "choices": [{"message": {"content": json.dumps({"tool": tool, "args": args, "final": True})}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 120, "completion_tokens": 40}})

            await self.execute(root, handler, budget=5000)
            for name, _, _, _, expected in cases:
                self.assertEqual((root / name / "answer.txt").read_text().strip(), expected)
