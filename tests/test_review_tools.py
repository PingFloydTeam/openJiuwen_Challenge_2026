"""Independent regression contracts: failures describe behavior to repair, not xfails.

No API key, external network, contest expected fields, or production mutations.
Run in the contest Linux image; see reviews/2026-11-12/README.md.
"""
from fractions import Fraction
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from ping_floyd.coding import Coding, test_run
from ping_floyd.contracts import required_inputs, universal_contract_conflicts
from ping_floyd.domain import NoUniqueAnswer, Task
from ping_floyd.fast import solve_fast
from ping_floyd.knowledge import kb_route
from ping_floyd.repair import try_repair
from ping_floyd.tools import calc, db_query


def put(root, name, content):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def public_test(expression, expected):
    return ("import unittest\nimport mylib\nclass Public(unittest.TestCase):\n"
            f"    def test_result(self):\n        self.assertEqual({expression}, {expected})\n"
            "if __name__ == '__main__': unittest.main()\n")


class IndependentToolContracts(unittest.TestCase):
    def test_control_exact_arithmetic_and_decimal_aggregation(self):
        self.assertEqual(Fraction(calc("0.1 + 0.2")), Fraction(3, 10))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            put(root, "sales.csv", "region,units,price\nnorth,3,0.1\nsouth,99,9\nnorth,2,0.2\n")
            self.assertEqual(Fraction(db_query(root, "sales.csv", "sum_product", "units",
                                              [["region", "=", "north"]], factor="price")), Fraction(7, 10))

    def test_zero_quantity_makes_unknown_finite_price_irrelevant(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            put(root, "sales.csv", "region,units,price\nNorth,0,\nNorth,2,3\n")
            try:
                value = db_query(root, "sales.csv", "sum_product", "units",
                                 [["region", "=", "North"]], factor="price")
            except NoUniqueAnswer as error:
                self.fail(f"Zero times any unknown finite price contributes exactly zero: {error}")
            self.assertEqual(Fraction(value), 6)

    def test_direct_shortest_route_does_not_need_an_unrelated_missing_record(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            put(root, "a.md", "# A\nNext: B\nBackup: Missing\n")
            put(root, "b.md", "# B\n")
            try:
                distance = kb_route(root, "A", "B", ["Next", "Backup"], "distance", strict=True)
            except NoUniqueAnswer as error:
                self.fail(f"The present A -> B edge proves the shortest distance is one: {error}")
            self.assertEqual(distance, 1)

    def test_birth_field_alias_is_not_proof_of_missing_information(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            put(root, "workspace/kb/ada.md", "# Ada Reed\nBirth year: 1983\n")
            task = Task(root, "search", "According to kb/, In which year was Ada Reed born?", 5, "exact_str")
            try:
                result = solve_fast(task)
            except NoUniqueAnswer as error:
                self.fail(f"A known birth year must be solved or deferred, not proven impossible: {error}")
            self.assertIn(result, (None, "1983"))

    def test_birth_date_must_not_be_submitted_as_birth_year(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            put(root, "workspace/kb/ada.md", "# Ada Reed\nBorn: 1983-04-17\n")
            task = Task(root, "search", "According to kb/, In which year was Ada Reed born?", 5, "exact_str")
            self.assertIn(solve_fast(task), (None, "1983"), "Extract the year or defer to the model")

    def test_runtime_fixture_is_not_a_missing_task_input(self):
        with tempfile.TemporaryDirectory() as folder:
            instruction = ("Implement load_total(path), reading integer values from numbers.txt supplied by "
                           "the caller at runtime. The tests create the file in a temporary directory; "
                           "no numbers.txt is included in the workspace. Return the sum of its lines.")
            try:
                required_inputs(Path(folder), instruction)
            except NoUniqueAnswer as error:
                self.fail(f"A runtime fixture is part of the function contract, not unavailable evidence: {error}")

    def test_stateful_counter_examples_are_not_a_logical_contradiction(self):
        instruction = ("Implement next_id() using a module-level counter initially zero. Each call increments it "
                       "and returns the new value. These assertions execute in sequence: "
                       "`assert next_id() == 1`, then `assert next_id() == 2`.")
        try:
            universal_contract_conflicts(instruction)
        except NoUniqueAnswer as error:
            self.fail(f"Sequential calls to an explicitly stateful counter may differ: {error}")

    @unittest.skipUnless(sys.platform == "linux", "requires the actual Linux seccomp worker")
    def test_regular_stdlib_import_is_usable_in_coding_tests(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            put(root, "mylib.py", "import tomllib\ndef parse_count(text):\n    return tomllib.loads(text)['count']\n")
            put(root, "test_mylib.py", public_test("mylib.parse_count('count = 7')", "7"))
            reference = subprocess.run([sys.executable, "-I", "-S", "-c",
                                        "import sys; sys.path.insert(0, '.'); import runpy; runpy.run_path('test_mylib.py', run_name='__main__')"],
                                       cwd=root, capture_output=True, text=True, timeout=10)
            self.assertEqual(reference.returncode, 0, reference.stderr)
            actual = test_run(root, "test_mylib.py")
            self.assertTrue(actual["passed"], actual["output"])

    @unittest.skipUnless(sys.platform == "linux", "requires the actual Linux seccomp worker")
    def test_coding_can_read_its_own_read_only_fixture(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            put(root, "numbers.txt", "2\n3\n")
            put(root, "mylib.py", "def total_file(path):\n    with open(path) as stream:\n        return sum(map(int, stream))\n")
            put(root, "test_mylib.py", public_test("mylib.total_file('numbers.txt')", "5"))
            actual = test_run(root, "test_mylib.py")
            self.assertTrue(actual["passed"], actual["output"])

    @unittest.skipUnless(sys.platform == "linux", "requires the actual Linux seccomp worker")
    def test_accepted_fast_repair_respects_explicit_empty_input_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            workspace, stage = root / "workspace", root / "stage"
            workspace.mkdir()
            stage.mkdir()
            instruction = ("Fix average(values) in mylib.py. Return the arithmetic mean for a nonempty list "
                           "and return 0 for an empty list. Run `python3 test_mylib.py`. Do not edit tests.")
            put(workspace, "mylib.py", "def average(values):\n    return sum(values) // len(values)\n")
            public = public_test("mylib.average([1, 2])", "1.5")
            put(workspace, "test_mylib.py", public)
            coding = Coding(workspace, stage, instruction)
            initial = coding.test()
            self.assertFalse(initial["passed"])
            accepted = try_repair(coding, initial["output"], instruction)
            self.assertEqual((stage / "test_mylib.py").read_text(), public)
            if accepted:
                # Independent hidden check, only after the repair claims completion.
                namespace = {}
                exec(compile((stage / "mylib.py").read_text(), "mylib.py", "exec"), namespace)
                try:
                    actual = namespace["average"]([])
                except Exception as error:
                    self.fail(f"Accepted repair violates explicit average([]) == 0: {type(error).__name__}")
                self.assertEqual(actual, 0)


if __name__ == "__main__":
    unittest.main()
