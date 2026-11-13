"""Check the harder fixtures and tool dispatch without using model quota."""
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import httpx

from bench import policy_checks
from bench.hard import Suite, coding_cases, data_cases, math_cases, search_cases
from bench.suite import private_coding_passes
from bench.transfer import generate as transfer_suite
from ping_floyd.algorithms import grid_paths, weighted_path
from ping_floyd.coding import Coding
from ping_floyd.constraints import solve_congruences, solve_linear
from ping_floyd.domain import NoUniqueAnswer, Task
from ping_floyd.fast import solve_fast
from ping_floyd.input_form import canonical_question, unwrap_question
from ping_floyd.relational import sql_query
from ping_floyd.structured import latest_kb_notice, net_ledger
from ping_floyd.tools import db_query


DATA_QUERIES = {
    "joined_revenue": """SELECT SUM(o.units*o.cents) FROM orders o
        JOIN accounts a ON o.account_id=a.id WHERE a.region='North' AND o.month BETWEEN 2 AND 5""",
    "antijoin": """SELECT COUNT(*) FROM accounts a WHERE a.active=1 AND NOT EXISTS
        (SELECT 1 FROM orders o WHERE o.account_id=a.id AND o.month=1)""",
    "first_order": """WITH ranked AS (SELECT units, ROW_NUMBER() OVER
        (PARTITION BY account_id ORDER BY month,id) AS rn FROM orders)
        SELECT SUM(units) FROM ranked WHERE rn=1""",
    "grouped_ratio": """WITH totals AS (SELECT a.region, SUM(o.units*o.cents) AS revenue,
        SUM(o.units) AS quantity FROM orders o JOIN accounts a ON o.account_id=a.id GROUP BY a.region)
        SELECT region FROM totals t WHERE NOT EXISTS (SELECT 1 FROM totals u WHERE
        u.region<>t.region AND u.revenue*t.quantity >= t.revenue*u.quantity)""",
}


class HardChecks(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.suite = Suite(Path(folder.name) / "suite", 20261110, None)

    def test_exact_constraints(self):
        self.assertEqual(solve_linear(["x+y+z=3/7", "y+z=2/7"], "x"), "1/7")
        with self.assertRaises(NoUniqueAnswer):
            solve_linear(["x+y=3/7", "2*x+2*y=6/7"], "x")
        self.assertEqual(solve_congruences([[5, 6], [5, 8]], -40, 0, first=True), "-19")
        with self.assertRaises(NoUniqueAnswer):
            solve_congruences([[1, 6], [2, 8]], 0, 100)

    def test_ledger_filters_before_revision_and_preserves_text_keys(self):
        root = self.suite.root
        (root / "accounts.csv").write_text("account,region\n007,North\n7,South\n")
        journal = root / "events.csv"
        base = ("order,account,revision,state,event,cents\n"
                "a,007,1,posted,sale,23\na,007,2,draft,sale,80\n"
                "b,7,1,posted,sale,14\nc,007,1,posted,refund,5\n")
        journal.write_text(base)
        with self.assertRaisesRegex(ValueError, "state column present"):
            net_ledger(root, "accounts.csv", "events.csv", "North", exclude_states=None)
        self.assertEqual(net_ledger(root, "accounts.csv", "events.csv", "North", exclude_states=["draft"]), 18)
        journal.write_text(base + "d,008,1,posted,sale,4\n")
        with self.assertRaisesRegex(NoUniqueAnswer, "unresolved region"):
            net_ledger(root, "accounts.csv", "events.csv", "North", exclude_states=["draft"])

    def test_group_ties_require_an_explicit_rule(self):
        root = self.suite.root
        (root / "bids.csv").write_text("bidder,amount,day\nZed,8,1\nAlfa,3,2\nAlfa,5,3\nOutside,90,4\n")
        args = (root, "bids.csv", "argmax_sum", "amount", [["day", "<=", 3]])
        with self.assertRaises(NoUniqueAnswer):
            db_query(*args, group="bidder")
        self.assertEqual(db_query(*args, group="bidder", tie_break="alphabetical_first"), "Alfa")

    def test_latest_notice_checks_scope_approval_and_conflicting_ties(self):
        root = self.suite.root
        kb = root / "kb"
        kb.mkdir()

        def notice(name, date, code, status="approved", scope="local"):
            (kb / (name + ".md")).write_text(f"# {name}\nProject: Aspen\nStatus: {status}\nScope: {scope}\n"
                                           f"Date: {date}\nContact code: {code}\n")

        notice("old", "2025-01-01", "X1")
        notice("new", "2025-01-02", "X2")
        notice("draft", "2025-01-03", "X3", status="draft")
        notice("global", "2025-01-04", "X4", scope="global")
        args = (root, "kb", "Aspen", "approved", "local", "Contact code")
        self.assertEqual(latest_kb_notice(*args), "X2")
        notice("copy", "2025-01-02", "X2")
        self.assertEqual(latest_kb_notice(*args), "X2")
        notice("conflict", "2025-01-02", "X5")
        with self.assertRaisesRegex(NoUniqueAnswer, "unique value"):
            latest_kb_notice(*args)

    def test_question_conversion_preserves_constraints_and_literals(self):
        instruction = "## Условие:\n• Длина прямоугольника 12 м, ширина 5 м.\n• Найдите площадь в квадратных метрах."
        task = Task(self.suite.root, "math", instruction, 1, "exact_int")
        self.assertEqual(solve_fast(task), "60")
        modified = instruction + "\n• Исключите из результата 3 квадратных метра."
        self.assertIsNone(solve_fast(Task(self.suite.root, "math", modified, 1, "exact_int")))
        self.assertIn("- x", unwrap_question("Условие:\nSolve over the rationals:\n- x=2. Return x."))
        self.assertEqual(unwrap_question('Вопрос:\n"first\n    second"'), 'Вопрос:\n"first\n    second"')
        source = "Откройте `docs/Ä-note.txt`. Какое значение указано в поле `Service code`?"
        self.assertEqual(canonical_question(source, "search"), "What is the Service code recorded in docs/Ä-note.txt?")

    def test_hard_math_and_search_fast_path(self):
        math_cases(self.suite)
        search_cases(self.suite)
        for name, answer in self.suite.oracle.items():
            with self.subTest(task=name):
                task = Task.load(self.suite.root / "tasks" / name / "task.json")
                if answer is None:
                    # Unsupported field mappings must remain eligible for LLM
                    # fallback; they do not prove that the whole task is impossible.
                    try:
                        solve_fast(task)
                    except NoUniqueAnswer:
                        pass
                    except ValueError as error:
                        self.assertIn('unsupported field mapping',str(error))
                    else:
                        self.fail('ambiguous task must not produce a fast answer')
                else:
                    self.assertEqual(str(solve_fast(task)), str(answer))

    def test_transfer_algorithms_and_independent_oracles(self):
        root = self.suite.root / "transfer"
        transfer_suite(root, 20261111)
        for name, answer in json.loads((root / "oracle.json").read_text()).items():
            if "coding" in name:
                continue
            with self.subTest(task=name):
                task = Task.load(root / "tasks" / name / "task.json")
                if answer is None:
                    with self.assertRaises(NoUniqueAnswer):
                        solve_fast(task)
                else:
                    self.assertEqual(str(solve_fast(task)), str(answer))
        self.assertEqual(grid_paths(10, 10, [[1, 1]]), 0)
        self.assertEqual(weighted_path([("A", "B", -3), ("B", "C", -2)], "A", "C", "longest_dag"), -5)
        with self.assertRaises(ValueError):
            weighted_path([("A", "B", -3)], "A", "B")

    def test_all_relational_cases(self):
        math_cases(self.suite)
        search_cases(self.suite)
        data_cases(self.suite)
        for name, entry in self.suite.cases.items():
            if entry["family"] not in DATA_QUERIES:
                continue
            with self.subTest(task=name):
                root = self.suite.root / "tasks" / name / "workspace"
                self.assertEqual(sql_query(root, "commerce.sqlite", DATA_QUERIES[entry["family"]]),
                                 self.suite.oracle[name])
        case = self.suite.case_id("data_analysis", "indeterminate", 0)
        root = self.suite.root / "tasks" / case / "workspace"
        db = "commerce.sqlite"
        self.assertEqual(sql_query(root, db, "SELECT COUNT(*) FROM orders"), 2)
        for statement in (DATA_QUERIES["grouped_ratio"], "SELECT account_id FROM orders WHERE month=2",
                          "SELECT AVG(units) FROM orders WHERE month=12"):
            with self.assertRaises(NoUniqueAnswer):
                sql_query(root, db, statement)
        for statement in ("DELETE FROM orders", "ATTACH DATABASE '/tmp/pf-other.sqlite' AS other",
                          "PRAGMA table_info(orders)", "EXPLAIN SELECT 1"):
            with self.assertRaises(ValueError):
                sql_query(root, db, statement)
        for statement in ("WITH a AS (SELECT 1) DELETE FROM orders", "SELECT load_extension('missing')",
                          "SELECT tax_cents FROM orders", "SELECT COUNT(*) FROM payments"):
            with self.assertRaises(sqlite3.DatabaseError):
                sql_query(root, db, statement)
        with self.assertRaises(ValueError):
            sql_query(root, db, "SELECT a.id FROM accounts a, orders b, accounts c, orders d, accounts e", "rows")

    async def test_sql_joins_csv_and_tsv_with_integer_and_null_columns(self):
        root = self.suite.root / "tasks"
        root.mkdir()
        policy_checks.write_task(root, "join", "Combine orders.csv and accounts.tsv to total North amounts.",
                                 3, "data_analysis")
        workspace = root / "join/workspace"
        workspace.mkdir()
        (workspace / "orders.csv").write_text("account,amount\n001,10\n001,\n002,20\n1,90\n")
        (workspace / "accounts.tsv").write_text("id\tregion\n001\tNorth\n002\tSouth\n1\tSouth\n")
        tables = {"orders": "orders.csv", "accounts": "accounts.tsv"}
        integers = {"orders": ["amount"]}
        query = "SELECT SUM(o.amount) FROM orders o JOIN accounts a ON o.account=a.id WHERE a.region='North'"

        def handler(request):
            prompt = json.loads(request.content)["messages"][-1]["content"]
            self.assertIn('"accounts.tsv":["id","region"]', prompt)
            action = {"tool": "sql", "args": {"statement": query, "tables": tables, "integers": integers}, "final": True}
            return httpx.Response(200, json={
                "choices": [{"message": {"content": json.dumps(action)}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10},
            })

        await policy_checks.OfflinePolicyChecks.execute(self, root, handler)
        self.assertEqual((root / "join/answer.txt").read_text(), "10\n")
        self.assertEqual(sql_query(workspace, "", "SELECT COUNT(*) FROM orders WHERE amount IS NULL",
                                   tables=tables, integers=integers), 1)
        with self.assertRaises(sqlite3.DatabaseError):
            sql_query(workspace, "", "WITH x AS (SELECT 1) DELETE FROM orders", tables=tables)
        with self.assertRaises(ValueError):
            sql_query(workspace, "", query, tables={"orders": "../orders.csv"})

    async def test_openjiuwen_sql_and_two_file_patch(self):
        data_cases(self.suite)
        coding_cases(self.suite)
        root = self.suite.root / "tasks"
        coding = self.suite.case_id("coding", "decode", 0)
        data = self.suite.case_id("data_analysis", "joined_revenue", 0)
        for task in root.iterdir():
            if task.name not in {coding, data}:
                (task / "task.json").unlink()
        coding_root = root / coding / "workspace"
        with tempfile.TemporaryDirectory() as stage:
            staged = Coding(coding_root, Path(stage), "")
            with self.assertRaises(ValueError):
                staged.patch_files({"rules.py": [["(count - 1)", "count"]], "pipeline.py": [["absent", "x"]]})
            self.assertEqual((Path(stage) / "rules.py").read_bytes(), (coding_root / "rules.py").read_bytes())
            self.assertFalse(staged.changed)

        def handler(request):
            payload = json.loads(request.content)
            if "Expand run-length pairs" in payload["messages"][-1]["content"]:
                action = {"tool": "patch_files", "args": {"patches": {
                    "rules.py": [["(count - 1)", "count"]], "pipeline.py": [["pairs[1:]", "pairs"]],
                }}}
                if "Recent tool results:" in payload["messages"][-1]["content"]:
                    action = {"answer": "DONE", "review": "All run-length pairs are expanded using their exact counts, including the first pair and zero counts."}
            else:
                self.assertIn('"orders":["id","account_id","month","units","cents"]',
                              payload["messages"][-1]["content"])
                action = {"tool": "sql", "args": {"file": "commerce.sqlite",
                          "statement": DATA_QUERIES["joined_revenue"]}, "final": True}
            return httpx.Response(200, json={
                "choices": [{"message": {"content": json.dumps(action)}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10},
            })

        events = await policy_checks.OfflinePolicyChecks.execute(self, root, handler, budget=5000,
                                                                 extra=("--tool-profile", "compact"))
        self.assertEqual({e["tool"] for e in events if e["event"] == "tool"}, {"sql", "patch_files"})
        self.assertEqual((root / data / "answer.txt").read_text(), f"{self.suite.oracle[data]}\n")
        self.assertEqual((root / coding / "answer.txt").read_text(), "DONE\n")
        self.assertTrue(private_coding_passes(self.suite.root, coding, self.suite.cases[coding]))
        self.assertEqual(events[-1]["spent_tenths"], 400)

    async def test_failed_code_edit_limit_keeps_original_sources(self):
        coding_cases(self.suite)
        root = self.suite.root / "tasks"
        name = self.suite.case_id("coding", "decode", 0)
        for folder in root.iterdir():
            if folder.name != name:
                (folder / "task.json").unlink()
        source = root / name / "workspace/rules.py"
        original = source.read_bytes()

        def handler(_request):
            action = {"tool": "patch", "args": {"file": "rules.py", "edits": [["count - 1", "count + 1"]]}}
            return httpx.Response(200, json={
                "choices": [{"message": {"content": json.dumps(action)}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10},
            })

        events = await policy_checks.OfflinePolicyChecks.execute(self, root, handler, budget=5000,
                                                                 extra=("--repair-limit", "1"))
        self.assertEqual(source.read_bytes(), original)
        self.assertFalse((root / name / "answer.txt").exists())
        self.assertEqual([event["route"] for event in events if event["event"] == "task"], ["repair_limit"])
        self.assertEqual(events[-1]["spent_tenths"], 200)


if __name__ == "__main__":
    unittest.main()
