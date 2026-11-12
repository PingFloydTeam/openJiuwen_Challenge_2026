"""Exercise the actual scheduler and proxy accounting with no sockets."""
import asyncio
from collections import Counter
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import signal
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

import httpx

import run_agent
from ping_floyd.coding import Coding
from ping_floyd.proxy import Proxy
from ping_floyd.tools import calc, db_query


def write_task(root, name, instruction, reward, kind="math", check="exact_int"):
    folder = root / name
    folder.mkdir()
    (folder / "task.json").write_text(json.dumps({
        "task_id": name, "task_type": kind, "check": check,
        "instruction": instruction, "reward": reward,
    }))


def calculator(expr, final=False):
    return {"tool": "calc", "args": {"expr": expr}, "final": final}


class OfflinePolicyChecks(unittest.IsolatedAsyncioTestCase):
    async def execute(self, root, handler, budget=1000, extra=()):
        original_client = httpx.AsyncClient
        with patch.dict(os.environ, {"LLM_API_KEY": "local-test", "LLM_BASE_URL": "http://offline.invalid/v1"}):
            with patch("sys.argv", ["run_agent.py", "--tasks-dir", str(root), "--budget", str(budget), *extra]):
                args = run_agent.options()

            def proxy_factory(*args):
                with patch("ping_floyd.proxy.httpx.AsyncClient",
                           lambda **kwargs: original_client(transport=httpx.MockTransport(handler), **kwargs)):
                    return Proxy(*args)

            output = io.StringIO()
            with patch("run_agent.Proxy", side_effect=proxy_factory), redirect_stdout(output):
                await run_agent.main(args)
            return [json.loads(line) for line in output.getvalue().splitlines() if line.startswith("{")]

    async def test_local_first_rewards_and_abstention(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_task(root, "local", "Calculate 20 / 5.", 1)
            write_task(root, "impossible", "Evaluate exactly 20 / 0.", 5)
            write_task(root, "high", "Highest reward fallback. Subtract two from ten.", 5)
            write_task(root, "low", "Repeated fallback. Return the sum of one and two.", 1)
            write_task(root, "wrong", "Unsupported fallback. How much is thirteen?", 2)
            calls = []
            per_case = Counter()

            def handler(request):
                self.assertEqual((root / "local" / "answer.txt").read_text(), "4\n")
                self.assertFalse((root / "impossible" / "answer.txt").exists())
                prompt = json.loads(request.content)["messages"][-1]["content"]
                if "Highest reward" in prompt:
                    key, action = "high", calculator("10-2", True)
                elif "Repeated fallback" in prompt:
                    key, action = "low", calculator("1+2")
                else:
                    key = "wrong"
                    action = calculator("13") if not per_case[key] else {"answer": "99"}
                calls.append(key)
                per_case[key] += 1
                return httpx.Response(200, json={
                    "choices": [{"message": {"content": json.dumps(action)}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 10},
                })

            events = await self.execute(root, handler)
            self.assertEqual(calls, ["high", "wrong", "wrong", "wrong", "low", "low", "low"])
            self.assertEqual((root / "high" / "answer.txt").read_text(), "8\n")
            self.assertFalse((root / "low" / "answer.txt").exists())
            self.assertFalse((root / "wrong" / "answer.txt").exists())
            tasks = {event["task"]: event for event in events if event.get("event") == "task"}
            self.assertEqual(tasks["impossible"]["route"], "proven_skip")
            self.assertEqual(tasks["wrong"]["route"], "no_progress")
            self.assertEqual(tasks["low"]["route"], "no_progress")
            self.assertEqual(events[-1]["spent_tenths"], len(calls) * 200)

    async def test_small_budget_still_saves_local_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_task(root, "local", "Calculate 1 + 2.", 1)
            write_task(root, "model", "Add the integers eight and two.", 5)

            def handler(_request):
                self.fail("no model call can fit into one weighted token")

            events = await self.execute(root, handler, budget=1)
            self.assertEqual((root / "local" / "answer.txt").read_text(), "3\n")
            self.assertEqual(events[-1]["deferred_unattempted"], 1)
            self.assertEqual(events[-1]["spent_tenths"], 0)

    async def test_shortlist_expands_to_compact_menu_after_invalid_action(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_task(root, "data", "A custom task. Compute the sum of two and three.", 4,
                       kind="data_analysis")
            requests = []

            def handler(request):
                system = json.loads(request.content)["messages"][0]["content"]
                requests.append(system)
                action = ({"tool": "unsupported", "args": {}} if len(requests) == 1
                          else calculator("2+3", True))
                return httpx.Response(200, json={
                    "choices": [{"message": {"content": json.dumps(action)}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 10},
                })

            events = await self.execute(root, handler, extra=("--tool-profile", "shortlist"))
            self.assertEqual(len(requests), 2)
            self.assertNotIn("\ncalc:", requests[0])
            self.assertIn("\ncalc:", requests[1])
            self.assertEqual((root / "data/answer.txt").read_text(), "5\n")
            self.assertEqual([e["reason"] for e in events if e["event"] == "tool_menu_expanded"],
                             ["action_error"])

    async def test_http_429_stops_new_calls_and_preserves_answers_for_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_task(root, "local", "Calculate 10-3.", 1)
            write_task(root, "first", "First fallback case. Return five.", 5)
            write_task(root, "second", "Second fallback case. Return five.", 4)
            write_task(root, "third", "Third fallback case. Return five.", 3)
            out = root / "output"
            out.mkdir()
            snapshot = out / "answers.txt"
            (root / "local" / "task.json").write_text(
                (root / "local" / "task.json").read_text().replace('"local"', '"external-id"'))

            def success():
                return httpx.Response(200, json={
                    "choices": [{"message": {"content": json.dumps(calculator("2+3", True))}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 10},
                })

            calls = iter([success(), httpx.Response(429)])

            def handler(_request):
                response = next(calls)
                expected = "external-id\t7\n" if response.status_code == 200 else "external-id\t7\nfirst\t5\n"
                self.assertEqual(snapshot.read_text(), expected)
                return response

            with patch.dict(os.environ, {"OUT": str(out)}):
                events = await self.execute(root, handler)
            self.assertEqual(events[-1]["model_attempted"], 2)
            self.assertEqual(events[-1]["deferred_unattempted"], 1)
            self.assertGreater(events[-1]["unknown_reserved_tenths"], 0)
            self.assertEqual((root / "local/answer.txt").read_text(), "7\n")
            self.assertEqual((root / "first/answer.txt").read_text(), "5\n")
            self.assertFalse((root / "second/answer.txt").exists())
            self.assertFalse((root / "third/answer.txt").exists())

            snapshot.unlink()
            resumed = await self.execute(root, lambda _: success(), budget=events[-1]["remaining_tenths"] // 10,
                                         extra=("--out", str(snapshot)))
            self.assertEqual({event["task"] for event in resumed if event["event"] == "existing_answer"}, {"first", "local"})
            self.assertEqual(resumed[-1]["submitted"], 2)
            self.assertEqual((root / "second/answer.txt").read_text(), "5\n")
            self.assertEqual((root / "third/answer.txt").read_text(), "5\n")
            self.assertEqual(snapshot.read_text(), "external-id\t7\nfirst\t5\nsecond\t5\nthird\t5\n")

    async def test_missing_usage_reserves_tokens_and_silent_proxy_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(4):
                write_task(root, f"job{index}", "Calculate the sum of two and three.", 5 - index)
            calls = 0

            def handler(request):
                nonlocal calls
                calls += 1
                if calls == 1:
                    return httpx.Response(200, json={"choices": [{
                        "message": {"content": json.dumps(calculator("2+3", True))}, "finish_reason": "stop"
                    }]})
                if calls == 3:
                    return httpx.Response(200)
                raise httpx.ReadTimeout("no response", request=request)

            events = await self.execute(root, handler, budget=4000)
            llm = [event for event in events if event["event"] == "llm"]
            self.assertEqual(calls, 3)
            self.assertTrue(llm[0]["usage_unavailable"])
            self.assertEqual(llm[-1]["proxy_disabled"], "consecutive_no_reply")
            self.assertEqual((root / "job0/answer.txt").read_text(), "5\n")
            self.assertEqual(events[-1]["deferred_unattempted"], 1)
            self.assertEqual(events[-1]["spent_tenths"], 0)
            self.assertEqual(events[-1]["unknown_reserved_tenths"], sum(e["reserved_tenths"] for e in llm))

    async def test_sigterm_preserves_saved_answers_and_reserves_inflight_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_task(root, "local", "Calculate 1 + 2.", 1)
            write_task(root, "inflight", "Compute an opaque first fallback.", 5)
            write_task(root, "pending", "Compute an opaque second fallback.", 2)
            calls = 0

            async def handler(_request):
                nonlocal calls
                calls += 1
                os.kill(os.getpid(), signal.SIGTERM)
                await asyncio.sleep(10)
                self.fail("inflight call was not cancelled after the shutdown grace period")

            events = await self.execute(root, handler, budget=1000)
            self.assertEqual(calls, 1)
            self.assertEqual((root / "local/answer.txt").read_text(), "3\n")
            self.assertFalse((root / "inflight/answer.txt").exists())
            self.assertFalse((root / "pending/answer.txt").exists())
            summary = events[-1]
            self.assertEqual(summary["stop_reason"], "SIGTERM")
            self.assertEqual(summary["cancelled"], 1)
            self.assertEqual(summary["deferred_unattempted"], 1)
            self.assertGreater(summary["unknown_reserved_tenths"], 0)

    async def test_second_model_runs_after_first_pass_with_its_own_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_task(root, "stalls", "Ambiguous route. Calculate three times four.", 5)
            write_task(root, "works", "Usual route. Calculate four plus three.", 2)
            calls = []
            second = "qwen3.8-27b-noreason"

            def handler(request):
                payload = json.loads(request.content)
                model = payload["model"]
                stalled = "Ambiguous route" in payload["messages"][-1]["content"]
                calls.append((model, stalled))
                if model == second:
                    self.assertEqual((root / "works" / "answer.txt").read_text(), "7\n")
                    self.assertLessEqual(payload["max_tokens"], 250)
                action = calculator("3*4", model == second) if stalled else calculator("4+3", True)
                return httpx.Response(200, json={
                    "choices": [{"message": {"content": json.dumps(action)}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 10},
                })

            first = "qwen3.6-35b-a3b-noreason"
            events = await self.execute(root, handler, budget=2500,
                                        extra=("--model", first, "--retry-model", second))
            self.assertEqual(calls, [(first, True)] * 3 + [(first, False), (second, True)])
            self.assertEqual((root / "stalls" / "answer.txt").read_text(), "12\n")
            self.assertEqual(events[-1]["model_retried"], 1)

    async def test_concurrent_calls_share_reservations(self):
        for token_budget, expected_peak in ((3000, 5), (250, 1)):
            with self.subTest(token_budget=token_budget), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for i in range(5):
                    write_task(root, f"fallback{i}", f"Compute fallback case {i} with the calculator.", i+1)
                held = paid = active = peak = 0

                async def handler(request):
                    nonlocal held, paid, active, peak
                    payload = json.loads(request.content)
                    reservation = sum(len(m["content"].encode()) for m in payload["messages"]) + 128
                    reservation += 10 * payload["max_tokens"]
                    held += reservation
                    active += 1
                    peak = max(peak, active)
                    self.assertLessEqual(held + paid, token_budget * 10)
                    await asyncio.sleep(0.02)
                    held -= reservation
                    paid += 200
                    active -= 1
                    action = calculator("3*4", True)
                    return httpx.Response(200, json={
                        "choices": [{"message": {"content": json.dumps(action)}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 100, "completion_tokens": 10},
                    })

                events = await self.execute(root, handler, budget=token_budget, extra=("--workers", "5"))
                self.assertEqual(peak, expected_peak)
                self.assertEqual(events[-1]["submitted"], 5)
                self.assertEqual(events[-1]["spent_tenths"], 1000)
                self.assertEqual([event["cost_tenths"] for event in events if event["event"] == "task"], [200] * 5)

    async def test_temporary_reservation_does_not_reduce_the_next_task_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_task(root, "slow", "Opaque slow call.", 5)
            write_task(root, "quick", "Opaque quick call.", 4)
            write_task(root, "large", "Opaque long call. " + "Extra context. " * 160, 3)
            order = []

            async def handler(request):
                prompt = json.loads(request.content)["messages"][-1]["content"]
                if "slow" in prompt:
                    await asyncio.sleep(0.15)
                    order.append("slow_finished")
                elif "quick" in prompt:
                    order.append("quick_finished")
                else:
                    order.append("large_started")
                return httpx.Response(200, json={
                    "choices": [{"message": {"content": json.dumps(calculator("2+3", True))}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 10},
                })

            events = await self.execute(root, handler, budget=720, extra=("--workers", "2", "--queue", "reward"))
            self.assertEqual(order, ["quick_finished", "slow_finished", "large_started"])
            self.assertEqual(events[-1]["submitted"], 3)
            self.assertEqual(events[-1]["spent_tenths"], 600)

    async def test_search_does_not_submit_intermediate_office_as_city(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_task(root, "office", "In kb/, what city is listed for the office of the mentor of Lea Park?",
                       3, "search", "exact_str")
            kb = root / "office/workspace/kb"
            kb.mkdir(parents=True)
            (kb / "lea.md").write_text("# Lea Park\nMentor: Leo Hall\n")
            (kb / "leo.md").write_text("# Leo Hall\nOffice: Redwood Annex\n")
            (kb / "annex.md").write_text("# Redwood Annex\nCity: Oslo\n")
            actions = iter([
                {"tool": "lookup", "args": {"entity": "Lea Park", "fields": ["Mentor", "Office"]}, "final": True},
                {"answer": "Redwood Annex"},
                {"tool": "lookup", "args": {"entity": "Lea Park", "fields": ["Mentor", "Office", "City"]}, "final": True},
            ])

            def handler(_request):
                return httpx.Response(200, json={
                    "choices": [{"message": {"content": json.dumps(next(actions))}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 10},
                })

            events = await self.execute(root, handler)
            self.assertEqual((root / "office/answer.txt").read_text(), "Oslo\n")
            self.assertEqual(sum(e["event"] == "tool_error" for e in events), 1)

    async def test_inventory_fallback_must_use_the_stated_closing_total(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_task(root, "stock", "The two stocks had 20 boxes in all. First gained 3 boxes; second lost 2 boxes. "
                       "The first ended with double the second. The closing total was 22. Give the initial first stock.", 3)
            actions = iter([
                calculator("13", True),
                {"tool": "balance", "args": {"total": 20, "first_change": 3, "second_change": -2, "ratio": 2}, "final": True},
                {"tool": "balance", "args": {"total": 20, "first_change": 3, "second_change": -2, "ratio": 2,
                                               "closing": 22}, "final": True},
                {"skip": "Closing stock contradicts the other facts."},
            ])

            def handler(_request):
                return httpx.Response(200, json={
                    "choices": [{"message": {"content": json.dumps(next(actions))}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 10},
                })

            events = await self.execute(root, handler)
            self.assertFalse((root / "stock/answer.txt").exists())
            errors = [event["kind"] for event in events if event["event"] == "tool_error"]
            self.assertEqual(errors, ["ValueError", "NoUniqueAnswer"])

    async def test_search_uses_named_subject_and_rejects_distractor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_task(root, "contact", "Find a headquarters city for the employer of Nora Finch in kb/.",
                       2, "search", "exact_str")
            kb = root / "contact/workspace/kb"
            kb.mkdir(parents=True)
            (kb / "person.md").write_text("# Nora Finch\nEmployer: Red Studio\n")
            (kb / "company.md").write_text("# Red Studio\nHeadquarters: Lyon\n")
            (kb / "other.md").write_text("# Blue Studio\nHeadquarters: Quito\n")
            actions = iter([
                json.dumps({"tool": "read", "args": {"file": "kb/other.md"}}),
                json.dumps({"answer": "Quito"}),
                'I will look up the named person.\n```json\n' + json.dumps({
                    "name": "lookup", "arguments": {"entity": "Nora Finch", "fields": ["Employer", "Headquarters"]},
                    "final": True}) + '\n```',
            ])

            def handler(request):
                prompt = json.loads(request.content)["messages"][-1]["content"]
                self.assertIn('"entity":"Nora Finch"', prompt)
                self.assertIn('"entity":"Red Studio"', prompt)
                self.assertNotIn('"entity":"Blue Studio"', prompt)
                return httpx.Response(200, json={
                    "choices": [{"message": {"content": next(actions)}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 10},
                })

            events = await self.execute(root, handler)
            self.assertEqual((root / "contact/answer.txt").read_text(), "Lyon\n")
            self.assertEqual(sum(e["event"] == "tool_error" for e in events), 1)

    async def test_malformed_named_table_is_not_sent_to_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_task(root, "table", "Using only sales.csv, what is the sum of amount?",
                       2, "data_analysis")
            workspace = root / "table/workspace"
            workspace.mkdir()
            (workspace / "sales.csv").write_text("id,amount,amount\n1,12,30\n")

            def handler(_request):
                self.fail("a failed integrity check must not be bypassed by the model")

            events = await self.execute(root, handler)
            self.assertFalse((root / "table/answer.txt").exists())
            self.assertIn("invalid_input", [e.get("route") for e in events])
            self.assertEqual(events[-1]["spent_tenths"], 0)

    async def test_read_final_cannot_submit_entire_excerpt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_task(root, "notice", "What is the service code recorded in duty.txt?", 2, "search", "exact_str")
            workspace = root / "notice/workspace"
            workspace.mkdir()
            (workspace / "duty.txt").write_text("For the evening shift, the service code is P61.\n")
            actions = iter([
                {"tool": "read", "args": {"file": "duty.txt"}, "final": True},
                {"answer": "P61"},
            ])

            def handler(_request):
                self.assertFalse((root / "notice/answer.txt").exists())
                return httpx.Response(200, json={
                    "choices": [{"message": {"content": json.dumps(next(actions))}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 10},
                })

            events = await self.execute(root, handler)
            self.assertEqual((root / "notice/answer.txt").read_text(), "P61\n")
            self.assertEqual(sum(e["event"] == "llm" for e in events), 2)


class ExactToolChecks(unittest.TestCase):
    def test_table_arithmetic_does_not_round(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "amounts.csv").write_text(
                "value,quantity\n10000000000000000000000000000.01,1\n0.02,2\n")
            self.assertEqual(db_query(root, "amounts.csv", "sum", "value"),
                             "10000000000000000000000000000.03")
            self.assertEqual(db_query(root, "amounts.csv", "sum_product", "value", factor="quantity"),
                             "10000000000000000000000000000.05")
            (root / "mean.csv").write_text("value\n0\n0\n1\n")
            self.assertEqual(db_query(root, "mean.csv", "avg", "value"), "1/3")
        with self.assertRaises(ValueError):
            calc("1e100000000")

    def test_sqlite_view_returns_values_but_cannot_allocate_unbounded_cell(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            conn = sqlite3.connect(root / "stocks.sqlite")
            with conn:
                conn.execute("CREATE TABLE base (quantity INTEGER)")
                conn.executemany("INSERT INTO base VALUES (?)", [(2,), (5,)])
                conn.execute("CREATE VIEW available AS SELECT quantity FROM base WHERE quantity > 2")
                conn.execute("CREATE VIEW oversized AS SELECT zeroblob(4000000) AS value")
            conn.close()
            self.assertEqual(db_query(root, "stocks.sqlite", "sum", "quantity", table="AVAILABLE"), "5")
            with self.assertRaises(sqlite3.DataError):
                db_query(root, "stocks.sqlite", "rows", table="oversized")

    @unittest.skipUnless(sys.platform == "linux", "source execution requires Linux seccomp")
    def test_replace_body_is_scoped_and_runs_original_tests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace, stage = root / "workspace", root / "stage"
            workspace.mkdir()
            stage.mkdir()
            original = ('def inc(x=2):\n    """Keep this separator \u2028 in the docstring."""\n'
                        '    return x - 1\n\ndef other():\n    return 4\n')
            tests = ("import unittest\nimport lib\nclass T(unittest.TestCase):\n"
                     "    def test_original(self):\n        self.assertEqual(lib.inc(), 3)\n"
                     "        self.assertEqual(lib.other(), 4)\n"
                     "if __name__ == '__main__':\n    unittest.main()\n")
            (workspace / "lib.py").write_text(original)
            (workspace / "test_lib.py").write_text(tests)
            coding = Coding(workspace, stage, "Run `python3 test_lib.py`.")
            self.assertFalse(coding.test()["passed"])
            self.assertTrue(coding.replace_body("lib.py", "inc", "return x + 1")["passed"])
            self.assertEqual((workspace / "lib.py").read_text(), original)
            coding.commit()
            self.assertEqual((workspace / "lib.py").read_text(), original.replace("x - 1", "x + 1"))
            self.assertEqual((workspace / "test_lib.py").read_text(), tests)


if __name__ == "__main__":
    unittest.main()
