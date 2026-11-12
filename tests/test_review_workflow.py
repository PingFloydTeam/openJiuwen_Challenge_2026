"""End-to-end regression cases with real openJiuwen and a scripted HTTP transport.

The transport controls model variability only. Scheduling, tool validation,
workflows, answer files, and budget accounting remain the production code.
"""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
import run_agent
from ping_floyd.proxy import Proxy, token_usage


def task(root, name, kind, instruction, assets=None, reward=5, check="exact_str"):
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "task.json").write_text(json.dumps({
        "task_id": name, "task_type": kind, "instruction": instruction,
        "reward": reward, "check": check,
    }))
    for name, content in (assets or {}).items():
        path = directory / "workspace" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return directory


def reply(action, usage=None):
    return httpx.Response(200, json={
        "choices": [{"message": {"content": json.dumps(action)}, "finish_reason": "stop"}],
        "usage": usage or {"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140},
    })


class IndependentWorkflowContracts(unittest.IsolatedAsyncioTestCase):
    async def execute(self, root, actions, extra=()):
        original_client = httpx.AsyncClient
        requests = []
        queue = iter(actions)

        def handler(request):
            requests.append(json.loads(request.content))
            action = next(queue, {"skip": "scripted responses exhausted"})
            return action if isinstance(action, httpx.Response) else reply(action)

        def factory(*args):
            with patch("ping_floyd.proxy.httpx.AsyncClient", lambda **kwargs:
                       original_client(transport=httpx.MockTransport(handler), **kwargs)):
                return Proxy(*args)

        output = io.StringIO()
        with patch.dict(os.environ, {"LLM_API_KEY": "review-placeholder", "LLM_BASE_URL": "https://offline.invalid/v1",
                                     "LLM_MODEL": "qwen3.8-27b-noreason", "LLM_TOOL_PROFILE": "compact"}), \
                patch("sys.argv", ["run_agent.py", "--tasks-dir", str(root), "--budget", "10000",
                                   "--seconds", "30", *extra]), \
                patch("run_agent.Proxy", side_effect=factory), redirect_stdout(output):
            await run_agent.main(run_agent.options())
        events = [json.loads(line) for line in output.getvalue().splitlines() if line.startswith("{")]
        return requests, events

    async def test_control_real_workflow_writes_answer_and_reconciles_usage(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            directory = task(root, "math", "math", "Return the product of six and seven.", check="exact_int")
            _, events = await self.execute(root, [{"tool": "calc", "args": {"expr": "6*7"}, "final": True}])
            self.assertEqual((directory / "answer.txt").read_text().strip(), "42")
            self.assertEqual(events[-1]["spent_tenths"], 500)

    async def test_field_alias_reaches_model_fallback(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            directory = task(root, "search", "search", "According to kb/, In which year was Ada Reed born?",
                             {"kb/ada.md": "# Ada Reed\nBirth year: 1983\n"})
            _, events = await self.execute(root, [
                {"tool": "lookup", "args": {"entity": "Ada Reed", "fields": ["Birth year"]}, "final": True}])
            self.assertTrue((directory / "answer.txt").exists(), str(events))
            self.assertEqual((directory / "answer.txt").read_text().strip(), "1983")

    async def test_truncation_gets_a_bounded_retry_when_budget_is_available(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            directory = task(root, "math", "math", "Return the product of six and seven.", check="exact_int")
            truncated = httpx.Response(200, json={
                "choices": [{"message": {"content": '{"tool":"calc",'}, "finish_reason": "length"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 450, "total_tokens": 550},
            })
            _, events = await self.execute(root, [truncated,
                {"tool": "calc", "args": {"expr": "6*7"}, "final": True}])
            self.assertTrue((directory / "answer.txt").exists(), str(events))
            self.assertEqual((directory / "answer.txt").read_text().strip(), "42")
            self.assertEqual(events[-1]["spent_tenths"], 5100)

    async def test_retry_after_rate_limit_does_not_disable_the_entire_run(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            directory = task(root, "math", "math", "Return the product of six and seven.", check="exact_int")
            limited = httpx.Response(429, headers={"Retry-After": "0"}, json={"error": "temporary rate limit"})
            requests, events = await self.execute(root, [limited,
                {"tool": "calc", "args": {"expr": "6*7"}, "final": True}])
            self.assertTrue((directory / "answer.txt").exists(), str(events))
            self.assertEqual((directory / "answer.txt").read_text().strip(), "42")
            self.assertLessEqual(len(requests), 3)
            end = events[-1]
            self.assertLessEqual(end["spent_tenths"] + end["unknown_reserved_tenths"], 100000)

    async def test_narrative_birth_year_is_valid_read_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            directory = task(root, "search", "search", "Read biography.txt. In which year was Ada Reed born?",
                             {"biography.txt": "Ada Reed was born in 1983 in Oslo.\n"})
            _, events = await self.execute(root, [
                {"tool": "read", "args": {"file": "biography.txt"}}, {"answer": "1983"}])
            self.assertTrue((directory / "answer.txt").exists(), str(events))
            self.assertEqual((directory / "answer.txt").read_text().strip(), "1983")

    async def test_complete_bounded_proof_is_not_rejected_by_parity_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            directory = task(root, "math", "math",
                             "A machine starts with an odd integer x between 0 and 10 inclusive. "
                             "Add 1, divide by 3 and round down. The result is 0. Return x.", check="exact_int")
            action = {"tool": "finite_search", "args": {"domains": {"x": [0, 10]},
                      "constraints": ["x%2=1", "(x+1)//3=0"], "target": "x"}, "final": True}
            _, events = await self.execute(root, [action, {"answer": "1"}])
            self.assertTrue((directory / "answer.txt").exists(), str(events))
            self.assertEqual((directory / "answer.txt").read_text().strip(), "1")

    async def test_search_target_can_follow_more_than_two_links(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            directory = task(root, "search", "search",
                             "Using kb/, follow the mentor of Ada Reed, then that person's mentor, "
                             "then their mentor. Which company employs the final person?", {
                                 "kb/a.md": "# Ada Reed\nMentor: Ben Reed\n",
                                 "kb/b.md": "# Ben Reed\nMentor: Cara Reed\n",
                                 "kb/c.md": "# Cara Reed\nMentor: Dan Reed\n",
                                 "kb/d.md": "# Dan Reed\nEmployer: Acme\n",
                             })
            _, events = await self.execute(root, [
                {"tool": "lookup", "args": {"entity": "Ada Reed", "fields": ["Mentor", "Mentor", "Mentor"]}},
                {"tool": "lookup", "args": {"entity": "Dan Reed", "fields": ["Employer"]}, "final": True},
            ])
            self.assertTrue((directory / "answer.txt").exists(), str(events))
            self.assertEqual((directory / "answer.txt").read_text().strip(), "Acme")


class IndependentTransportContracts(unittest.IsolatedAsyncioTestCase):
    def test_unexplained_total_tokens_are_not_silently_underbilled(self):
        usage = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 220}
        try:
            prompt, completion, _ = token_usage(usage)
        except ValueError:
            return  # Conservatively reserving unknown usage is also valid.
        self.assertGreaterEqual(prompt + 10 * completion, 1300,
                                "100 additional tokens must be accounted for or the usage rejected")

    def test_control_reasoning_accounted_once_in_both_usage_shapes(self):
        inclusive = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150,
                     "completion_tokens_details": {"reasoning_tokens": 30}}
        separate = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 150,
                    "reasoning_tokens": 30}
        self.assertEqual(token_usage(inclusive)[:2], (100, 50))
        self.assertEqual(token_usage(separate)[:2], (100, 50))

if __name__ == "__main__":
    unittest.main()
