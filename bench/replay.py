"""Exercise the real SDK, tool dispatch and HTTP accounting with scripted model replies."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Thread

from .suite import grade


def action(tool, final=False, **args):
    return {"tool": tool, "args": args, "final": final}


def response(prompt):
    if "1752 boxes" in prompt:
        return action("calc", True, expr="((1752//2*2)//3*2)//3")
    if "1669 boxes" in prompt:
        return action("calc", True, expr="1669-36*3+2*87")
    if "Irina Smith born" in prompt:
        return action("lookup", True, entity="Irina Smith", fields=["Born"])
    if "Sergei Garcia's employer" in prompt:
        feedback = prompt.partition("Recent tool results:")[2]
        if "Headquarters" in feedback:
            return {"answer": "Osaka"}
        if "Fjord Data" in feedback:
            return action("kb_facts", entity="Fjord Data")
        return action("kb_facts", entity="Sergei Garcia")
    if "product 'widget'" in prompt:
        start = 7 if "7 through 8" in prompt else 5
        return action("query", True, file="sales.csv", op="sum", column="units",
                      filters=[["product", "=", "widget"], ["month", ">=", start], ["month", "<=", start+1]])
    if "test_reverse_words" in prompt and "sorted(s.split())" in prompt:
        return action("patch", file="mylib.py", edits=[
            ["sorted(s.split())", "reversed(s.split())"], ["s -= x", "s += x"]])
    if "test_count_even" in prompt and "if x < best" in prompt:
        return action("patch", file="mylib.py", edits=[["if x < best", "if x > best"], ["x % 2 == 1", "x % 2 == 0"]])
    raise ValueError("unexpected fixture prompt")


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert self.path == "/v1/chat/completions"
            assert self.headers.get("Authorization") == "Bearer fixture-only"
            prompt = request["messages"][-1]["content"]
            assert '"expected"' not in prompt
            answer = response(prompt)
            content = json.dumps(answer)
            # Synthetic counts explicitly exercise nested reasoning and cached input.
            input_tokens = sum(len(item["content"]) for item in request["messages"]) // 4
            output_tokens = len(content) // 4 + 11
            assert output_tokens <= request["max_tokens"]
            body = json.dumps({
                "choices": [{"message": {"content": content, "reasoning_content": "fixture"},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": input_tokens, "completion_tokens": output_tokens,
                          "total_tokens": input_tokens + output_tokens,
                          "prompt_tokens_details": {"cached_tokens": 17},
                          "completion_tokens_details": {"reasoning_tokens": 11}},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (AssertionError, KeyError, ValueError) as error:
            self.send_error(400, str(error))

    def log_message(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", type=Path)
    parser.add_argument("--hybrid", action="store_true")
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    trace = args.suite / "replay.jsonl"
    if trace.exists():
        parser.error("prepare a fresh suite for replay")
    env = dict(os.environ, LLM_BASE_URL=f"http://127.0.0.1:{server.server_port}/v1",
               LLM_API_KEY="fixture-only", BUDGET="100000")
    command = [sys.executable, "run_agent.py", "--tasks-dir", str(args.suite / "tasks"),
               "--trace", str(trace), "--tool-profile", "full"]
    if not args.hybrid:
        command.append("--agent-only")
    try:
        subprocess.run(command, env=env, check=True)
    finally:
        server.shutdown()
        server.server_close()
    events = [json.loads(line) for line in trace.read_text().splitlines()]
    calls = [event for event in events if event["event"] == "llm"]
    expected_calls = 0 if args.hybrid else 10
    assert len(calls) == expected_calls, (len(calls), expected_calls)
    assert all(event["reasoning_tokens"] == 11 for event in calls)
    end = events[-1]
    assert end["spent_tenths"] == sum(e["input_tokens"] + 10 * e["output_tokens"] for e in calls)
    assert end["unknown_reserved_tenths"] == 0
    return grade(args.suite, require_all=True)


if __name__ == "__main__":
    raise SystemExit(main())
