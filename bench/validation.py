"""Reproduce one isolated run, grade it separately, and verify persisted outputs."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import time

from .experiments import docker_prefix, jsonl


def run(args):
    if any(args.suite.glob("tasks/*/answer.txt")):
        raise ValueError("validation requires pristine tasks")
    root = args.output.resolve()
    image_info = subprocess.run(["docker", "image", "inspect", args.image, "--format", "{{.Id}} {{.Size}}"],
                                capture_output=True, text=True, check=True).stdout.strip().split()
    bench_id = subprocess.run(["docker", "image", "inspect", args.bench_image, "--format", "{{.Id}}"],
                             capture_output=True, text=True, check=True).stdout.strip()
    root.mkdir(parents=True, exist_ok=False)
    # Keep both immutable IDs reachable while later builds move the user's tags.
    # Otherwise a long agent run can finish after its untagged judge was collected.
    run_key = sha256(str(root).encode()).hexdigest()[:16]
    pinned = {"agent": f"ping-floyd-validation:{run_key}-agent", "judge": f"ping-floyd-validation:{run_key}-judge"}
    for role, identifier in (("agent", image_info[0]), ("judge", bench_id)):
        subprocess.run(["docker", "tag", identifier, pinned[role]], check=True)
    shutil.copytree(args.suite, root / "suite")
    (root / "out").mkdir()
    flags = ["--workers", "1", "--seconds", str(args.seconds), "--model", args.model]
    if not args.live:
        flags.append("--offline")
    env = ["--env-file", str(args.env_file.resolve())] if args.env_file else ["-e", "LLM_API_KEY", "-e", "LLM_BASE_URL"]
    cmd = [*docker_prefix("bridge" if args.live else "none"), *(env if args.live else []),
           "-v", f"{root}/suite/tasks:/tasks:rw", "-v", f"{root}/out:/out:rw",
           "-e", "TASKS=/tasks", "-e", "OUT=/out/answers.txt", "-e", f"BUDGET={args.budget}",
           "--entrypoint", "sh", image_info[0], "-c",
           'python /app/run_agent.py "$@"; result=$?; cat /sys/fs/cgroup/memory.peak > /out/memory-bytes.txt; exit "$result"',
           "validation", *flags]
    started = time.monotonic()
    with (root / "run.jsonl").open("w") as log:
        agent = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.monotonic() - started
    with (root / "grade.jsonl").open("w") as log:
        judge = subprocess.run([*docker_prefix("none"), "-v", f"{root}/suite:/suite:ro",
                                "--entrypoint", "python", bench_id, "-m", "bench.suite", "grade", "/suite"],
                               stdout=log, stderr=subprocess.STDOUT)
    result = summarize(root, args.suite)
    result.update(agent_exit=agent.returncode, judge_exit=judge.returncode, elapsed_s=round(elapsed, 2),
                  image=args.image, model=args.model if args.live else None, budget=args.budget,
                  live=args.live)
    result.update(image_id=image_info[0], image_bytes=int(image_info[1]), bench_image_id=bench_id)
    result["pinned_images"] = pinned
    (root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    if agent.returncode or judge.returncode or not result["run_end"]:
        raise SystemExit("run failed; inspect saved logs")


def summarize(root, pristine):
    events = jsonl(root / "run.jsonl")
    calls = [e for e in events if e.get("event") == "llm"]
    end = next((e for e in reversed(events) if e.get("event") == "run_end"), {})
    grade = next((e for e in reversed(jsonl(root / "grade.jsonl")) if "score" in e), {})
    for path in (pristine / "tasks").rglob("*"):
        if path.is_file() and (path.name == "task.json" or path.name.startswith("test") or path.name.endswith("_test.py")):
            counterpart = root / "suite/tasks" / path.relative_to(pristine / "tasks")
            if path.read_bytes() != counterpart.read_bytes():
                raise ValueError(f"protected input changed: {counterpart}")
    snapshot = (root / "out/answers.txt").read_text()
    saved = []
    escape = lambda value: value.translate(str.maketrans({"\\": "\\\\", "\t": "\\t", "\n": "\\n", "\r": "\\r"}))
    for answer in (root / "suite/tasks").glob("*/answer.txt"):
        task = json.loads((answer.parent / "task.json").read_text())
        saved.append((task.get("task_id") or answer.parent.name, answer.read_text().removesuffix("\n")))
    expected = "".join(f"{escape(name)}\t{escape(answer)}\n" for name, answer in sorted(saved))
    if snapshot != expected:
        raise ValueError("OUT disagrees with task answers")
    if end and end["spent_tenths"] != sum(e.get("cost_tenths", 0) for e in calls):
        raise ValueError("token ledger does not reconcile")
    if end and end["unknown_reserved_tenths"] != sum(e.get("unknown_reserved_tenths", 0) for e in calls):
        raise ValueError("unknown usage reservations do not reconcile")
    return {"grade": grade, "run_end": end, "calls": len(calls),
            "weighted_tokens": sum(e.get("cost_tenths", 0) for e in calls) / 10,
            "input_tokens": sum(e.get("input_tokens", 0) for e in calls),
            "output_tokens": sum(e.get("output_tokens", 0) for e in calls),
            "unknown_reserved_tokens": sum(e.get("unknown_reserved_tenths", 0) for e in calls) / 10,
            "memory_peak_bytes": int((root / "out/memory-bytes.txt").read_text()),
            "out_records": len(saved), "out_matches_answers": True, "protected_inputs_unchanged": True,
            "oracle_sha256": sha256((root / "suite/oracle.json").read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--image", default="ping-floyd:requirements")
    parser.add_argument("--bench-image", default="ping-floyd:requirements-bench")
    parser.add_argument("--model", default="qwen3.8-27b-noreason")
    parser.add_argument("--budget", type=int, default=100000)
    parser.add_argument("--seconds", type=int, default=900)
    args = parser.parse_args()
    if min(args.budget, args.seconds) <= 0:
        parser.error("limits must be positive")
    run(args)


if __name__ == "__main__":
    main()
