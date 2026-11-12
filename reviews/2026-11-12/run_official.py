"""Run pristine official examples in Docker and grade outside the agent process.

Usage: python3 reviews/2026-11-12/run_official.py /path/task-dir /path/new-output
       --image ping-floyd:independent-review [--live] [--agent-only]

Credentials are inherited from LLM_API_KEY/LLM_BASE_URL, never written to logs.
The run receives no oracle. Coding is graded by ordinary Python in a separate
networkless, read-only container, not by the solver's own seccomp test worker.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def docker(network="none"):
    return ["docker", "run", "--rm", "--network", network, "--read-only",
            "--tmpfs", "/tmp:rw,size=2g", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--cpus", "4", "--memory", "16g", "--pids-limit", "512", "-e", "HOME=/tmp",
            "-e", "PYTHONDONTWRITEBYTECODE=1", "--workdir", "/app"]


def events(path):
    result = []
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            result.append(row)
    return result


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--image", default="ping-floyd:independent-review")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--agent-only", action="store_true")
    parser.add_argument("--budget", type=int, default=5000)
    parser.add_argument("--seconds", type=int, default=300)
    args = parser.parse_args()
    if args.agent_only and not args.live:
        parser.error("--agent-only needs --live")
    if args.live and not all(os.getenv(name) for name in ("LLM_API_KEY", "LLM_BASE_URL")):
        parser.error("live runs need LLM_API_KEY and LLM_BASE_URL in the environment")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    tasks, out = root / "tasks", root / "out"
    tasks.mkdir()
    out.mkdir()
    oracle, hashes = {}, {}
    for manifest in sorted(args.source.glob("*/task.json")):
        target = tasks / manifest.parent.name
        shutil.copytree(manifest.parent, target,
                        ignore=shutil.ignore_patterns("answer.txt", ".DS_Store", "__pycache__"))
        data = json.loads((target / "task.json").read_text())
        oracle[target.name] = data.pop("expected")
        (target / "task.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        for path in target.rglob("*"):
            if path.is_file():
                hashes[path.relative_to(tasks).as_posix()] = digest(path)
    if not oracle:
        parser.error("no examples with expected answers found")
    (root / "oracle.json").write_text(json.dumps(oracle, indent=2) + "\n")
    (root / "input-sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    image = json.loads(subprocess.check_output(["docker", "image", "inspect", args.image]))[0]
    image_size = subprocess.check_output(["docker", "images", args.image, "--format", "{{.Size}}"], text=True).strip()
    flags = ["--tasks-dir", "/data/tasks", "--budget", str(args.budget), "--seconds", str(args.seconds)]
    if not args.live:
        flags.append("--offline")
    elif args.agent_only:
        flags.append("--agent-only")
    credentials = ["-e", "LLM_API_KEY", "-e", "LLM_BASE_URL"] if args.live else []
    container_name = "pf-review-" + hashlib.sha256(str(root).encode()).hexdigest()[:12]
    command = [*docker("bridge" if args.live else "none"), "--name", container_name, *credentials,
               "-v", f"{tasks}:/data/tasks:rw", "-v", f"{out}:/out:rw",
               "-e", "OUT=/out/answers.txt", "--entrypoint", "python", image["Id"],
               "/app/run_agent.py", *flags]
    started = time.monotonic()
    with (root / "run.jsonl").open("w") as log:
        try:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.seconds + 90)
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "stop", "--time", "5", container_name],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
            raise
    elapsed = time.monotonic() - started
    grade = []
    for name, expected in oracle.items():
        folder = tasks / name
        manifest = json.loads((folder / "task.json").read_text())
        answer = folder / "answer.txt"
        actual = answer.read_text().strip() if answer.is_file() else None
        if manifest["check"] == "unittest":
            # These official examples explicitly invoke test_mylib.py.
            cmd = [*docker(), "-v", f"{folder / 'workspace'}:/case:ro", "--workdir", "/case",
                   "--entrypoint", "python", image["Id"], "-B", "test_mylib.py"]
            check = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
            (root / f"judge-{name}.log").write_text(check.stdout + check.stderr)
            correct = actual == "DONE" and check.returncode == 0
        elif manifest["check"] == "exact_int":
            try:
                correct = int(actual) == int(expected)
            except (ValueError, TypeError):
                correct = False
        else:
            correct = actual == str(expected)
        grade.append({"task": name, "correct": correct, "actual": actual,
                      "expected": expected, "reward": manifest["reward"]})
    changed = [name for name, before in hashes.items() if digest(tasks / name) != before]
    protected_changes = [name for name in changed if Path(name).name == "task.json" or
                         Path(name).name.startswith("test") or Path(name).suffix != ".py"]
    log = events(root / "run.jsonl")
    llm = [row for row in log if row.get("event") == "llm"]
    end = next((row for row in reversed(log) if row.get("event") == "run_end"), {})
    snapshot = "".join(f"{name}\t{(tasks / name / 'answer.txt').read_text().strip()}\n"
                       for name in sorted(oracle) if (tasks / name / "answer.txt").exists())
    known = sum(row.get("cost_tenths", 0) for row in llm)
    unknown = sum(row.get("unknown_reserved_tenths", 0) for row in llm)
    summary = {"image_id": image["Id"], "image_inspect_size_bytes": image["Size"],
               "contest_image_size_display": image_size, "architecture": image["Architecture"],
               "live": args.live, "agent_only": args.agent_only, "exit_code": result.returncode,
               "elapsed_s": round(elapsed, 2), "budget": args.budget,
               "correct": sum(item["correct"] for item in grade), "tasks": len(grade),
               "score": sum(item["reward"] for item in grade if item["correct"]),
               "max_score": sum(item["reward"] for item in grade), "llm_calls": len(llm),
               "input_tokens": sum(row.get("input_tokens", 0) for row in llm),
               "output_tokens": sum(row.get("output_tokens", 0) for row in llm),
               "weighted_tokens": known / 10, "unknown_reserved_tokens": unknown / 10,
               "ledger_matches": end.get("spent_tenths") == known and end.get("unknown_reserved_tenths") == unknown,
               "within_budget": known + unknown <= 10 * args.budget,
               "out_matches": (out / "answers.txt").exists() and (out / "answers.txt").read_text() == snapshot,
               "protected_changes": protected_changes, "modified_files": changed,
               "run_end": end, "grade": grade}
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    success = (result.returncode == 0 and summary["correct"] == summary["tasks"] and
               summary["ledger_matches"] and summary["within_budget"] and summary["out_matches"] and not protected_changes)
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
