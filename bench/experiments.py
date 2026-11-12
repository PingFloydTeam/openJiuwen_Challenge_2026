"""Run reproducible model/tool comparisons on copies of a frozen benchmark slice."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import shutil
import subprocess
import time

from .large import dump

SELECTION = {
    "irrelevant_speed": 2, "paraphrase": 2, "rotate_left": 2, "unique_in_order": 2,
    "unspecified_rule": 2, "sqlite": 1, "revenue": 1, "empty_average": 1, "conflicting_rows": 1,
}
MODELS = ("qwen3.6-35b-a3b-noreason", "qwen3.8-27b-noreason", "qwen3.6-35b-a3b", "qwen3.8-27b")
PROFILES = ("full", "compact", "body")


def jsonl(path):
    events = []
    if path.is_file():
        for line in path.read_text().splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
    return events


def prepare(source, trace, destination, seed):
    metadata = json.loads((source / "metadata.json").read_text())["cases"]
    oracle = json.loads((source / "oracle.json").read_text())
    deferred = {e["task"] for e in jsonl(trace) if e.get("route") == "offline_unresolved"}
    rng = random.Random(seed)
    selection = [name for family, count in SELECTION.items()
                 for name in rng.sample(sorted(n for n, e in metadata.items() if e["family"] == family), count)]
    selection.extend(sorted(n for n in deferred if metadata[n]["family"] == "missing"))
    if len(selection) != 16 or len(set(selection)) != len(selection):
        raise ValueError("expected 16 distinct tasks including both residual missing-data cases")
    destination.mkdir(parents=True, exist_ok=False)
    for name in sorted(selection):
        task = source / "tasks" / name
        if (task / "answer.txt").exists():
            raise ValueError("select from a pristine suite")
        shutil.copytree(task, destination / "tasks" / name)
        judge = source / "judge" / name
        if judge.is_dir():
            shutil.copytree(judge, destination / "judge" / name)
    dump(destination / "oracle.json", {name: oracle[name] for name in selection})
    dump(destination / "metadata.json", {"seed": seed, "cases": {n: metadata[n] for n in selection}})
    print(json.dumps({"tasks": len(selection), "solvable": sum(oracle[n] is not None for n in selection),
                      "previously_deferred": sum(n in deferred for n in selection)}), flush=True)


def docker_prefix(network):
    return ["docker", "run", "--rm", "--network", network, "--read-only", "--tmpfs", "/tmp:rw,size=2g",
            "-e", "HOME=/tmp", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--cpus", "4", "--memory", "16g", "--pids-limit", "512"]


def cell(args, model, profile, *, label=None, extra=()):
    out = args.output / (label or f"{model}__{profile}")
    out.mkdir()
    suite = out / "suite"
    shutil.copytree(args.suite, suite)
    mounts = ["-v", f"{args.calibration.resolve()}:/calibration.json:ro"] if "--calibration" in extra else []
    cmd = [*docker_prefix("bridge"), "--env-file", str(args.env_file), *mounts,
           "-v", f"{suite.resolve() / 'tasks'}:/data/tasks:rw", "--entrypoint", "python", args.image,
           "/app/run_agent.py", "--tasks-dir", "/data/tasks", "--model", model,
           "--tool-profile", profile, "--budget", str(args.budget), "--seconds", str(args.seconds), *extra]
    if not args.hybrid:
        cmd.append("--agent-only")
    started = time.monotonic()
    with (out / "run.jsonl").open("w") as stream:
        proc = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, check=False)
    with (out / "grade.jsonl").open("w") as stream:
        grade = subprocess.run([*docker_prefix("none"), "-v", f"{suite.resolve()}:/suite:ro",
                                "--entrypoint", "python", args.bench_image, "-m", "bench.suite", "grade", "/suite"],
                               stdout=stream, stderr=subprocess.STDOUT, check=False)
    events = jsonl(out / "run.jsonl")
    calls = [e for e in events if e.get("event") == "llm"]
    totals = next((e for e in reversed(events) if e.get("event") == "run_end"), {})
    metrics = next((e for e in reversed(jsonl(out / "grade.jsonl")) if "score" in e), {})
    return {"model": model, "profile": profile, "elapsed_s": round(time.monotonic()-started, 2),
            "agent_exit": proc.returncode, "grader_exit": grade.returncode, "calls": len(calls),
            "billed_calls": sum("cost_tenths" in e for e in calls),
            "http_errors": dict(Counter(e["status"] for e in calls if "status" in e)),
            "weighted_tokens": totals["spent_tenths"] / 10 if "spent_tenths" in totals else None,
            "unknown_reserved_tenths": totals.get("unknown_reserved_tenths"),
            "input_tokens": sum(e.get("input_tokens", 0) for e in calls),
            "output_tokens": sum(e.get("output_tokens", 0) for e in calls),
            "reported_reasoning_tokens": sum(e.get("reasoning_tokens") or 0 for e in calls),
            "attempted": totals.get("model_attempted"), "metrics": metrics, "path": str(out.resolve())}


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    cells = []

    def run_cell(model, profile):
        result = cell(args, model, profile)
        cells.append(result)
        dump(args.output / "matrix.json", {"budget_per_cell": args.budget,
                                         "mode": "hybrid" if args.hybrid else "agent_only",
                                         "suite": str(args.suite.resolve()), "cells": cells})
        print(json.dumps({k: result[k] for k in (
            "model", "profile", "calls", "billed_calls", "weighted_tokens", "http_errors"
        )} | {k: result["metrics"].get(k) for k in ("score", "max_score", "correct_answers", "false_accepts")}),
              flush=True)
        return result

    if args.variant:
        for variant in args.variant:
            run_cell(*variant)
        return
    for model in MODELS:
        run_cell(model, "compact")
    available = [r for r in cells if r["billed_calls"] and r["metrics"] and
                 not r["agent_exit"] and not r["grader_exit"] and not r["http_errors"]]
    if available:
        best = min(available, key=lambda r: (
            -r["metrics"]["score"], r["metrics"]["false_accepts"], r["weighted_tokens"]
        ))
        for profile in ("full", "body"):
            run_cell(best["model"], profile)


def parse_variant(text):
    model, _, profile = text.partition(":")
    if model not in MODELS or profile not in PROFILES:
        raise argparse.ArgumentTypeError("expected an allowed MODEL:PROFILE")
    return model, profile


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    selection = sub.add_parser("prepare")
    selection.add_argument("source", type=Path)
    selection.add_argument("trace", type=Path)
    selection.add_argument("destination", type=Path)
    selection.add_argument("--seed", type=int, default=1109)
    matrix = sub.add_parser("run")
    matrix.add_argument("suite", type=Path)
    matrix.add_argument("output", type=Path)
    matrix.add_argument("--env-file", required=True, type=Path)
    matrix.add_argument("--image", default="ping-floyd:experiment")
    matrix.add_argument("--bench-image", default="ping-floyd:experiment-bench")
    matrix.add_argument("--budget", type=int, default=5000)
    matrix.add_argument("--seconds", type=int, default=120)
    matrix.add_argument("--hybrid", action="store_true", help="run the deterministic pass before the model")
    matrix.add_argument("--variant", action="append", type=parse_variant,
                        help="run only these MODEL:PROFILE pairs; can be repeated")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.source, args.trace, args.destination, args.seed)
    else:
        if args.budget <= 0 or args.seconds <= 0 or not args.env_file.is_file():
            parser.error("use positive limits and an existing env-file")
        if args.variant and len(set(args.variant)) != len(args.variant):
            parser.error("use each model/profile pair only once")
        run(args)


if __name__ == "__main__":
    main()
