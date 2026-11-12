"""Frozen family-separated qualification and one-factor live comparisons."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import shutil
from statistics import median

from ping_floyd.domain import TYPES
from ping_floyd.planning import PRIORS

from .experiments import cell, jsonl
from .large import dump

MODEL = "qwen3.8-27b-noreason"
RETRY = "qwen3.6-35b-a3b-noreason"
VARIANTS = {
    "universal": (False, "compact", ()),
    "hybrid": (True, "compact", ()),
    "full_context": (True, "full", ()),
    "reward_queue": (True, "compact", ("--queue", "reward")),
    "fixed_cost": (True, "compact", ("--fixed-estimates",)),
    "parallel": (True, "compact", ("--workers", "3")),
    "second_model": (True, "compact", ("--retry-model", RETRY)),
    "calibrated": (True, "compact", ("--calibration", "/calibration.json")),
}


def freeze(source, destination, prior_suite, seed, development):
    cases = json.loads((source / "metadata.json").read_text())["cases"]
    oracle = json.loads((source / "oracle.json").read_text())
    exposed = set()
    if prior_suite:
        previous = json.loads((prior_suite / "metadata.json").read_text())["cases"]
        exposed = {(entry["kind"], entry["family"]) for entry in previous.values()}
    rng = random.Random(seed)
    selection = []
    for kind in sorted(TYPES):
        groups = (("dev", False, 2), ("dev", True, 1)) if development else (
            ("holdout", False, 1), ("hard", False, 3), ("hard", True, 1))
        for split, impossible, count in groups:
            families = defaultdict(list)
            for name, entry in sorted(cases.items()):
                family = entry["family"]
                if (entry["kind"] == kind and entry["split"] == split and (oracle[name] is None) == impossible
                        and (kind, family) not in exposed):
                    families[family].append(name)
            for family in rng.sample(sorted(families), count):
                selection.append(rng.choice(families[family]))
    destination.mkdir(parents=True, exist_ok=False)
    for name in sorted(selection):
        task = source / "tasks" / name
        if (task / "answer.txt").exists():
            raise ValueError("qualification requires a pristine suite")
        shutil.copytree(task, destination / "tasks" / name)
        judge = source / "judge" / name
        if judge.is_dir():
            shutil.copytree(judge, destination / "judge" / name)
    dump(destination / "oracle.json", {name: oracle[name] for name in selection})
    dump(destination / "metadata.json", {"seed": seed, "cases": {name: cases[name] for name in selection}})
    print(json.dumps({"tasks": len(selection), "unsolvable": sum(oracle[name] is None for name in selection)}))


def calibrate(suite, traces, output):
    cases = json.loads((suite / "metadata.json").read_text())["cases"]
    if any(entry["split"] != "dev" for entry in cases.values()):
        raise ValueError("calibration accepts development tasks only")
    samples = defaultdict(list)
    for folder in traces:
        grades = {e["grade"]: e["outcome"] for e in jsonl(folder / "grade.jsonl") if "grade" in e}
        events = jsonl(folder / "run.jsonl")
        failed = {e["task"] for e in events if e.get("event") == "llm" and "cost_tenths" not in e}
        for event in events:
            name = event.get("task")
            if (event.get("event") == "task" and event.get("phase") == "model" and event.get("cost_tenths", 0)
                    and name in cases and name in grades and name not in failed):
                samples[cases[name]["kind"]].append((event["cost_tenths"] / 10, grades[name] == "correct_answers"))
    kinds = {}
    for kind, entries in sorted(samples.items()):
        count = len(entries)
        costs = median(cost for cost, _ in entries)
        correct = sum(passed for _, passed in entries)
        kinds[kind] = {"weighted_tokens": round((PRIORS[kind] * 2 + costs * count) / (count + 2), 2),
                       "success_rate": round((3 + correct) / (4 + count), 4), "samples": count}
    if not kinds:
        raise ValueError("no billed, graded development attempts")
    dump(output, {"version": 1, "kinds": kinds})


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for label in args.variant:
        args.hybrid, profile, extra = VARIANTS[label]
        if label == "calibrated" and not args.calibration:
            raise ValueError("calibrated variant requires --calibration")
        result = cell(args, MODEL, profile, label=label, extra=extra)
        result.update(variant=label, options=list(extra))
        rows.append(result)
        dump(args.output / "matrix.json", {"budget_per_cell": args.budget, "suite": str(args.suite.resolve()),
                                         "transport": "live", "cells": rows})
        print(json.dumps({"variant": label, "score": result["metrics"].get("score"),
                          "calls": result["calls"], "weighted_tokens": result["weighted_tokens"],
                          "http_errors": result["http_errors"], "agent_exit": result["agent_exit"]}), flush=True)
        if (result["agent_exit"] or result["grader_exit"] or not result["metrics"]
                or set(result["http_errors"]) & {401, 402, 403, 404, 429}):
            raise SystemExit("matrix stopped after a failed cell; see the run and grade logs")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("source", type=Path)
    prep.add_argument("destination", type=Path)
    prep.add_argument("--exclude-families", type=Path)
    prep.add_argument("--development", action="store_true")
    prep.add_argument("--seed", type=int, default=20261112)
    cal = sub.add_parser("calibrate")
    cal.add_argument("suite", type=Path)
    cal.add_argument("output", type=Path)
    cal.add_argument("traces", nargs="+", type=Path)
    matrix = sub.add_parser("run")
    matrix.add_argument("suite", type=Path)
    matrix.add_argument("output", type=Path)
    matrix.add_argument("--env-file", required=True, type=Path)
    matrix.add_argument("--variant", choices=VARIANTS, action="append", required=True)
    matrix.add_argument("--calibration", type=Path)
    matrix.add_argument("--image", default="ping-floyd:release")
    matrix.add_argument("--bench-image", default="ping-floyd:release-bench")
    matrix.add_argument("--budget", type=int, default=1400)
    matrix.add_argument("--seconds", type=int, default=180)
    args = parser.parse_args()
    if args.command == "prepare":
        freeze(args.source, args.destination, args.exclude_families, args.seed, args.development)
    elif args.command == "calibrate":
        calibrate(args.suite, args.traces, args.output)
    else:
        if min(args.budget, args.seconds) <= 0 or not args.env_file.is_file() or len(set(args.variant)) != len(args.variant):
            parser.error("use positive limits, an existing env-file and unique variants")
        run(args)


if __name__ == "__main__":
    main()
