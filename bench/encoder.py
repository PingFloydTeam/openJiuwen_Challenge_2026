"""Compare text representations on completed, billed fallback trajectories."""
import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import mean
import time

from ping_floyd.domain import Task
from ping_floyd.planning import Planner

REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
ASSETS = {
    "tokenizer.json": "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037",
    "onnx/model_qint8_arm64.onnx": "4278337fd0ff3c68bfb6291042cad8ab363e1d9fbc43dcb499fe91c871902474",
}
MODEL = "qwen3.8-27b-noreason"
BUDGETS = (500, 1000, 2500, 5000, 10000)


@dataclass(frozen=True)
class Sample:
    task: Task
    family: str
    cost: int
    correct: bool
    points: int


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.startswith("{")]


def load(suite, trace):
    cases = json.loads((suite / "metadata.json").read_text())["cases"]
    events = jsonl(trace / "run.jsonl")
    if events[0]["model"] != MODEL:
        raise ValueError("select traces for the same model")
    grades = {row["grade"]: row for row in jsonl(trace / "grade.jsonl") if "grade" in row}
    unknown = {row["task"] for row in events if row["event"] == "llm" and "cost_tenths" not in row}
    samples = []
    for row in events:
        if row["event"] != "task" or row["phase"] != "model" or not row["cost_tenths"]:
            continue
        key = row["task"]
        if key in unknown:
            continue
        task = Task.load(suite / "tasks" / key / "task.json")
        grade = grades[key]
        samples.append(Sample(task, cases[key]["family"], row["cost_tenths"],
                              grade["outcome"] == "correct_answers", grade["reward"]))
    return samples


def char_vector(text):
    text = " " + " ".join(text.casefold().split()) + " "
    counts = Counter(text[start:start + length] for length in (3, 4)
                     for start in range(len(text) - length + 1))
    norm = math.sqrt(sum(count * count for count in counts.values())) or 1
    return {gram: count / norm for gram, count in counts.items()}


def char_similarity(a, b):
    return sum(weight * b.get(gram, 0) for gram, weight in a.items())


def encode(texts, assets):
    import numpy as np
    import onnxruntime as ort
    from tokenizers import Tokenizer

    for name, expected in ASSETS.items():
        if hashlib.sha256((assets / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"unexpected encoder asset: {name}")
    tokenizer = Tokenizer.from_file(str(assets / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=256)
    tokenizer.enable_padding()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(assets / "onnx/model_qint8_arm64.onnx"), options,
                                   providers=["CPUExecutionProvider"])
    rows = []
    for start in range(0, len(texts), 16):
        tokens = tokenizer.encode_batch(texts[start:start + 16])
        batch = {"input_ids": [t.ids for t in tokens], "attention_mask": [t.attention_mask for t in tokens],
                 "token_type_ids": [t.type_ids for t in tokens]}
        inputs = {item.name: np.asarray(batch[item.name], dtype=np.int64) for item in session.get_inputs()}
        hidden = session.run(None, inputs)[0]
        mask = inputs["attention_mask"][..., None]
        pooled = (hidden * mask).sum(axis=1) / mask.sum(axis=1)
        rows.extend(pooled / np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12))
    return dict(zip(texts, rows))


def fit_estimates(training, testing, similarity=None):
    # Strong shrinkage toward the task type because the trace bank is small.
    global_cost = mean(row.cost for row in training)
    global_success = mean(row.correct for row in training)
    by_kind = {}
    for row in training:
        by_kind.setdefault(row.task.kind, []).append(row)
    estimates = []
    for row in testing:
        matches = by_kind.get(row.task.kind, [])
        cost = (2 * global_cost + sum(item.cost for item in matches)) / (2 + len(matches))
        success = (4 * global_success + sum(item.correct for item in matches)) / (4 + len(matches))
        if similarity and matches:
            ranked = sorted(((similarity(row.task.instruction, item.task.instruction), item) for item in matches),
                            key=lambda pair: pair[0], reverse=True)[:5]
            weights = [max(0, score)**2 for score, _ in ranked]
            normalizer = 2 + sum(weights)
            cost = (2 * cost + sum(w * item.cost for w, (_, item) in zip(weights, ranked))) / normalizer
            success = (2 * success + sum(w * item.correct for w, (_, item) in zip(weights, ranked))) / normalizer
        estimates.append((cost, success))
    return estimates


def evaluate(rows, estimates):
    queue = sorted(zip(rows, estimates), key=lambda pair: (
        -pair[0].task.reward * pair[1][1] / pair[1][0], -pair[0].task.reward, pair[0].task.root.name))
    replay = {}
    for budget in BUDGETS:
        spent = points = attempted = 0
        for row, _ in queue:
            if spent + row.cost > 10 * budget:
                break
            spent += row.cost
            points += row.points
            attempted += 1
        replay[budget] = {"points": points, "spent": spent / 10, "attempted": attempted}
    return {
        "mae_weighted_tokens": round(mean(abs(cost - row.cost) for row, (cost, _) in zip(rows, estimates)) / 10, 2),
        "brier": round(mean((probability - row.correct)**2
                            for row, (_, probability) in zip(rows, estimates)), 4),
        "predicted_total_weighted_tokens": round(sum(cost for cost, _ in estimates) / 10, 1),
        "replay": replay,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="append", nargs=2, type=Path, required=True, metavar=("SUITE", "TRACE"))
    parser.add_argument("--test", action="append", nargs=3, required=True, metavar=("NAME", "SUITE", "TRACE"))
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tests = {name: load(Path(suite), Path(trace)) for name, suite, trace in args.test}
    test_families = {row.family for rows in tests.values() for row in rows}
    test_texts = {row.task.instruction for rows in tests.values() for row in rows}
    training = [row for suite, trace in args.train for row in load(suite, trace)
                if row.family not in test_families and row.task.instruction not in test_texts]
    texts = list(dict.fromkeys([row.task.instruction for row in training] +
                              [row.task.instruction for rows in tests.values() for row in rows]))
    started = time.perf_counter()
    vectors = {text: char_vector(text) for text in texts}
    char_time = time.perf_counter() - started
    started = time.perf_counter()
    encoded = encode(texts, args.assets)
    encoder_time = time.perf_counter() - started
    current = Planner(32000, adaptive=False)
    output = {
        "trace_model": MODEL,
        "encoder_revision": REVISION,
        "asset_checksums": ASSETS,
        "train_rows": len(training),
        "train_families": len({row.family for row in training}),
        "train_by_kind": dict(Counter(row.task.kind for row in training)),
        "sources": {"train": [[str(path) for path in pair] for pair in args.train], "test": args.test},
        "replay_method": "Frozen order, known-cost prefix of model attempts; excludes local points and unknown usage.",
        "encoding_wall_seconds": {"rows": len(texts), "char_ngrams": round(char_time, 3),
                                  "minilm_int8": round(encoder_time, 3)},
        "tests": {},
    }
    for name, rows in tests.items():
        strategies = {
            "current_static": [(10 * current.estimate(row.task), 0.75) for row in rows],
            "calibrated_kind": fit_estimates(training, rows),
            "char_ngrams": fit_estimates(training, rows, lambda a, b: char_similarity(vectors[a], vectors[b])),
            "minilm_int8": fit_estimates(training, rows, lambda a, b: float(encoded[a] @ encoded[b])),
        }
        output["tests"][name] = {
            "tasks": len(rows),
            "families": len({row.family for row in rows}),
            "actual_cost": sum(row.cost for row in rows) / 10,
            "actual_points": sum(row.points for row in rows),
            "strategies": {policy: evaluate(rows, estimates) for policy, estimates in strategies.items()},
        }
    for name, _, trace in args.test:
        logged = {event["task"]: event["estimated_tenths"] for event in jsonl(Path(trace) / "run.jsonl")
                  if event["event"] == "fallback_start"}
        rows = [row for row in tests[name] if row.task.root.name in logged]
        if rows:
            output["tests"][name]["logged_online_cost"] = {
                "tasks": len(rows),
                "predicted_total_weighted_tokens": sum(logged[row.task.root.name] for row in rows) / 10,
                "actual_total_weighted_tokens": sum(row.cost for row in rows) / 10,
                "mae_weighted_tokens": round(mean(abs(logged[row.task.root.name] - row.cost)
                                                 for row in rows) / 10, 2),
            }
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
