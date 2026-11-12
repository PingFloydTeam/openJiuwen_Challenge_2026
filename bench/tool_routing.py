"""Measure how much of a successful tool trajectory fits in a smaller prompt."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean
import time

from bench.encoder import ASSETS, REVISION, char_similarity, char_vector, encode, jsonl
from ping_floyd.agent import TOOL_ARGS, toolset
from ping_floyd.domain import Task
from ping_floyd.tool_menu import shortlist

DESCRIPTIONS = {
    "calc": "Evaluate an arithmetic expression, percentages, fractions, integer division or rounding.",
    "linear": "Solve simultaneous linear equations, find the unknowns from equalities or a system of constraints.",
    "congruences": "Find an integer by periodic remainders, divisibility, modular arithmetic and congruences.",
    "grid_paths": "Count routes through a grid with blocked squares, moving right and down.",
    "balance": "Find original inventory across two depots from incoming and outgoing stock, ratio and combined totals.",
    "graph": "Find the shortest or longest weighted directed path between two nodes in a graph.",
    "read": "Read text from a named file and inspect its contents or particular lines.",
    "grep": "Search workspace files for a literal, phrase, keyword, exact string or matching lines.",
    "schema": "Inspect columns, headers, table names and sample rows of a database or delimited file.",
    "query": "Filter rows of a table, calculate count, sum, average, minimum, maximum, group totals or distinct values.",
    "sql": "Join multiple tables, run SQL in SQLite, aggregate data from several tables or use a window query.",
    "ledger": "Calculate order totals after revisions and refunds, join journal entries to the account registry.",
    "kb_facts": "Find a named entity in a Markdown knowledge base and read its fields.",
    "lookup": "Find a field of a named entity in a knowledge base, follow relationships to find a fact about a linked entity.",
    "kb_route": "Follow several links in a knowledge base, find a route or distance between named entities.",
    "kb_select": "Filter named Markdown records by matching several fields and return the heading.",
    "latest_notice": "Read the policy and find the latest approved notice for a project, report its code.",
    "latest_kb_notice": "Find the newest eligible Markdown notice by project, scope, approval and date.",
}
SHORTLIST_SIZES = (2, 3, 4)


def collect(name, suite, trace):
    checks = {e["grade"] for e in jsonl(trace / "grade.jsonl") if e.get("outcome") == "correct_answers"}
    events = jsonl(trace / "run.jsonl")
    tools = defaultdict(list)
    for event in events:
        if event["event"] == "tool" and event["task"] in checks:
            tools[event["task"]].append(event["tool"])
    cases = json.loads((suite / "metadata.json").read_text())["cases"]
    rows = []
    for key, used in sorted(tools.items()):
        task = Task.load(suite / "tasks" / key / "task.json")
        if task.kind == "coding":
            continue
        rows.append({
            "id": key, "suite": name, "family": cases[key]["family"], "kind": task.kind,
            "text": task.instruction, "used": list(dict.fromkeys(used)),
            "compact": list(toolset(task, None, "compact")), "full": list(toolset(task, None, "full")),
        })
    return rows


def score(rows, menus):
    sizes = [sum(len(f"{tool}: {TOOL_ARGS[tool]}\n".encode()) for tool in menu) for menu in menus]
    first = sum(row["used"][0] in menu for row, menu in zip(rows, menus, strict=True))
    all_tools = sum(set(row["used"]).issubset(menu) for row, menu in zip(rows, menus, strict=True))
    return {
        "first_tool_covered": first,
        "whole_trajectory_covered": all_tools,
        "avg_menu_bytes": round(mean(sizes), 1),
        "avg_tools": round(mean(map(len, menus)), 2),
    }


def evaluate(rows, vectors):
    comparisons = {"compact": [row["compact"] for row in rows]}
    for limit in SHORTLIST_SIZES:
        for method in ("rules", "char_ngrams", "minilm_int8"):
            menus = []
            for row in rows:
                # Apply the same cap to all selectors and never grow the compact menu.
                cap = min(limit, len(row["compact"]))
                if method == "rules":
                    rank = shortlist(row["kind"], row["compact"], cap)
                elif method == "char_ngrams":
                    rank = sorted(row["full"], reverse=True, key=lambda tool: char_similarity(
                        vectors["char_ngrams"][row["text"]], vectors["char_ngrams"][DESCRIPTIONS[tool]]))
                else:
                    rank = sorted(row["full"], reverse=True, key=lambda tool: float(
                        vectors["minilm_int8"][row["text"]] @ vectors["minilm_int8"][DESCRIPTIONS[tool]]))
                menus.append(rank[:cap])
            comparisons[f"{method}_top{limit}"] = menus
    return {
        "tasks": len(rows),
        "policies": {name: score(rows, menus) for name, menus in comparisons.items()},
        "top3_misses": {name: [{"task": row["id"], "family": row["family"],
                                "omitted": sorted(set(row["used"]) - set(menu))}
                               for row, menu in zip(rows, menus, strict=True) if not set(row["used"]).issubset(menu)]
                         for name, menus in comparisons.items() if name.endswith("_top3")},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", action="append", nargs=3, required=True, metavar=("NAME", "SUITE", "TRACE"))
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [row for name, suite, trace in args.trace for row in collect(name, Path(suite), Path(trace))]
    if not rows:
        parser.error("no successful non-coding tool trajectories found")
    texts = list(dict.fromkeys([row["text"] for row in rows] + list(DESCRIPTIONS.values())))
    started = time.perf_counter()
    vectors = {"char_ngrams": {text: char_vector(text) for text in texts}}
    char_time = time.perf_counter() - started
    started = time.perf_counter()
    vectors["minilm_int8"] = encode(texts, args.assets)
    model_time = time.perf_counter() - started
    results = {
        "sources": args.trace,
        "encoder_revision": REVISION,
        "encoder_assets_bytes": sum((args.assets / path).stat().st_size for path in ASSETS),
        "vectors": len(texts),
        "vectorization_seconds": {"char_ngrams": round(char_time, 3), "minilm_int8": round(model_time, 3)},
        "all": evaluate(rows, vectors),
        "by_kind": {kind: evaluate([row for row in rows if row["kind"] == kind], vectors)
                    for kind in sorted({row["kind"] for row in rows})},
    }
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
