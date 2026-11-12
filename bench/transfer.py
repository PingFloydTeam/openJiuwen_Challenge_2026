"""Small independent controls with original assets inspired by public problems."""
import argparse
from itertools import combinations
from pathlib import Path

from .hard import Suite
from .large import archive, verify_coding

SOURCES = {
    "math": "https://projecteuler.net/problem=15",
    "search": "https://cses.fi/problemset/task/1671",
    "data_analysis": "https://projecteuler.net/problem=18",
    "coding": "https://cses.fi/problemset/task/1679",
}


def count_by_enumeration(rows, cols, obstacles):
    total = 0
    for downs in combinations(range(rows + cols - 2), rows - 1):
        x = y = 1
        cells = {(x, y)}
        for step in range(rows + cols - 2):
            x += step in downs
            y += step not in downs
            cells.add((x, y))
        total += not bool(cells & obstacles)
    return total


def path_sums(edges, start, end, visited=frozenset()):
    if start == end:
        yield 0
    else:
        for left, right, weight in edges:
            if left == start and right not in visited | {start}:
                for suffix in path_sums(edges, right, end, visited | {start}):
                    yield weight + suffix


def graph_case(suite, kind, variant, edges, answer, reason=None):
    mode = "shortest" if kind == "search" else "longest DAG"
    assets = {"links.tsv": "source\ttarget\tweight\n" + "".join(f"{a}\t{b}\t{w}\n" for a, b, w in edges)}
    suite.add(kind, "weighted_routes", variant,
              f"Use directed edges from links.tsv (source,target,weight). Return the {mode} distance from 'A' to 'F'.",
              answer, reason=reason, assets=assets)


def contract(entries, expected, extra=()):
    assertions = "".join(f"        self.assertEqual(schedule({edges!r}), {order!r})\n"
                         for edges, order in ((entries, expected), *extra))
    return ("import unittest\nfrom schedule import schedule\n\nclass Checks(unittest.TestCase):\n"
            f"    def test_order(self):\n{assertions}"
            "\nif __name__ == '__main__':\n    unittest.main()\n")


REFERENCE = '''from heapq import heapify, heappop, heappush

def schedule(edges):
    degrees = {item: 0 for edge in edges for item in edge}
    following = {item: [] for item in degrees}
    for left, right in edges:
        degrees[right] += 1
        following[left].append(right)
    available = [item for item, degree in degrees.items() if degree == 0]
    heapify(available)
    order = []
    while available:
        item = heappop(available)
        order.append(item)
        for neighbor in following[item]:
            degrees[neighbor] -= 1
            if degrees[neighbor] == 0:
                heappush(available, neighbor)
    return order if len(order) == len(degrees) else []
'''


def generate(destination, seed):
    suite = Suite(destination, seed, None)
    for i in range(6):
        rows, cols = 5 + i % 3, 5 + i // 2
        blocked = {(2, 2 + i % 3), (4, 3), (3, 5)}
        cells = ", ".join(f"({a},{b})" for a, b in sorted(blocked))
        suite.add("math", "grid_obstacles", i,
                  f"Count right/down paths in a {rows} by {cols} cell grid from (1,1) to ({rows},{cols}). Blocked cells: {cells}.",
                  count_by_enumeration(rows, cols, blocked))
        edges = [(a, b, suite.rng.randrange(1, 20)) for a, b in
                 (("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"), ("C", "E"), ("D", "F"), ("E", "F"))]
        graph_case(suite, "search", i, edges + [("E", "A", 4)], min(path_sums(edges, "A", "F")))
        edges = [(a, b, weight - 8) for a, b, weight in edges]
        graph_case(suite, "data_analysis", i, edges, max(path_sums(edges, "A", "F")))
        first, second, third = [chr(ord("A") + i * 3 + j) for j in range(3)]
        hidden = contract([(third, first), (second, first)], [second, third, first],
                          extra=(([(third, first), (first, third)], []), ([], [])))
        if i % 2:
            broken = REFERENCE.replace("return order if len(order) == len(degrees) else []", "return order")
            public = contract([(first, third), (third, first), (second, first)], [])
        else:
            broken = REFERENCE.replace("item = heappop(available)", "item = available.pop()")
            public = contract([(first, third), (second, third)], [first, second, third])
        suite.add("coding", "topological_order", i,
                  "Repair schedule(edges) in schedule.py. Return the lexicographically smallest valid topological order "
                  "of the course IDs in the edge list, or [] for a cycle. Run `python3 test_schedule.py`. Return DONE.",
                  "tests_pass", check="unittest", assets={"schedule.py": broken, "test_schedule.py": public},
                  references={"schedule.py": REFERENCE}, hidden=hidden)

    for i in range(2):
        suite.add("math", "undefined_grid_ratio", i,
                  f"Evaluate exactly {count_by_enumeration(4+i, 5, set())} / (12-12).", None,
                  reason=("undefined_math", "the denominator is zero"))
        graph_case(suite, "search", 6+i, [("A", "B", 1+i), ("F", "A", 2)], None,
                   ("logical_error", "F is unreachable from A"))
        graph_case(suite, "data_analysis", 6+i, [("A", "B", i), ("B", "A", 1), ("B", "F", 2)], None,
                   ("logical_error", "the directed graph is cyclic"))
        suite.add("coding", "contradictory_order", i,
                  f"Implement schedule() as a deterministic function. The same call must return {i} "
                  f"and must return {i+1}. Both requirements are mandatory.", None, check="unittest",
                  assets={"schedule.py": "def schedule():\n    raise NotImplementedError\n"},
                  reason=("logical_error", "one call is required to return distinct values"))
    for entry in suite.cases.values():
        entry.update(split="transfer", source=SOURCES[entry["kind"]])
    suite.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", type=Path)
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--zip", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--verify-coding", action="store_true")
    args = parser.parse_args()
    if not args.audit_only:
        generate(args.suite, args.seed)
    if args.verify_coding:
        verify_coding(args.suite)
    if args.zip:
        archive(args.suite, args.zip)


if __name__ == "__main__":
    main()
