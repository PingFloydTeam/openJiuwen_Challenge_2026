"""Seeded extension with deeper reasoning and multi-file repairs."""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random
import shutil
import sqlite3

from .large import archive, dump, markdown, verify_coding


class Suite:
    def __init__(self, root, seed, base):
        self.root = root
        self.seed = seed
        self.rng = random.Random(seed)
        self.oracle, self.cases = {}, {}
        root.mkdir(parents=True, exist_ok=False)
        if base:
            self.oracle = json.loads((base / "oracle.json").read_text())
            self.cases = json.loads((base / "metadata.json").read_text())["cases"]
            if set(self.oracle) != set(self.cases) or any((base / "tasks").glob("*/answer.txt")):
                raise ValueError("use an unmodified base suite with grading metadata")
            shutil.copytree(base / "tasks", root / "tasks")
            if (base / "judge").is_dir():
                shutil.copytree(base / "judge", root / "judge")

    def case_id(self, kind, family, variant):
        key = f"{self.seed}/{kind}/{family}/{variant}".encode()
        return f"hard_{kind}_{hashlib.sha256(key).hexdigest()[:12]}"

    def add(self, kind, family, variant, instruction, answer, *, assets=None, references=None,
            hidden=None, reason=None, check="exact_int"):
        name = self.case_id(kind, family, variant)
        folder = self.root / "tasks" / name
        folder.mkdir(parents=True)
        dump(folder / "task.json", {"task_id": name, "task_type": kind, "instruction": instruction,
                                   "check": check, "reward": self.rng.randint(1, 5)})
        assets = assets or {}
        for filename, text in assets.items():
            target = folder / "workspace" / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        private_files = {}
        if references:
            judge = self.root / "judge" / name
            judge.mkdir(parents=True)
            for filename, source in references.items():
                reference = f"ref_{filename}"
                private_files[filename] = reference
                (judge / reference).write_text(source, encoding="utf-8")
            (judge / "test_private.py").write_text(hidden, encoding="utf-8")
        self.oracle[name] = answer
        self.cases[name] = {"kind": kind, "family": family, "split": "hard",
                            "reason": {"category": reason[0], "detail": reason[1]} if reason else None,
                            "reference_files": private_files,
                            "public_tests": {path: hashlib.sha256(text.encode()).hexdigest()
                                             for path, text in assets.items() if path.startswith("test_")}}
        return folder / "workspace"

    def finish(self):
        dump(self.root / "oracle.json", self.oracle)
        dump(self.root / "metadata.json", {"seed": self.seed, "cases": self.cases})


def math_cases(suite):
    def linear(name, i, equations, variable, answer, reason=None):
        suite.add("math", name, i, "Solve over the rationals: " + "; ".join(equations)
                  + f". Return {variable}.", answer, check="exact_str", reason=reason)

    def modular(name, i, pairs, lower, upper, mode, answer, reason=None):
        rules = "; ".join(f"x mod {period} = {residue}" for residue, period in pairs)
        suite.add("math", name, i, f"Find the {mode} integer in [{lower}, {upper}] satisfying: {rules}.",
                  answer, reason=reason)

    for i in range(5):
        x, y, z = suite.rng.sample(range(-15, 16), 3)
        equations = [f"2*y-z={2*y-z}", f"3*x+y+2*z={3*x+y+2*z}", f"x-2*y+z={x-2*y+z}"]
        linear("signed_system", i, equations, "z", str(z))
        frac = Fraction(2*i + 3, 7)
        linear("fraction", i, [f"4*x-3*y={4*frac-3*y}", f"x+2*y={frac+2*y}"], "x", str(frac))
        linear("unique_coordinate", i, [f"x+y+z-w={x+y+z}", f"y+z-w={y+z}", f"2*z-w={i}"], "x", str(x))
        lower, upper = -10+i, 150+i
        target = 5 + 13*i
        pairs = [(target % m, m) for m in (6, 8, 9)]
        answers = [n for n in range(lower, upper+1) if all(n % m == r for r, m in pairs)]
        modular("noncoprime", i, pairs, lower, upper, "least", answers[0])
    linear("conflicting", 0, ["x+y=4", "2*x+2*y=9"], "x", None, ("logical_error", "inconsistent equations"))
    linear("free_coordinate", 0, ["2*x+y=7", "4*x+2*y=14"], "x", None,
           ("missing_information", "requested coordinate is free"))
    linear("zero_division", 0, ["x/(4-4)+y=2"], "x", None, ("undefined_math", "zero denominator"))
    modular("conflicting_residues", 0, [(1, 6), (2, 8)], 0, 100, "least", None,
            ("logical_error", "incompatible residues modulo their common divisor"))
    modular("ambiguous_residues", 0, [(0, 4), (0, 6)], 0, 30, "unique", None,
            ("missing_information", "several solutions in range"))


def search_cases(suite):
    for i in range(5):
        headings = [f"{prefix}-{i+10}" for prefix in ("Dock", "Gate", "Span", "Hub", "Depot", "Back", "Relay")]
        start, gate, span, hub, end, back, relay = headings
        first, decoy = (gate, relay) if i % 2 == 0 else (relay, gate)
        interior = [first, span, hub, f"Tunnel-{i}", f"Bridge-{i}"][:2 + i % 4]
        path = [start, *interior, end]
        edges = {node: {} for node in (*path, back, decoy)}
        for step, (left, right) in enumerate(zip(path, path[1:])):
            edges[left]["Rail" if step % 2 else "Road"] = right
        edges[start]["Rail"] = back
        edges[back] = {"Road": start, "Rail": decoy}
        edges[decoy] = {"Road": back}
        docs = {f"kb/node{j}.md": markdown(name, values) for j, (name, values) in enumerate(edges.items())}
        route = f"In kb/, take the shortest directed route from {start} to {end} using Road, Rail links."
        suite.add("search", "directed_distance", i, route + " Return the number of links.", len(path)-1, assets=docs)
        suite.add("search", "directed_first", i, route + " Return the first heading after the start.", first,
                  check="exact_str", assets=docs)
        catalog = {}
        regions, levels = ["Delta", "North", "South"], ["junior", "senior"]
        target = 3*i+2
        for j in range(18):
            catalog[f"kb/item{j}.md"] = markdown(f"Employee-{i+10}-{j:02}", {
                "Region": regions[j % 3], "Level": levels[j % 2], "Badge": "violet" if j % 3 == 2 else "blue",
                "Shift": f"Shift-{j // 3}",
            })
        suite.add("search", "reverse_filter", i,
                  f"In kb/, return the unique heading with Region = '{regions[target % 3]}' and "
                  f"Level = '{levels[target % 2]}' and Badge = 'violet' and Shift = 'Shift-{target // 3}'.",
                  f"Employee-{i+10}-{target:02}", check="exact_str", assets=catalog)
        city = suite.rng.choice(["Riga", "Oslo", "Lima", "Pune", "Accra"])
        steps = 4+i % 3
        ordinal = ("fourth", "fifth", "sixth")[i % 3]
        office = f"Studio-{i}"
        chain = {f"kb/link{j}.md": markdown(heading, {"Mentor": headings[j+1]} if j < steps else {"Office": office})
                 for j, heading in enumerate(headings[:steps+1])}
        chain["kb/office.md"] = markdown(office, {"City": city})
        suite.add("search", "deep_chain", i,
                  f"In kb/, what city is listed for the office of {start}'s {ordinal} successive mentor?",
                  city, check="exact_str", assets=chain)
    def bad(name, docs, query, why):
        suite.add("search", name, 0, query, None, check="exact_str",
                  assets={f"kb/{key}.md": markdown(key, values) for key, values in docs.items()}, reason=why)
    route = "In kb/, take the shortest directed route from A to D using Road, Rail links. Return the first heading after the start."
    bad("tied_route", {"A":{"Road":"B","Rail":"C"},"B":{"Road":"D"},"C":{"Rail":"D"},"D":{}},
        route, ("missing_information", "shortest routes start with B and C"))
    bad("unreachable", {"A":{"Road":"B"},"B":{"Road":"A"},"D":{}},
        route, ("logical_error", "no directed route to D"))
    bad("missing_node", {"A":{"Road":"B"},"D":{}}, route, ("missing_information", "B has no record"))
    bad("duplicate_filter", {"A":{"Team":"Gold"},"B":{"Team":"Gold"}},
        "In kb/, return the unique heading with Team = 'Gold'.", ("missing_information", "two matching headings"))
    bad("missing_link", {"A":{"Mentor":"B"},"B":{"Office":"C"}},
        "In kb/, what city is listed for the office of A's first mentor?",
        ("missing_information", "office C has no record"))


def data_cases(suite):
    def database(name, i, question, answer, accounts, orders, reason=None, check="exact_int"):
        root = suite.add("data_analysis", name, i, "Using only commerce.sqlite, " + question,
                         answer, reason=reason, check=check)
        root.mkdir()
        conn = sqlite3.connect(root / "commerce.sqlite")
        with conn:
            conn.executescript("CREATE TABLE accounts (id INTEGER PRIMARY KEY, region TEXT, active INTEGER);"
                               "CREATE TABLE orders (id INTEGER PRIMARY KEY, account_id INTEGER, month INTEGER,"
                               "units INTEGER, cents INTEGER);")
            conn.executemany("INSERT INTO accounts VALUES (?,?,?)", accounts)
            conn.executemany("INSERT INTO orders VALUES (?,?,?,?,?)", orders)
        conn.close()

    for i in range(5):
        accounts = [(j, ["North", "South", "West"][j % 3], j % 2) for j in range(1, 17+i)]
        lookup = {j: (region, active) for j, region, active in accounts}
        orders = [(j, suite.rng.randrange(1, 15+i), 1+suite.rng.randrange(6),
                   1+suite.rng.randrange(9), 20+suite.rng.randrange(200)) for j in range(1, 121+i*12)]
        relevant = [row for row in orders if lookup[row[1]][0] == "North" and 2 <= row[2] <= 5]
        database("joined_revenue", i,
                 "join orders.account_id to accounts.id. What is the sum of units times cents for North accounts in months 2 through 5 inclusive?",
                 sum(row[3]*row[4] for row in relevant), accounts, orders)
        sold = {row[1] for row in orders if row[2] == 1}
        database("antijoin", i,
                 "how many active accounts have no orders in month 1? Accounts.active=1 means active.",
                 sum(active and ident not in sold for ident, _, active in accounts), accounts, orders)
        first = {}
        for row in sorted(orders, key=lambda item: (item[2], item[0])):
            first.setdefault(row[1], row)
        database("first_order", i,
                 "among the first order per account (earliest month, lowest order id breaks ties), what is the sum of units?",
                 sum(row[3] for row in first.values()), accounts, orders)
        totals = {}
        for _, ident, _, units, cents in orders:
            region = lookup[ident][0]
            quantity, revenue = totals.get(region, (0, 0))
            totals[region] = quantity+units, revenue+units*cents
        rates = {region: Fraction(revenue, units) for region, (units, revenue) in totals.items()}
        best = [region for region, rate in rates.items() if rate == max(rates.values())]
        if len(best) != 1:
            raise ValueError("seed produced a tie in a solvable fixture")
        database("grouped_ratio", i,
                 "join orders to accounts on orders.account_id = accounts.id. Which region has the largest unit-weighted price, defined as sum(units*cents)/sum(units)? Return one region.",
                 best[0], accounts, orders, check="exact_str")
    accounts = [(1,"North",1),(2,"South",1)]
    orders = [(1,1,2,2,100),(2,2,2,5,100)]
    for i, (query, reason, check) in enumerate([
        ("what is the total orders.tax_cents?", ("missing_information","tax_cents is absent"), "exact_int"),
        ("what is the count of rows in payments?", ("missing_information","payments is absent"), "exact_int"),
        ("which region has a strictly higher unit-weighted price than all others? Join account_id to id.",
         ("logical_error","the two rates are equal"), "exact_str"),
        ("what is the average units per order in month 12?", ("undefined_math","month 12 has no orders"), "exact_int"),
        ("what is the unique account_id for orders in month 2?", ("missing_information","there are two accounts"), "exact_int"),
    ]):
        database("indeterminate", i, query, None, accounts, orders, reason, check)


def tests(assertions):
    body = "\n".join(f"        self.assertEqual({expr}, {answer!r})" for expr, answer in assertions)
    return ("import unittest\nimport pipeline\nimport rules\n\n"
            "class Checks(unittest.TestCase):\n    def test_contract(self):\n" + body +
            "\n\nif __name__ == '__main__':\n    unittest.main()\n")


def coding_cases(suite):
    def repair(name, i, contract, rules, pipeline, public, hidden):
        suite.add("coding", name, i, contract + " Run `python3 test_contract.py`. "
                  "Fix both modules and keep the original tests. Return DONE.", "tests_pass", check="unittest",
                  assets={"rules.py":rules[0], "pipeline.py":pipeline[0], "test_contract.py":tests(public)},
                  references={"rules.py":rules[1], "pipeline.py":pipeline[1]}, hidden=tests(hidden))

    for i in range(5):
        n = 10+i*3
        repair("decode", i, "Expand run-length pairs in order. Zero repetitions add nothing.",
               ("def copies(value, count):\n    return [value] * (count - 1)\n",
                "def copies(value, count):\n    return [value] * count\n"),
               ("from rules import copies\n\ndef decode(pairs):\n    return [x for value, n in pairs[1:] for x in copies(value, n)]\n",
                "from rules import copies\n\ndef decode(pairs):\n    return [x for value, n in pairs for x in copies(value, n)]\n"),
               [(f"rules.copies({n},2)", [n,n]), (f"pipeline.decode([({n},2),(3,1)])", [n,n,3])],
               [("pipeline.decode([(9,0),('x',3)])", ["x"]*3), ("pipeline.decode([])", [])])
        repair("grid", i, "A movement is (dx, dy). Walk from the supplied start using all steps.",
               ("def move(p, d):\n    return p[0]+d[1], p[1]+d[0]\n",
                "def move(p, d):\n    return p[0]+d[0], p[1]+d[1]\n"),
               ("from rules import move\n\ndef walk(p, steps):\n    for step in steps[:1]:\n        p = move(p, step)\n    return p\n",
                "from rules import move\n\ndef walk(p, steps):\n    for step in steps:\n        p = move(p, step)\n    return p\n"),
               [(f"rules.move(({n},2),(1,4))", (n+1,6)), (f"pipeline.walk(({n},2),[(1,4),(2,-6)])", (n+3,0))],
               [("pipeline.walk((5,-2),[(2,0),(-3,1),(0,-5)])", (4,-6)), ("pipeline.walk((8,1),[])", (8,1))])
        repair("ledger", i, "A refund decreases a balance. Accumulate all entries separately per account.",
               ("def signed(value, refund):\n    return value if refund else -value\n",
                "def signed(value, refund):\n    return -value if refund else value\n"),
               ("from rules import signed\n\ndef balances(entries):\n    result = {}\n    for key, value, refund in entries:\n        result[key] = signed(value, refund)\n    return result\n",
                "from rules import signed\n\ndef balances(entries):\n    result = {}\n    for key, value, refund in entries:\n        result[key] = result.get(key, 0) + signed(value, refund)\n    return result\n"),
               [(f"rules.signed({n},True)", -n),
                (f"pipeline.balances([('a',{n},False),('a',2,True),('b',3,False)])", {"a":n-2,"b":3})],
               [("pipeline.balances([('z',4,True),('z',4,False),('y',1,True)])", {"z":0,"y":-1}),
                ("pipeline.balances([])", {})])
        repair("closed_intervals", i, "Intervals are closed. Return how many overlapping unordered pairs exist.",
               ("def overlap(a, b):\n    return max(a[0],b[0]) < min(a[1],b[1])\n",
                "def overlap(a, b):\n    return max(a[0],b[0]) <= min(a[1],b[1])\n"),
               ("from rules import overlap\n\ndef pairs(xs):\n    return sum(overlap(a,b) for a,b in zip(xs,xs[1:]))\n",
                "from rules import overlap\n\ndef pairs(xs):\n    return sum(overlap(a,b) for i,a in enumerate(xs) for b in xs[i+1:])\n"),
               [(f"rules.overlap((0,{n}),({n},{n+2}))", True),
                (f"pipeline.pairs([(0,{n}),(20,24),(2,{n})])", 1 if n < 20 else 3)],
               [("pipeline.pairs([(3,3),(1,3),(3,4),(8,9)])", 3), ("pipeline.pairs([])", 0)])
    for i in range(5):
        if i % 2:
            prompt = f"Implement score() as a deterministic function. The same call must return {i} and must return {i+1}. Both requirements are mandatory."
            reason = ("logical_error", "one deterministic call cannot have distinct values")
        else:
            prompt = f"Implement fees(x) for the settlement system. For input {i+10}, return the result required by the settlement policy. The policy and examples are unavailable."
            reason = ("missing_information", "the only fee rule is not available")
        suite.add("coding", "broken_contract", i, prompt, None, check="unittest", reason=reason,
                  assets={"pipeline.py":"def run(x):\n    raise NotImplementedError\n"})


def audit(root):
    oracle = json.loads((root / "oracle.json").read_text())
    cases = json.loads((root / "metadata.json").read_text())["cases"]
    assert set(oracle) == set(cases) == {p.parent.name for p in (root / "tasks").glob("*/task.json")}
    hard = {n: data for n, data in cases.items() if data["split"] == "hard"}
    totals = Counter((entry["kind"], oracle[name] is None) for name, entry in hard.items())
    for kind in ("math", "search", "data_analysis", "coding"):
        assert totals[kind, True] == 5 and totals[kind, False] == 20
    for name, entry in cases.items():
        manifest = root / "tasks" / name / "task.json"
        payload = json.loads(manifest.read_text())
        assert set(payload) == {"task_id", "task_type", "reward", "check", "instruction"}
        assert payload["task_id"] == name and payload["task_type"] == entry["kind"]
        assert (entry["reason"] is not None) == (oracle[name] is None)
        assert not (manifest.parent / "answer.txt").exists()
        for test, digest in entry["public_tests"].items():
            assert hashlib.sha256((manifest.parent / "workspace" / test).read_bytes()).hexdigest() == digest
        for filename in entry.get("reference_files", {}).values():
            assert (root / "judge" / name / filename).is_file()
    print(json.dumps({"tasks": len(oracle), "unsolvable": sum(v is None for v in oracle.values()),
                      "hard": len(hard), "hard_unsolvable": sum(oracle[n] is None for n in hard),
                      "hard_families": len({(e["kind"], e["family"]) for e in hard.values()})}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", type=Path)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--zip", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--verify-coding", action="store_true")
    args = parser.parse_args()
    if not args.audit_only:
        suite = Suite(args.suite, args.seed, args.base)
        for build in (math_cases, search_cases, data_cases, coding_cases):
            build(suite)
        suite.finish()
    audit(args.suite)
    if args.verify_coding:
        verify_coding(args.suite)
    if args.zip:
        archive(args.suite, args.zip)


if __name__ == "__main__":
    main()
