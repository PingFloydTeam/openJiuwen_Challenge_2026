"""Adversarial but judge-shaped tasks with local workspaces and hidden oracles."""
import argparse
from collections import Counter
from fractions import Fraction
from io import StringIO
from pathlib import Path
import csv
import json
import shutil
import sqlite3

from .hard import Suite
from .large import archive, dump, markdown, verify_coding


UNKNOWN = ("missing_information", "the supplied facts do not identify a unique answer")
CONFLICT = ("logical_error", "the requirements contradict each other")
UNDEFINED = ("undefined_math", "the requested operation has no defined result")


def math_cases(suite):
    for i in range(5):
        start = 330 + suite.rng.randrange(75)
        returns = 22 + i
        transfers = 12 + 3*i
        text = (f"В начале дня было {start} ящиков. Утром склад передал треть, округлив число отданных вниз. "
                f"Затем вернули {returns} ящиков, после чего списали четверть текущего остатка, также округлив списание вниз. "
                f"Вечером склад получил {transfers} ящиков, а затем оставил себе две пятых нового остатка, округлив вниз. "
                "Начальный отчёт по другой площадке содержал 400 ящиков. Сколько ящиков оставил себе этот склад?")
        after_return = start - start//3 + returns
        answer = (after_return - after_return//4 + transfers)*2//5
        suite.add("math", "ordered_rounding", i, text, answer)

        amount, limit = 60 + 3*i, 13 + i
        price_a, price_b, price_c = 3, 5 + i, 7
        allowed = range(limit+1)
        ways = sum(a+b+c == limit and a*price_a+b*price_b+c*price_c <= amount and b >= 1 and c % 2 == 0
                   for a in allowed for b in allowed for c in allowed)
        suite.add("math", "bounded_purchases", i,
                  f"Buy exactly {limit} items. A, B, and C cost {price_a}, {price_b}, and {price_c} credits per item. "
                  f"Spend at most {amount}. Buy at least one B and an even number of C, which may be zero. "
                  "How many distinct nonnegative quantity triples (A,B,C) satisfy all constraints? Count quantity triples only.", ways)

        intervals = [(j, j+suite.rng.randrange(1, 8)) for j in suite.rng.sample(range(-4, 21), 12)]
        count = sum(sum(a <= tick < b for a, b in intervals) >= 3 for tick in range(-4, 29))
        suite.add("math", "overlap_duration", i,
                  f"Reservations occupy these half-open time intervals [start,end): {intervals}. "
                  "What is the total duration with at least three reservations active, in whole time units?", count)

        lo, hi = -37-i, 100+i
        anchor = 4 + 3*i
        mods = [6, 10, 7]
        residues = [anchor % n for n in mods]
        numbers = [n for n in range(lo, hi+1)
                   if (n % mods[0] == residues[0] or n % mods[1] == residues[1]) and n % mods[2] != residues[2]]
        suite.add("math", "modular_inclusion", i,
                  f"Count integers from {lo} through {hi}, including both endpoints. Include a number if its remainder "
                  f"modulo 6 is {residues[0]} or modulo 10 is {residues[1]} (or both), but exclude it if its remainder "
                  f"modulo 7 is {residues[2]}. Use remainders between zero and modulus minus one.", len(numbers))

    for i, (question, reason) in enumerate((
        ("The warehouse ended with 15 boxes after rounding half of its starting stock down. Give the unique starting count.", UNKNOWN),
        ("An order used a nonempty set of boxes. Each box contained 4 pieces and the full order contained 19 pieces. How many boxes were used?", CONFLICT),
        ("Calculate the arithmetic mean of the positive entries of [0, -1, -5].", UNDEFINED),
        ("Two boats separated by 14 km travel in opposite directions at 2 and 3 km/h for 2 hours. Their headings relative to the other boat are unspecified. Give their unique final separation.", UNKNOWN),
        ("A region is covered by half-open intervals [0,4), [2,t), [3,8). The only restriction is 2 < t < 10. Give the unique total covered length.", UNKNOWN),
    )):
        suite.add("math", "nonunique_conditions", i, question, None, reason=reason)


def search_cases(suite):
    for i in range(5):
        person, employer = f"Dana Sen-{i}", f"Orchid Lab-{i}"
        city, wrong = suite.rng.sample(["Lima", "Riga", "Oslo", "Pune", "Quito"], 2)
        badge = f"00{i+1}"
        files = {
            "roster.tsv": f"badge\tname\n{badge}\t{person}\n{i+1}\tDara Sen-{i}\n",
            "kb/assigned.md": markdown(person, {"Employer": employer}),
            "kb/neighbor.md": markdown(f"Dara Sen-{i}", {"Employer": f"Orchid Labs-{i}"}),
            "kb/current.md": markdown(employer, {"Headquarters": city}),
            "kb/similar.md": markdown(f"Orchid Labs-{i}", {"Headquarters": wrong}),
        }
        for j in range(40):
            files[f"kb/reference-{j:02}.md"] = markdown(f"Orchid Research-{i}-{j}", {"Headquarters": wrong})
        suite.add("search", "badge_identity", i,
                  f"Resolve badge `{badge}` in roster.tsv as an exact string, including leading zeroes. Find that person's employer "
                  "in kb/ and then the headquarters city of that employer. Report the city.", city, check="exact_str", assets=files)

        project, code = f"Fern-{i}", f"FN-{900+i}"
        base = {"Project": project, "Status": "approved", "Scope": "local"}
        notices = {
            "kb/a.md": markdown("Notice A", base | {"Date": "2025-04-01", "Contact code": "STALE"}),
            "kb/b.md": markdown("Notice B", base | {"Date": "2025-04-09", "Contact code": code}),
            "kb/c.md": markdown("Notice C", base | {"Date": "2025-04-10", "Status": "draft", "Contact code": "DRAFT"}),
            "kb/d.md": markdown("Notice D", base | {"Date": "2025-04-11", "Scope": "global", "Contact code": "GLOBAL"}),
        }
        question = (f"For project {project}, what is the contact code in the newest local approved notice in kb/? "
                    "Date is the issue date. A global notice or a draft cannot replace a local approved notice.")
        suite.add("search", "approved_precedence", i, question, code, check="exact_str", assets=notices)
        duplicate = markdown("Notice E", base | {"Date": "2025-04-09", "Contact code": "CONFLICT"})
        suite.add("search", "equal_priority_conflict", i, question, None, check="exact_str", reason=UNKNOWN,
                  assets=notices | {"kb/e.md": duplicate})

        route = [f"Relay-{i}-{j}" for j in range(8)]
        nodes = {f"kb/hop{j}.md": markdown(name, {"Approved successor": route[j+1], "Draft successor": "Wrong Relay"})
                 for j, name in enumerate(route[:-1])}
        terminal = f"КП-{i}{suite.rng.randrange(10, 99)}"
        nodes["kb/end.md"] = markdown(route[-1], {"Permit code": terminal})
        nodes["kb/decoy.md"] = markdown("Wrong Relay", {"Permit code": "WRONG"})
        suite.add("search", "approved_chain", i,
                  f"Start at {route[0]} in kb/. Follow the Approved successor field exactly seven times. "
                  "Draft successor is only a proposal. Give the permit code of the last record.",
                  terminal, assets=nodes, check="exact_str")

        fact = f"АБ-{i}-007"
        suite.add("search", "literal_with_distractors", i,
                  "Use index.tsv to select the notice whose serial is `007` and stage is `release`. "
                  "Open the corresponding file and report the service code exactly as printed, preserving its alphabet.",
                  fact, check="exact_str", assets={
                      "index.tsv": "serial\tstage\tpath\n7\trelease\tdocs/a.txt\n007\tdraft\tdocs/b.txt\n007\trelease\tdocs/c.txt\n",
                      "docs/a.txt": "Service code: AB-007\n", "docs/b.txt": "Service code: OLD-007\n",
                      "docs/c.txt": f"Service code: {fact}\n",
                  })


def table(header, rows):
    stream = StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(header)
    writer.writerows(rows)
    return stream.getvalue()


def data_cases(suite):
    for i in range(5):
        amounts = [suite.rng.randrange(10, 100) for _ in range(6)]
        rows = [("p1", "001", 1, "posted", "sale", amounts[0]), ("p1", "001", 2, "draft", "sale", amounts[1]),
                ("p2", "001", 1, "posted", "refund", amounts[2]), ("p3", "1", 1, "posted", "sale", amounts[3]),
                ("p4", "010", 1, "posted", "sale", amounts[4]), ("p4", "010", 2, "posted", "sale", amounts[5])]
        files = {"events.csv": table(("order", "account", "revision", "state", "event", "cents"), rows),
                 "accounts.csv": "account,region\n001,North\n1,South\n010,North\n"}
        question = ("Use events.csv and accounts.csv. Discard draft rows before selecting the greatest revision per order. "
                    "Join account IDs as exact strings with leading zeroes. For North, add sale cents and subtract refund cents. "
                    "An unregistered account can belong to North or South. Give net cents for North.")
        suite.add("data_analysis", "revision_after_filter", i, question, amounts[0]-amounts[2]+amounts[5], assets=files)
        suite.add("data_analysis", "missing_foreign_key", i, question, None, reason=UNKNOWN,
                  assets=files | {"accounts.csv": files["accounts.csv"].replace("010,North\n", "")})

        samples = [(account, month, j, (account+2)*(j+1)+i) for account in range(1, 7)
                   for month in (1, 2, 3) for j in range(1, 4)]
        expected = sum(value for _, month, version, value in samples if month == 2 and version == 3)
        workspace = suite.add("data_analysis", "sqlite_rank_after_cutoff", i,
                  "In telemetry.sqlite, table readings has account, month, version, value. Consider only month 2. "
                  "Keep the highest version separately for each account and sum the value column.", expected)
        workspace.mkdir(exist_ok=True)
        db = sqlite3.connect(workspace / "telemetry.sqlite")
        with db:
            db.execute("CREATE TABLE readings(account INTEGER, month INTEGER, version INTEGER, value INTEGER)")
            db.executemany("INSERT INTO readings VALUES (?,?,?,?)", samples)
        db.close()

        minutes = [i+1, i+4, i+9, 60, 90]
        latency = {"requests.tsv": "key\tzone\tlatency_ms\n" + "".join(
            f"id-{j}\t{'east' if j < 3 else 'west'}\t{value}\n" for j, value in enumerate(minutes))}
        val = Fraction(sum(minutes[:3]), 3)
        suite.add("data_analysis", "exact_average", i,
                  "For the east zone in requests.tsv, report the arithmetic mean of latency_ms as a reduced fraction "
                  "(use an integer if exact).", str(val), check="exact_str", assets=latency)

        score_rows = [("B", 30+i, 2), ("B", 10, 0), ("A", 40+i, 3), ("C", 70, 9)]
        suite.add("data_analysis", "tie_break_with_cutoff", i,
                  "In bids.csv retain rows with day <= 3, then sum amount by bidder. Return the bidder with the largest sum. "
                  "If sums tie, return the alphabetically first bidder.", "A", check="exact_str",
                  assets={"bids.csv": table(("bidder", "amount", "day"), score_rows)})


WINDOWS = """def merge(intervals):
    result = []
    for start, end in sorted(intervals):
        if start == end:
            continue
        if result and start < result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result
"""
LEDGER = """from rules import can_apply

def apply_balances(balances, changes):
    result = dict(balances)
    for name, delta in changes:
        previous = result.get(name, 0)
        if not can_apply(previous, delta):
            return None
        result[name] = previous + delta
    return result
"""
ROLLING = """from collections import deque

def window_totals(values, width):
    if width <= 0:
        raise ValueError('width must be positive')
    active = deque()
    total = 0
    result = []
    for value in values:
        active.append(value)
        total += value
        if len(active) > width:
            total -= active.popleft()
        if len(active) == width:
            result.append(total)
    return result
"""
ALIASES = """def resolve(name, links):
    visited = set()
    while name in links:
        if name in visited:
            return None
        visited.add(name)
        name = links[name]
    return name
"""


def coding_cases(suite):
    def tests(body):
        header = "import unittest\nfrom app import *\n\nclass Contract(unittest.TestCase):\n    def test_contract(self):\n"
        return header + "\n".join("        " + line for line in body) + "\n\nif __name__ == '__main__':\n    unittest.main()\n"

    def add(family, i, prompt, original, reference, public, hidden):
        suite.add("coding", family, i, prompt + " Run `python3 test_app.py`. Fix the source and preserve the tests. Return DONE.",
                  "tests_pass", assets=original | {"test_app.py": tests(public)}, check="unittest",
                  references=reference, hidden=tests(hidden))

    for i in range(5):
        limit = 8+i
        add("non_touching_windows", i, "Merge overlapping half-open intervals with integer endpoints. Adjacent intervals "
            "do not overlap. Ignore empty intervals. Return sorted pairs and leave the input unchanged.",
            {"app.py": WINDOWS.replace("start < result", "start <= result")}, {"app.py": WINDOWS},
            [f"self.assertEqual(merge([({limit}, 20), (0, {limit})]), [(0, {limit}), ({limit}, 20)])"],
            ["data = [(3, 6), (0, 3), (2, 4), (4, 4), (9, 12), (9, 12)]",
             "self.assertEqual(merge(data), [(0, 6), (9, 12)])",
             "self.assertEqual(data, [(3, 6), (0, 3), (2, 4), (4, 4), (9, 12), (9, 12)])",
             "self.assertEqual(merge([(-6, -6)]), [])"])
        rules = "def can_apply(previous, delta):\n    return previous + delta >= 0\n"
        add("atomic_ledger", i, "Apply signed account deltas in order. Missing accounts start at zero. If any intermediate "
            "balance would be negative, return None. Otherwise return the new dictionary. Never change the input dictionary.",
            {"app.py": LEDGER.replace("dict(balances)", "balances"), "rules.py": rules.replace(">=", ">")},
            {"app.py": LEDGER, "rules.py": rules},
            [f"data = {{'a': {limit}}}", f"self.assertEqual(apply_balances(data, [('a', -{limit}), ('b', 2)]), {{'a': 0, 'b': 2}})",
             f"self.assertEqual(data, {{'a': {limit}}})"],
            ["data = {'a': 5}", "self.assertIsNone(apply_balances(data, [('a', 1), ('b', -1), ('b', 9)]))",
             "self.assertEqual(data, {'a': 5})", "self.assertEqual(apply_balances({}, [('b', 2), ('b', -2)]), {'b': 0})"])
        data = [i, 4, -3, 0, 6, i+2]
        width = 2+i%3
        expected = [sum(data[j:j+width]) for j in range(len(data)-width+1)]
        add("one_pass_rolling", i, "Return sums of all consecutive windows of exactly width in an iterable. The iterable may "
            "be a one-shot iterator. An incomplete final window produces no result. A nonpositive width raises ValueError.",
            {"app.py": ROLLING.replace("total -= active.popleft()", "total -= active.pop()")}, {"app.py": ROLLING},
            [f"self.assertEqual(window_totals(iter({data!r}), {width}), {expected!r})"],
            ["self.assertEqual(window_totals(iter([3, -2, 7]), 1), [3, -2, 7])",
             "self.assertEqual(window_totals(iter([3]), 2), [])", "self.assertEqual(window_totals(iter([]), 3), [])",
             "with self.assertRaises(ValueError):", "    window_totals(iter([1]), 0)"])
        add("transitive_aliases", i, "Resolve exact names through the directed mapping links. Return the first terminal name "
            "with no outgoing link, including the input name if absent. A repeated name means a cycle, so return None. "
            "Do not modify links.", {"app.py": "def resolve(name, links):\n    return links.get(name, name)\n"},
            {"app.py": ALIASES},
            [f"self.assertEqual(resolve('a{i}', {{'a{i}': 'b', 'b': 'c'}}), 'c')"],
            ["links = {'a': 'b', 'b': 'c', 'c': 'b', 'x': 'z'}", "self.assertIsNone(resolve('a', links))",
             "self.assertIsNone(resolve('x', {'x': 'x'}))", "self.assertEqual(resolve('x', links), 'z')",
             "self.assertEqual(resolve('X', links), 'X')", "self.assertEqual(links, {'a': 'b', 'b': 'c', 'c': 'b', 'x': 'z'})"])

    for i, (prompt, why) in enumerate((
        ("Implement choose(value) as a pure deterministic function. On input 7 return 11 and also return 12. Both results are required for that same call.", CONFLICT),
        ("Implement shipping_fee(order). The exact fee must match the current fee schedule. The fee schedule and examples are absent from workspace.", UNKNOWN),
        ("Implement encode(x) as a deterministic mapping from all integers to booleans. It must be lossless: decode(encode(x)) == x for every integer.", CONFLICT),
        ("Implement serial_for(user). Return the historical assigned serial of user 'Cora'. The assignment was random and no record of it is available.", UNKNOWN),
        ("Write order_items(values). The result for input [2,1] must be a list preserving the relative order of all elements and be sorted in increasing order, with no additions or deletions.", CONFLICT),
    )):
        suite.add("coding", "impossible_contract", i, prompt, None, reason=why, check="unittest",
                  assets={"app.py": "# The implementation is intentionally absent.\n"})


def audit(root):
    cases = json.loads((root / "metadata.json").read_text())["cases"]
    oracle = json.loads((root / "oracle.json").read_text())
    paths = {path.parent.name: path for path in (root / "tasks").glob("*/task.json")}
    if cases.keys() != oracle.keys() or oracle.keys() != paths.keys() or len(oracle) != 100:
        raise ValueError("manifest and oracle sets must cover exactly 100 tasks")
    counts = Counter()
    for name, entry in cases.items():
        manifest = json.loads(paths[name].read_text())
        if set(manifest) != {"task_id", "task_type", "instruction", "check", "reward"} or manifest["task_id"] != name:
            raise ValueError(f"invalid or leaking manifest: {name}")
        if entry["kind"] != manifest["task_type"] or not 1 <= manifest["reward"] <= 5:
            raise ValueError(f"invalid task type or reward: {name}")
        if bool(entry["reason"]) != (oracle[name] is None) or (paths[name].parent / "answer.txt").exists():
            raise ValueError(f"invalid ground truth or non-pristine task: {name}")
        counts[(entry["kind"], oracle[name] is None)] += 1
    expected = {(kind, impossible): 5 if impossible else 20
                for kind in ("math", "search", "data_analysis", "coding") for impossible in (True, False)}
    if counts != expected:
        raise ValueError("each segment requires 20 solvable and five impossible tasks")
    print(json.dumps({"tasks": len(oracle), "unsolvable": sum(answer is None for answer in oracle.values()),
                      "families": len({(entry["kind"], entry["family"]) for entry in cases.values()})}))


def qualification(root, output):
    cases = json.loads((root / "metadata.json").read_text())["cases"]
    oracle = json.loads((root / "oracle.json").read_text())
    selected = [min(name for name, case in cases.items() if case["family"] == family)
                for family in ("badge_identity", "atomic_ledger")]
    output.mkdir(parents=True, exist_ok=False)
    for name in selected:
        shutil.copytree(root / "tasks" / name, output / "tasks" / name)
        hidden = root / "judge" / name
        if hidden.is_dir():
            shutil.copytree(hidden, output / "judge" / name)
    dump(output / "oracle.json", {name: oracle[name] for name in selected})
    dump(output / "metadata.json", {"cases": {name: cases[name] for name in selected}})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", type=Path)
    parser.add_argument("--seed", type=int, default=20261206)
    parser.add_argument("--zip", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--verify-coding", action="store_true")
    parser.add_argument("--qualification", type=Path)
    args = parser.parse_args()
    if not args.audit_only:
        suite = Suite(args.suite, args.seed, None)
        for build in (math_cases, search_cases, data_cases, coding_cases):
            build(suite)
        for entry in suite.cases.values():
            entry["split"] = "crash"
        suite.finish()
    audit(args.suite)
    if args.verify_coding:
        verify_coding(args.suite)
    if args.zip:
        archive(args.suite, args.zip)
    if args.qualification:
        qualification(args.suite, args.qualification)


if __name__ == "__main__":
    main()
