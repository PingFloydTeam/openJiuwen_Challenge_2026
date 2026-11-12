"""Stress a frozen release with paraphrases, contrasts and independent code tests."""
import argparse
from collections import Counter
from itertools import combinations, permutations
import json
from math import prod
from pathlib import Path
import random
import shutil
import tempfile

from ping_floyd.coding import test_run

from .hard import Suite
from .large import markdown
from .transfer import REFERENCE


def stress(output):
    suite = Suite(output, 20261113, None)

    def triplet(kind, family, base, rewrite, contrast, answer, changed, *, assets=None, changed_assets=None,
                reason=None, check="exact_int"):
        for i, (label, prompt, expected) in enumerate(zip(("base", "paraphrase", "contrast"),
                (base, rewrite, contrast), (answer, answer, changed), strict=True)):
            suite.add(kind, family, i, prompt, expected, check=check,
                      assets=changed_assets if i == 2 and changed_assets is not None else assets,
                      reason=reason if expected is None else None)
            suite.cases[suite.case_id(kind, family, i)]["split"] = label

    missing = ("missing_information", "the requested value cannot be uniquely inferred")
    triplet("math", "area", "A rectangle is 13 m long and 7 m wide. What is its area in square meters?",
            "A rectangular floor measures 13 metres along one side and 7 along the other. How many square metres does it cover?",
            "A rectangle is 13 m long. Its width is not given. What is its area in square meters?", 91, None, reason=missing)
    triplet("math", "integer", "An integer must be greater than 4 and less than 6. What is the integer?",
            "Find the whole number strictly inside the interval with endpoints 4 and 6.",
            "An integer must be greater than 4 and less than 7. What is the integer?", 5, None, reason=missing)
    triplet("math", "direction", "Two cyclists start together in opposite directions at 11 and 17 km/h. "
            "How far apart are they after 3 hours? Give kilometers.",
            "Two cyclists leave the same point in opposite directions and ride at 11 and 17 km/h for three hours. "
            "How many kilometres separate them?",
            "Two cyclists start together in the same direction at 11 and 17 km/h. "
            "How far apart are they after 3 hours? Give kilometers.", 84, 18)

    files = {"kb/person.md": markdown("Rana Diaz", {"Born": 1997, "Employer": "Birch Studio"}),
             "kb/company.md": markdown("Birch Studio", {"Headquarters": "Riga"}),
             "kb/distractor.md": markdown("Other Studio", {"Headquarters": "Oslo"})}
    prompt = "According to kb/, In which city is the headquarters of Rana Diaz's employer?"
    triplet("search", "linked_company", prompt, "Read kb/ and identify the city that hosts the head office "
            "of the company that employs Rana Diaz.", prompt, "Riga", None, assets=files,
            changed_assets={name: data for name, data in files.items() if name != "kb/company.md"}, reason=missing, check="exact_str")
    prompt = "According to kb/, In which year was Rana Diaz born?"
    triplet("search", "contradictory_record", prompt, "Find Rana Diaz's year of birth in the local kb/ records.",
            prompt, "1997", None, assets=files, changed_assets=files | {"kb/copy.md": markdown("Rana Diaz", {"Born": 1998})},
            reason=("logical_error", "two records for the same heading disagree"), check="exact_str")
    prompt = "What is the service code recorded in docs/notice.txt?"
    triplet("search", "notice", prompt, "Find docs/notice.txt and give the value next to its Service code label.",
            prompt, "K24X", "H72Q", assets={"docs/notice.txt": "Service code: K24X\n"},
            changed_assets={"docs/notice.txt": "Service code: H72Q\n", "docs/archive.txt": "Service code: K24X\n"}, check="exact_str")

    sales = {"sales.csv": "region,month,units\nNorth,1,4\nNorth,2,6\nSouth,1,2\nSouth,2,3\n"}
    prompt = "Using only sales.csv, which region has the highest total units?"
    triplet("data_analysis", "top_group", prompt, "Group the entries in sales.csv by region. "
            "Give the single region with the greatest combined number of units.", prompt, "North", None,
            assets=sales, changed_assets={"sales.csv": sales["sales.csv"].replace("South,2,3", "South,2,8")},
            reason=missing, check="exact_str")
    triplet("data_analysis", "average", "Using only sales.csv, what is the average of units where month = 1?",
            "Take all rows in sales.csv with month 1. What is their mean number of units?",
            "Using only sales.csv, what is the average of units where month = 3?", 3, None,
            assets=sales, reason=("undefined_math", "an empty sample has no arithmetic mean"))
    triplet("data_analysis", "threshold", "Using only sales.csv, what is the sum of units where month >= 1?",
            "In sales.csv, add up units for records whose month is at least 1.",
            "Using only sales.csv, what is the sum of units where month > 1?", 15, 9, assets=sales)
    suite.finish()


def composed(output):
    suite = Suite(output, 20261115, None)
    depots = ("Two depots originally held 147 boxes in total. Depot A received 15 boxes and depot B shipped 12. "
              "After both movements A had twice as many boxes as B. ")
    question = "How many boxes did depot A have before the movements?"
    suite.add("math", "balance_with_redundancy", 0, depots + question, 85)
    suite.add("math", "balance_with_redundancy", 1, depots + "The final combined count was 152. " + question,
              None, reason=("logical_error", "the movements fix the final total at 150"))
    suite.add("search", "cross_source_alias", 0,
              "The badge roster in badges.tsv maps badge IDs to names. Find the person with badge A19, then find "
              "their employer in kb/. Using licenses.tsv, return that employer's active permit code for the North district.",
              "C7-418", check="exact_str", assets={
                  "badges.tsv": "id\tname\nA18\tMara Soto\nA19\tMarta Soto\n",
                  "kb/mara.md": markdown("Mara Soto", {"Employer": "Blue Works"}),
                  "kb/marta.md": markdown("Marta Soto", {"Employer": "Blu Works"}),
                  "licenses.tsv": "employer\tdistrict\tstatus\tcode\nBlue Works\tNorth\tactive\tA0-419\n"
                  "Blu Works\tNorth\texpired\tD6-109\nBlu Works\tSouth\tactive\tC7-118\nBlu Works\tNorth\tactive\tC7-418\n",
              })
    suite.add("search", "supersession_policy", 0,
              "Find the contact code for project Willow. Apply the document selection policy in policy.txt to the "
              "three notices under docs/. Return the code for the selected notice.", "W-602", check="exact_str", assets={
                  "policy.txt": "Only approved notices count. Use the most recent issue date among approved notices.\n",
                  "docs/a.txt": "Project: Willow\nIssued: 2025-04-02\nStatus: approved\nContact code: W-521\n",
                  "docs/b.txt": "Project: Willow\nIssued: 2025-04-09\nStatus: draft\nContact code: W-699\n",
                  "docs/c.txt": "Project: Willow\nIssued: 2025-04-05\nStatus: approved\nContact code: W-602\n",
              })
    journal = "order,account,revision,event,cents\nx,1,1,sale,1200\nx,1,2,sale,1700\ny,2,1,sale,600\n"
    journal += "z,1,1,refund,300\nw,3,1,sale,950\nv,4,1,sale,100\n"
    instruction = ("Compute the net revenue in cents for the North region from accounts.csv and journal.csv. "
                   "Join on account ID. Keep only the highest revision of each order ID and subtract refunds. "
                   "Every journal account has one region, and the region of an account missing from the registry "
                   "may be either North or South.")
    for i, accounts in enumerate(("1,North\n2,South\n3,North\n4,South\n", "1,North\n2,South\n3,North\n")):
        suite.add("data_analysis", "revisions_join", i, instruction, 2350 if not i else None,
                  assets={"accounts.csv": "account,region\n" + accounts, "journal.csv": journal},
                  reason=("missing_information", "unknown region contributes either 0 or 100") if i else None)

    def repair(family, instruction, modules, checks, hidden):
        public_name = "test_contract.py"
        refs = {file: after for file, (_, after) in modules.items()}
        public = "import unittest\nimport pipeline\n\nclass Contract(unittest.TestCase):\n    def test_output(self):\n"
        main = "\nif __name__ == '__main__':\n    unittest.main()\n"
        suite.add("coding", family, 0, instruction + f" Run `python3 {public_name}`. Return DONE.",
                  "tests_pass", check="unittest", assets={file: before for file, (before, _) in modules.items()}
                  | {public_name: public + checks + main}, references=refs, hidden=public + hidden + main)

    count = ("from labels import normalize\n\ndef accumulate(entries):\n    totals = {}\n"
             "    for raw, amount in entries:\n        key = normalize(raw)\n        if key:\n"
             "            totals[key] = totals.get(key, 0) + amount\n    return totals\n")
    summary = ("from counts import accumulate\n\ndef report(entries):\n    totals = accumulate(entries)\n"
               "    return sorted(totals.items(), key=lambda pair: (-pair[1], pair[0]))\n")
    repair("normalized_ledger", "Repair the three modules used by report(entries). Strip whitespace and casefold labels, "
           "ignore blank labels, sum signed amounts per label and return (label, total) pairs sorted by descending "
           "total and then ascending label. Keep labels with zero totals.", {
               "labels.py": ("def normalize(raw):\n    return raw.lower()\n", "def normalize(raw):\n    return raw.strip().casefold()\n"),
               "counts.py": (count.replace("totals.get(key, 0) + amount", "amount"), count),
               "pipeline.py": (summary.replace("-pair[1]", "pair[1]"), summary),
           }, "        self.assertEqual(pipeline.report([(' X ', 8), ('x', -3), (' y ', 3), (' ', 8)]), [('x', 5), ('y', 3)])\n",
           "        self.assertEqual(pipeline.report([('ß', 4), ('ss', -4), ('Z', 0)]), [('ss', 0), ('z', 0)])\n"
           "        self.assertEqual(pipeline.report([]), [])\n")
    peak = ("from windows import active\n\ndef peak(intervals):\n    moments = sorted({start for start, end in intervals if start < end})\n"
            "    if not moments:\n        return (0, None)\n    counts = [(sum(active(a, b, t) for a, b in intervals), t) for t in moments]\n"
            "    return max(counts, key=lambda pair: (pair[0], -pair[1]))\n")
    repair("interval_peak", "Repair peak(intervals). Intervals are half-open [start,end). Return (count, time) for the "
           "earliest instant at which the maximum number of intervals are active. For empty input or only empty intervals "
           "return (0, None).", {
               "windows.py": ("def active(a,b,t):\n    return a <= t <= b\n", "def active(a,b,t):\n    return a <= t < b\n"),
               "pipeline.py": (peak.replace("-pair[1]", "pair[1]"), peak),
           }, "        self.assertEqual(pipeline.peak([(0, 4), (1, 4), (4, 8), (5, 8)]), (2, 1))\n",
           "        self.assertEqual(pipeline.peak([(4, 4)]), (0, None))\n"
           "        self.assertEqual(pipeline.peak([(0, 2), (2, 4), (2, 3)]), (2, 2))\n")
    suite.finish()


def schedule_oracle(edges):
    nodes = sorted({node for edge in edges for node in edge})
    for order in permutations(nodes):
        rank = {node: index for index, node in enumerate(order)}
        if all(rank[left] < rank[right] for left, right in edges):
            return list(order)
    return []


def fresh_assertions(family):
    rng = random.Random(20261114)
    expressions = []
    for i in range(30):
        if family == "topological_order":
            edges = rng.sample(list(combinations("ABCDEF", 2)), i % 11)
            rng.shuffle(edges)
            if edges and i % 3 == 0:
                edges.append(edges[0][::-1])
            call, answer = f"schedule.schedule({edges!r})", schedule_oracle(edges)
        elif family == "decode":
            pairs = [(rng.choice(("a", 4, None)), rng.randrange(5)) for _ in range(i % 8)]
            call, answer = f"pipeline.decode({pairs!r})", [value for value, count in pairs for _ in range(count)]
        elif family == "ledger":
            entries = [(rng.choice("abcd"), rng.randrange(9), bool(rng.randrange(2))) for _ in range(i % 12)]
            call = f"pipeline.balances({entries!r})"
            answer = {key: sum((-1 if refund else 1) * value for k, value, refund in entries if k == key)
                      for key in dict.fromkeys(k for k, _, _ in entries)}
        elif family == "closed_intervals":
            items = [tuple(sorted(rng.choices(range(-5, 10), k=2))) for _ in range(i % 12)]
            call = f"pipeline.pairs({items!r})"
            answer = sum(not (a[1] < b[0] or b[1] < a[0]) for a, b in combinations(items, 2))
        elif family == "product":
            items = rng.choices(range(-4, 5), k=i % 9)
            call, answer = f"mylib.product({items!r})", prod(items)
        else:
            raise ValueError(f"unreviewed coding family: {family}")
        expressions.append(f"        self.assertEqual({call}, {answer!r})")
    module = {"topological_order": "schedule", "product": "mylib"}.get(family, "pipeline")
    return f"import unittest\nimport {module}\n\nclass Independent(unittest.TestCase):\n    def test_cases(self):\n" + "\n".join(expressions) + "\n"


def coding_audit(original, solved):
    cases = json.loads((original / "metadata.json").read_text())["cases"]
    summary = Counter()
    for name, entry in cases.items():
        task = solved / "tasks" / name
        if entry["kind"] != "coding" or entry["reason"] or not (task / "answer.txt").is_file():
            continue
        family = entry["family"]
        with tempfile.TemporaryDirectory() as directory:
            stage = Path(directory) / "workspace"
            shutil.copytree(task / "workspace", stage)
            (stage / "test_fresh.py").write_text(fresh_assertions(family))
            result = test_run(stage)
            if result["isolation_error"]:
                raise RuntimeError(result["output"])
            summary["delivered_total"] += 1
            summary["delivered_passed"] += result["passed"]
            if not result["passed"]:
                print(json.dumps({"task": name, "family": family, "failure": result["output"]}))
            if family == "topological_order":
                (stage / "schedule.py").write_text(REFERENCE.replace("heappop(available)", "available.pop()"))
                (stage / "test_fresh.py").unlink()
                shutil.copyfile(original / "judge" / name / "test_private.py", stage / "test_old_hidden.py")
                summary["wrong_mutants_tested"] += 1
                summary["wrong_mutants_passed_old"] += test_run(stage)["passed"]
                (stage / "test_fresh.py").write_text(fresh_assertions(family))
                summary["wrong_mutants_passed_fresh"] += test_run(stage)["passed"]
    print(json.dumps(dict(summary)))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("stress", "composed"):
        make = sub.add_parser(name)
        make.add_argument("output", type=Path)
    coding = sub.add_parser("coding")
    coding.add_argument("original", type=Path)
    coding.add_argument("solved", type=Path)
    args = parser.parse_args()
    if args.command == "stress":
        stress(args.output)
    elif args.command == "composed":
        composed(args.output)
    else:
        coding_audit(args.original, args.solved)


if __name__ == "__main__":
    main()
