"""Independent regression cases for ambiguous facts, data integrity and AST edits."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3

from .large import archive, dump


def generate(root):
    root.mkdir(parents=True, exist_ok=False)
    oracle, metadata = {}, {}

    def add(name, kind, instruction, answer, assets=None, check="exact_int", reason=None,
            private=None, source=None):
        directory = root / "tasks" / name
        directory.mkdir(parents=True)
        assets = assets or {}
        dump(directory / "task.json", {"task_id": name, "task_type": kind,
             "instruction": instruction, "reward": 1, "check": check})
        for filename, content in assets.items():
            path = directory / "workspace" / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content.encode("utf-8"))
        oracle[name] = answer
        metadata[name] = {"kind": kind, "family": name, "split": "edge",
                          "reason": {"category": reason} if reason else None,
                          "source": source,
                          "public_tests": {p: hashlib.sha256(s.encode()).hexdigest()
                                           for p, s in assets.items() if p.startswith("test_")}}
        if private:
            target = root / "judge" / name
            target.mkdir(parents=True)
            for filename, content in private.items():
                (target / filename).write_text(content, encoding="utf-8")
        return directory

    add("unicode-arithmetic", "math", "Compute (9 − 3) × 4?", 24)
    add("lowercase-decimal", "math", "evaluate exactly (2.5 + 0.5) × 8.", 24)
    add("unicode-zero", "math", "Compute 9 ÷ (3 − 3).", None, reason="undefined_math")
    add("impossible-inventory", "math",
        "A warehouse starts with 2 boxes. It ships 2 boxes per day for 2 days. It receives 4 deliveries of 1 boxes each. How many boxes does it have at the end?",
        None, reason="logical_error")
    add("zero-inventory", "math",
        "A warehouse starts with 4 boxes. It ships 2 boxes per day for 2 days. How many boxes does it have at the end?", 0)
    add("unrecognized-word-problem", "math",
        "A child eats one apple out of a basket of five. How many apples are left in the basket?", 4)

    add("bullet-records", "search", "According to kb/, where is Imani Rao's employer headquartered?",
        "Lyon", {"kb/imani.md": "# Imani Rao\n\n- employer: Loom Studio\n",
                 "kb/loom.md": "# Loom Studio\n\n* HEADQUARTERS: Lyon\n"}, "exact_str")
    add("casefold-conflict", "search",
        "The directory kb/ contains a small knowledge base of markdown files. In which year was Jo Lim born?",
        None, {"kb/jo.md": "# Jo Lim\nBorn: 1990\nborn: 1995\n"}, "exact_int", "logical_error")
    add("prose-notice", "search", "What is the service code recorded in notices/duty.txt?", "Q72",
        {"notices/duty.txt": "For today's duty shift, the service code is Q72.\n"}, "exact_str")
    add("bullet-notice", "search", "What is the service code recorded in notices/duty.txt?", "Q73",
        {"notices/duty.txt": "- Shift: evening\n- SERVICE CODE: Q73\n"}, "exact_str")
    add("not-determined", "search", "According to kb/, what year was Sia Ye born?", None,
        {"kb/sia.md": "# Sia Ye\nEmployer: Loom Studio\nBorn: \n"}, "exact_int", "missing_information")

    report = {"shipments.csv": "region,units,month\nNorth,7,1\nNorth,6,2\nSouth,11,2\nWest,2,2\n"}
    add("highest-group", "data_analysis", "In shipments.csv, which region has the highest total units?",
        "North", report, "exact_str")
    add("filtered-group", "data_analysis",
        "Using only shipments.csv, which region has the lowest sum of units where month is at least 2?",
        "West", report, "exact_str")
    add("tied-groups", "data_analysis", "In shipments.csv, which region has the largest total units?",
        None, {"shipments.csv": "region,units\nNorth,7\nSouth,5\nSouth,2\n"}, "exact_str", "missing_information")
    add("quoted-and", "data_analysis",
        'In shipments.csv, how many rows match region="South and units >= 5" and units>=3?',
        1, {"shipments.csv": "region,units\nSouth and units >= 5,4\nSouth,9\nSouth and units >= 5,1\n"})
    add("blank-selected-cell", "data_analysis",
        "Using only ledger.csv, what is the sum of amount where month = 2?", None,
        {"ledger.csv": "month,amount\n2,10\n2,\n3,42\n"}, reason="missing_information")
    add("blank-excluded-cell", "data_analysis",
        "Using only ledger.csv, what is the sum of amount where month = 2?", 10,
        {"ledger.csv": "month,amount\n2,10\n3,\n"})
    add("duplicate-header", "data_analysis", "Using only ledger.csv, what is the sum of amount?",
        None, {"ledger.csv": "id,amount,amount\n1,12,30\n"}, reason="logical_error")
    add("ragged-row", "data_analysis", "Using only ledger.csv, what is the sum of amount?",
        None, {"ledger.csv": "id,amount\n1,12\n2\n"}, reason="missing_information")
    add("empty-group", "data_analysis", "In shipments.csv, which region has the highest total units where month = 12?",
        None, report, "exact_str", "undefined_math")
    sql_cases = [
        ("forty-first-table", "Using only warehouse.sqlite, table STOCK, what is the sum of units?", 10),
        ("existing-view", "Using only warehouse.sqlite, table available, what is the sum of units?", 10),
    ]
    for name, instruction, value in sql_cases:
        folder = add(name, "data_analysis", instruction, value) / "workspace"
        folder.mkdir()
        with sqlite3.connect(folder / "warehouse.sqlite") as connection:
            for i in range(41):
                connection.execute(f"CREATE TABLE dummy{i} (value INTEGER)")
            connection.execute("CREATE TABLE stock (units INTEGER)")
            connection.executemany("INSERT INTO stock VALUES (?)", [(3,), (7,)])
            connection.execute("CREATE VIEW available AS SELECT units FROM stock")
        connection.close()

    def coding(name, broken, fixed, public, private):
        def tests(lines):
            return ("import unittest\n" + lines + "\n\nif __name__ == '__main__':\n    unittest.main()\n")
        add(name, "coding", "Run `python3 test_mylib.py`. Fix mylib.py and keep other functions and the tests intact.",
            "tests_pass", {"mylib.py": broken, "test_mylib.py": tests(public)}, "unittest",
            private={"reference.py": fixed, "test_private.py": tests(private)}, source="mylib.py")

    extrema = "def maximum(xs):\n    best = xs[0]\n    for x in xs:\n        if x < best:\n            best = x\n    return best\n"
    coding("direct-import-repair", extrema, extrema.replace("<", ">"),
           "from mylib import maximum as largest\nclass T(unittest.TestCase):\n"
           "    def test_max(self): self.assertEqual(largest([2, 4, 1]), 4)",
           "from mylib import maximum\nclass T(unittest.TestCase):\n"
           "    def test_private(self): self.assertEqual(maximum([-8, -3, -10]), -3)")
    original_total = "def total(xs):\n    s = 0\n    for x in xs:\n        s -= x\n    return s\n"
    unrelated = "\n" + original_total.replace("total", "subtract_all")
    coding("duplicate-edit-span", original_total + unrelated, original_total.replace("-=", "+=") + unrelated,
           "import mylib\nclass T(unittest.TestCase):\n"
           "    def test_total(self): self.assertEqual(mylib.total([3, 8]), 11)",
           "import mylib\nclass T(unittest.TestCase):\n"
           "    def test_private(self):\n        self.assertEqual(mylib.total([-2, 5]), 3)\n"
           "        self.assertEqual(mylib.subtract_all([-2, 5]), -3)")
    unicode_source = 'def word_count(s): "Décompte des mots"; return len(s)\r\n'
    coding("unicode-source-edit", unicode_source, unicode_source.replace("len(s)", "len(s.split())"),
           "import mylib\nclass T(unittest.TestCase):\n"
           "    def test_words(self): self.assertEqual(mylib.word_count('one two'), 2)",
           "import mylib\nclass T(unittest.TestCase):\n"
           "    def test_private(self):\n        self.assertEqual(mylib.word_count('  a  b   c  '), 3)\n"
           "        self.assertEqual(mylib.word_count('  '), 0)")

    dump(root / "oracle.json", oracle)
    dump(root / "metadata.json", {"cases": metadata})
    print(json.dumps({"cases": len(oracle), "unsolvable": sum(v is None for v in oracle.values()),
                      "by_type": dict(Counter(e["kind"] for e in metadata.values()))}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    parser.add_argument("--zip", type=Path)
    args = parser.parse_args()
    generate(args.destination)
    if args.zip:
        archive(args.destination, args.zip)


if __name__ == "__main__":
    main()
