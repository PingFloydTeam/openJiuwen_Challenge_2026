"""Seeded judge-shaped corpus. Labels and private tests live outside tasks/."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import dataclass, field
import hashlib
import io
import json
from math import prod
from pathlib import Path
import random
import shutil
import sqlite3
import zipfile

SEED = 20261109
REPEATS = 5
CAUSES = {"undefined_math", "missing_information", "logical_error"}
JURY_TOOLS = {
    "math": "Compute it with the bash tool (python3 -c).",
    "data_analysis": "Use bash with python3 to compute it.",
    "search": "Use the file tools or bash (ls, grep, cat) to find it.",
}
JURY_OUTPUT = ("Work only inside the current directory. When you are done, write ONLY the final "
               "answer to the file answer.txt in the current directory (for example with bash: echo 42 > answer.txt).")


@dataclass
class Case:
    family: str
    prompt: str
    answer: str | int | None
    assets: dict[str, str] = field(default_factory=dict)
    check: str = "exact_int"
    reason: tuple[str, str] | None = None
    private: dict[str, str] = field(default_factory=dict)
    sql: tuple[str, list[tuple]] | None = None


def no(family, prompt, cause, detail, assets=None, check="exact_int"):
    return Case(family, prompt, None, assets or {}, check, (cause, detail))


def math_cases(r, i):
    initial = 200 + 300*i + r.randrange(1500) % 300
    day, sent, boxes = r.randrange(2, 7), r.randrange(3, 21), r.randrange(5, 35)
    m, n, k = r.randrange(2, 7), r.randrange(2, 9), r.randrange(2, 12)
    bound = m + 10*i
    area = f"A rectangle is {bound} m long and {n} m wide. What is its area in square meters?"
    driver = ["Валера", "Миша", "Ира", "Антон", "Лена"][i]
    bus = f"Ночью в дождь едет автобус, водителя зовут {driver}."
    quantity = ["килограмм", "2 килограмма", "3 килограмма", "4 килограмма", "5 килограммов"][i]
    goods = ["гвоздей", "ваты", "муки", "песка", "перьев"][i]
    yield Case("stock_ship", f"A warehouse starts with {initial} boxes. It ships {sent} boxes per day for {day} days. It receives {m} deliveries of {boxes} boxes each. How many boxes does it have at the end?", initial - sent * day + m * boxes)
    yield Case("stock_split", f"A warehouse starts with {initial} boxes. Then the stock is split equally into {m} warehouses and it keeps one share (rounded down). Then the stock is doubled by merging with a sister warehouse. Then the stock is split equally into {n} warehouses and it keeps one share (rounded down). How many boxes does it have at the end?", (initial // m * 2) // n)
    yield Case("expression", f"Calculate ({initial} - {sent} * {day}) // {n} + {k}.", (initial - sent * day) // n + k)
    yield Case("area", area, bound * n)
    yield Case("one_integer", f"An integer must be greater than {bound} and less than {bound+2}. What is the integer?", bound + 1)
    yield Case("discount", f"A ticket costs {100*bound} rubles. The price drops by {k}% and then rises by {n} rubles. Give the final price in rubles.", bound * (100-k) + n)
    yield Case("rate", f"Two cyclists start together in opposite directions at {m+10*i} and {n} km/h. How far apart are they after {day} hours? Give kilometers.", (m+10*i+n) * day)
    yield Case("balance", f"A jar holds {initial} beads. Split them into bags of {n}, using only full bags. How many beads remain outside the bags?", initial % n)
    yield Case("average", f"The three measured values are {initial-k}, {initial}, and {initial+k}. Find their arithmetic mean.", initial)
    yield Case("weight_riddle", f"Летели ёжики на юг со скоростью {15+i} км/ч, водителя автобуса зовут {driver}. Сколько весит {quantity} {goods}?", f"{i+1} кг", check="exact_str")
    yield Case("rain_riddle", bus + " Каким был его открытый наружный номер — сухим или мокрым?", "мокрым", check="exact_str")
    yield Case("irrelevant_speed", f"Едет автобус со скоростью {20+bound} км/ч. В коробке {n} конфет. Сколько конфет в двух таких полных коробках?", 2*n)
    expr = [f"{initial} / 0", f"{initial} % 0", f"{initial} // ({m} - {m})",
            f"{initial} / ({m} - {m})", "0 / 0"][i]
    yield no("undefined", f"Evaluate exactly {expr}.", "undefined_math", "denominator or modulus is zero")
    prompts = [
        f"A rectangle is {bound} m long. Its width is not given. What is its area in square meters?",
        f"A rectangle is {bound} m long. Its width is not given. What is its area in square meters?",
        f"Two bags hold {initial} marbles in total. How many marbles are in the first bag?",
        bus + " Какие цифры были на его номере?",
        "Ночью в дождь едет автобус. Водителя зовут Валера. Какие цифры были на его номере?",
    ]
    yield no("missing", prompts[i], "missing_information",
             ["width unspecified", "width unspecified", "total does not determine the split",
              "driver and weather do not identify registration digits", "registration digits unspecified"][i],
             check="exact_str" if i >= 3 else "exact_int")
    yield no("inconsistent_bounds",
             f"An integer must be greater than {bound+2} and less than {bound+(i%3)}. What is the integer?",
             "logical_error", "no integer can satisfy both bounds")


def markdown(title, fields):
    return "# " + title + "\n\n" + "\n".join(f"{key}: {value}" for key, value in fields.items()) + "\n"


def search_cases(r, i):
    person = ["Ada Chen", "Rita Okafor", "Sam Lee", "Omar Rossi", "Mina Shah"][i]
    mentor = ["Leo Hall", "Ren Ito", "Noor Ahmed", "Tom Green", "Eva Li"][i]
    firms = ["Cedar Lab", "Indigo Lab", "Stone Lab", "Polar Lab", "Juniper Lab"]
    employer, mentor_company = firms[i], firms[(i+1) % len(firms)]
    city, mentor_city = r.sample(["Kyoto", "Oslo", "Minsk", "Tbilisi", "Tomsk", "Riga", "Quito"], 2)
    born, mentor_born, founded = r.randrange(1985, 2005), r.randrange(1950, 1980), r.randrange(1980, 2018)
    code = f"R{r.randrange(200, 999)}K"
    birth_date = f"{born}-04-{12+i}"
    person_fields = {"Born": born, "Birth date": birth_date, "Mentor": mentor, "Employer": employer}
    mentor_fields = {"Born": mentor_born, "Employer": mentor_company}
    data = {"kb/person.md": markdown(person, person_fields),
            "kb/mentor.md": markdown(mentor, mentor_fields),
            "kb/employer.md": markdown(employer, {"Headquarters": city, "Founded": founded}),
            "kb/mentor_employer.md": markdown(mentor_company, {"Headquarters": mentor_city}),
            "docs/notice.txt": f"Badge color: green\nService code: {code}\n"}
    for n in range(r.randrange(8, 23)):
        data[f"kb/note_{n:02}.md"] = markdown(f"{person} {n+1}", {
            "Born": r.randrange(1950, 2010), "Employer": firms[(i+n+2) % len(firms)]})
    prefix = "The directory kb/ contains a small knowledge base of markdown files. "
    prompts = [
        ("born", f"In which year was {person} born?", born),
        ("founded", f"When was {employer} founded?", founded),
        ("employer", f"Who employs {person}?", employer),
        ("mentor", f"Who mentors {person}?", mentor),
        ("employer_city", f"In which city is the headquarters of {person}'s employer?", city),
        ("mentor_birth", f"In which year was {person}’s mentor born?", mentor_born),
        ("mentor_city", f"In which city is the headquarters of {person}'s mentor's employer?", mentor_city),
        ("employer_founded", f"When was {person}'s employer founded?", founded),
        ("birth_date", f"What is the birth date of {person}?", birth_date),
        ("paraphrase", f"Trace {person} to their employer and report the employer's headquarters city.", city),
        ("consistent_duplicate", f"In which year was {person} born?", born),
    ]
    for family, prompt, answer in prompts:
        assets = data
        if family == "consistent_duplicate":
            assets = data | {"kb/copy.md": markdown(person, {"Born": born})}
        yield Case(family, prefix + prompt, str(answer), assets, "exact_str")
    yield Case("notice", "What is the service code recorded in docs/notice.txt?", code, data, "exact_str")
    yield no("missing_field", prefix + f"What is the birth date of {person}?",
             "missing_information", "record has birth year but no day or month",
             data | {"kb/person.md": markdown(person, {k: v for k, v in person_fields.items() if k != "Birth date"})},
             "exact_str")
    yield no("missing_link", prefix + f"In which city is the headquarters of {person}'s employer?",
             "missing_information", "employer's record is absent",
             {k: v for k, v in data.items() if k != "kb/employer.md"}, "exact_str")
    yield no("conflicting_fact", prefix + f"In which year was {person} born?",
             "logical_error", "two records assign different years to the same person",
             data | {"kb/copy.md": markdown(person, {"Born": born+2})}, "exact_str")


def table_text(columns, rows, delimiter=","):
    sink = io.StringIO(newline="")
    writer = csv.writer(sink, delimiter=delimiter, lineterminator="\n")
    writer.writerow(columns)
    writer.writerows(rows)
    return sink.getvalue()


def data_cases(r, i):
    columns = ("order_id", "region", "product", "month", "units", "price")
    regions = ["north", "south", "west"]
    products = ["widget", "bolt", "gear"]
    rows = [(j+1, regions[j%3], r.choice(products), r.randrange(1, 13),
             r.randrange(0, 21), r.randrange(2, 21)) for j in range([20, 35, 70, 160, 300][i])]
    region = regions[i%3]
    selected = [row for row in rows if row[1] == region]
    assets = {"sales.csv": table_text(columns, rows),
              "sales.tsv": table_text(columns, rows, "\t"),
              "fixed.csv": table_text(("category", "units"), [("a", 8+i), ("a", 12+i), ("b", 90)])}
    pre = "Using only sales.csv, "
    yield Case("sum_units", pre + f"what is the sum of units where region = '{region}'?", sum(row[4] for row in selected), assets)
    yield Case("count_rows", pre + f"how many rows match region = '{region}' and units >= {i+3}?", sum(row[4] >= i+3 for row in selected), assets)
    yield Case("minimum", pre + f"what is the minimum of price where region = '{region}'?", min(row[5] for row in selected), assets)
    yield Case("maximum", pre + f"what is the maximum of units where region = '{region}'?", max(row[4] for row in selected), assets)
    yield Case("average", "Using only fixed.csv, what is the average of units where category = 'a'?", 10+i, assets)
    yield Case("distinct", pre + f"what is the number of distinct values of product where region = '{region}'?", len({row[2] for row in selected}), assets)
    row = r.choice(rows)
    yield Case("unique", pre + f"what is the unique value of product where order_id = {row[0]}?", row[2], assets, "exact_str")
    yield Case("zero_sum", pre + "what is the sum of units where region = 'east'?", 0, assets)
    yield Case("tsv", f"Using only sales.tsv, what is the sum of units where region = '{region}'?", sum(row[4] for row in selected), assets)
    sql_rows = [(f"p{j}", r.randrange(0, 9)) for j in range(15+i)]
    yield Case("sqlite", f"Using only inventory.sqlite, table stock, how many rows match amount < {i+2}?",
               sum(amount < i+2 for _, amount in sql_rows),
               sql=("CREATE TABLE stock (sku TEXT, amount INTEGER)", sql_rows))
    yield Case("revenue", f"From sales.csv, return total revenue for {region}: sum units times price per matching row.", sum(row[4]*row[5] for row in selected), assets)
    start = 2+i
    yield Case("month_window", f"The file sales.csv has columns order_id,region,product,month,units,price ({len(rows)} rows). How many units of product 'widget' were sold in months {start} through {start+1} inclusive?", sum(row[4] for row in rows if row[2] == "widget" and start <= row[3] <= start+1), assets)
    yield no("missing_column", pre + f"what is the sum of {['discount', 'tax', 'returned', 'shipping', 'margin'][i]}?",
             "missing_information", "requested column does not exist", assets)
    yield no("empty_average", pre + "what is the average of units where region = 'east'?",
             "undefined_math", "there are no rows for east, so there is no mean", assets)
    conflict = assets | {"status.csv": table_text(("sku", "city"), [(f"P{i}", "Riga"), (f"P{i}", "Oslo")])}
    yield no("conflicting_rows", f"Using only status.csv, what is the unique value of city where sku = 'P{i}'?",
             "logical_error", "matching rows have two distinct cities and no precedence", conflict, "exact_str")


def py_tests(module, function, args_list, oracle):
    assertions = "\n".join(f"        self.assertEqual(lib.{function}(*{args!r}), {oracle(*args)!r})" for args in args_list)
    return (f"import unittest\nimport {module} as lib\n\nclass Checks(unittest.TestCase):\n"
            f"    def test_{function}(self):\n{assertions}\n\n"
            "if __name__ == '__main__':\n    unittest.main()\n")


def coding_cases(r, i):
    n = r.randrange(2, 8)
    unique_number = n + 10*i
    words = " ".join(r.sample(["apricot", "pear", "lime", "fig", "melon"], 4))
    sequence = [n, n-1, -n, n+1]
    accumulator = "    acc = {init}\n    for x in xs:\n        acc {op}= x\n    return acc\n"
    extrema = "    best = xs[0]\n    for x in xs:\n        if x {op} best:\n            best = x\n    return best\n"
    cases = [
        ("total", "xs", accumulator.format(init=0, op="-"), accumulator.format(init=0, op="+"),
         sum, [(sequence,), ([-1, -4],), ([],)], [([n, n, 0],), ([0],)]),
        ("product", "xs", accumulator.format(init=1, op="+"), accumulator.format(init=1, op="*"),
         prod, [(sequence,), ([-1, -4],), ([],)], [([n, 0],), ([n],)]),
        ("maximum", "xs", extrema.format(op="<"), extrema.format(op=">"),
         max, [(sequence,), ([-1, -4],), ([1],)], [([n, n, 0],), ([-n, -n-1],)]),
        ("minimum", "xs", extrema.format(op=">"), extrema.format(op="<"),
         min, [(sequence,), ([-1, -4],), ([1],)], [([n, n, 0],), ([-n, -n-1],)]),
        ("count_even", "xs", "return sum(1 for x in xs if x % 2 == 1)", "return sum(1 for x in xs if x % 2 == 0)",
         lambda xs: len([x for x in xs if x % 2 == 0]), [([n*2, n*2+2, 1],), ([],), ([-2, 0, -3],)], [([-n*2, 1],), ([0, 0, 1, 3, 5],)]),
        ("count_odd", "xs", "return sum(1 for x in xs if x % 2 == 0)", "return sum(1 for x in xs if x % 2 == 1)",
         lambda xs: len([x for x in xs if x % 2 != 0]), [([n*2, n*2+2, 1],), ([],), ([-1, 0, -3],)], [([-n*2-1, 2],), ([0, 0, 1, 3, 5],)]),
        ("reverse_words", "s", "return ' '.join(sorted(s.split()))", "return ' '.join(reversed(s.split()))",
         lambda s: " ".join(s.split()[::-1]), [(words,), ("first  middle last",), ("",)], [("  Z  a  Z ",), ("last FIRST",)]),
        ("average", "xs", "return sum(xs) // len(xs)", "return sum(xs) / len(xs)",
         lambda xs: sum(xs) / len(xs), [([n, n+1],), ([1],), ([-1, -4],)], [([n, 0, n+1],), ([0, -1],)]),
        ("first_n", "xs, n", "return xs[:n + 1]", "return xs[:n]",
         lambda xs, count: xs[:count], [(sequence, 0), (sequence, 2), (sequence, 10)], [([n, 0], 1), ([], 0)]),
        ("clamp", "x, lo, hi", "return min(lo, max(hi, x))", "return min(hi, max(lo, x))",
         lambda x, lo, hi: lo if x < lo else hi if x > hi else x,
         [(n+5, 1, n+3), (-3, 1, n+3), (n, 1, n+3)], [(n, n, n), (0, -n, n)]),
        ("rotate_left", "xs, n", "return xs[-n:] + xs[:-n]",
         "if not xs:\n    return []\nn %= len(xs)\nreturn xs[n:] + xs[:n]",
         lambda xs, shift: xs[shift % len(xs):] + xs[:shift % len(xs)] if xs else [],
         [(sequence, 1), (sequence, 0), (sequence, 7)], [([], 3), (sequence, -1)]),
        ("unique_in_order", "xs", "return sorted(set(xs))", "return list(dict.fromkeys(xs))",
         lambda xs: list(dict.fromkeys(xs)), [([n, 1, n, 2],), ([],), ([2, 1, 2],)], [([3, 0, -1, 3],), ([n]*4,)]),
    ]
    module = ["mylib", "inventory", "logic", "tiny_lib", "helpers"][i]
    test_file = f"test_{module}.py"
    for function, args, before, after, oracle, public, hidden in cases:
        def source(body):
            if not body.startswith("    "):
                body = "\n".join("    " + line for line in body.splitlines()) + "\n"
            return f"def {function}({args}):\n{body}\n\ndef identity(value):\n    return value\n"

        prompt = (f"Run `python3 {test_file}`. Fix the bugs in {module}.py until the tests pass. "
                  "Preserve all tests and unrelated functions. Submit DONE after fixing the source.")
        tests = py_tests(module, function, public, oracle)
        yield Case(function, prompt, "tests_pass", {f"{module}.py": source(before), test_file: tests}, "unittest",
                   private={"reference.py": source(after), "test_private.py": py_tests(module, function, hidden, oracle)})
    key = ["code", "pin", "secret", "identifier", "key"][i]
    yield no("missing_spec",
             f"Implement read_code() to return the exact code recorded in {key}.txt. That file is the only source of truth.",
             "missing_information", "the only source-of-truth file is missing",
             {"mylib.py": "def read_code():\n    raise NotImplementedError\n"}, "unittest")
    yield no("contradictory_contract",
             f"Implement score() as a deterministic function. The same call must return {unique_number} and must return {unique_number+2}. Both requirements are mandatory.",
             "logical_error", "a deterministic call cannot return two distinct values",
             {"mylib.py": "def score():\n    return 0\n"}, "unittest")
    yield no("unspecified_rule",
             f"Implement fees(x) for the settlement system. For input {unique_number}, return the result required by the settlement policy. The policy and examples are unavailable.",
             "missing_information", "neither rule nor input/output examples determine the function",
             {"mylib.py": "def fees(x):\n    raise NotImplementedError\n"}, "unittest")


GENERATORS = {"math": math_cases, "search": search_cases, "data_analysis": data_cases, "coding": coding_cases}
HOLDOUT = {
    "math": {"discount", "rate", "irrelevant_speed", "missing"},
    "search": {"paraphrase", "consistent_duplicate", "mentor_city", "missing_link"},
    "data_analysis": {"sqlite", "revenue", "zero_sum", "empty_average"},
    "coding": {"rotate_left", "unique_in_order", "product", "unspecified_rule"},
}


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def generate(root: Path, seed=SEED):
    root.mkdir(parents=True, exist_ok=False)
    rng = random.Random(seed)
    cases = []
    for kind, gen in GENERATORS.items():
        for i in range(REPEATS):
            for case in gen(rng, i):
                if kind in JURY_TOOLS and i < 3 and case.prompt.isascii() and case.prompt.endswith("?"):
                    case.prompt += f" {JURY_TOOLS[kind]} {JURY_OUTPUT}"
                cases.append((kind, case))
    rng.shuffle(cases)
    oracle, entries = {}, {}
    for number, (kind, case) in enumerate(cases):
        name = f"task_{number:04d}"
        directory = root / "tasks" / name
        directory.mkdir(parents=True)
        dump(directory / "task.json", {
            "task_id": name, "task_type": kind, "reward": rng.randint(1, 5),
            "check": case.check, "instruction": case.prompt,
        })
        for file, content in case.assets.items():
            path = directory / "workspace" / file
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        if case.sql:
            statement, values = case.sql
            path = directory / "workspace/inventory.sqlite"
            path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(path) as conn:
                conn.execute(statement)
                conn.executemany("INSERT INTO stock VALUES (?, ?)", values)
            conn.close()
        if case.private:
            private = root / "judge" / name
            private.mkdir(parents=True)
            for file, content in case.private.items():
                (private / file).write_text(content, encoding="utf-8")
        oracle[name] = case.answer
        entries[name] = {"kind": kind, "family": case.family,
                         "split": "holdout" if case.family in HOLDOUT[kind] else "dev",
                         "reason": None if case.reason is None else dict(zip(("category", "detail"), case.reason)),
                         "public_tests": {
                             file: hashlib.sha256(content.encode()).hexdigest()
                             for file, content in case.assets.items() if kind == "coding" and file.startswith("test_")
                         },
                         "source": next((name for name in case.assets if name.endswith(".py") and not name.startswith("test_")), None)
                         if kind == "coding" else None}
    dump(root / "oracle.json", oracle)
    dump(root / "metadata.json", {"seed": seed, "cases": entries})


def audit(root):
    oracle = json.loads((root / "oracle.json").read_text())
    metadata = json.loads((root / "metadata.json").read_text())["cases"]
    totals, reasons, splits, families = Counter(), Counter(), Counter(), Counter()
    prompts = set()
    fingerprints = set()
    assert set(oracle) == set(metadata) == {p.parent.name for p in (root / "tasks").glob("*/task.json")}
    for name, entry in metadata.items():
        directory = root / "tasks" / name
        manifest = json.loads((directory / "task.json").read_text())
        assert set(manifest) == {"task_id", "task_type", "reward", "check", "instruction"}
        assert manifest["task_type"] == entry["kind"] and manifest["task_id"] == name
        assert 1 <= manifest["reward"] <= 5
        assert not (directory / "answer.txt").exists(), name
        reason = entry["reason"]
        assert (oracle[name] is None) == (reason is not None)
        if reason:
            assert reason["category"] in CAUSES and reason["detail"]
            reasons[reason["category"]] += 1
        for file, digest in entry["public_tests"].items():
            assert hashlib.sha256((directory / "workspace" / file).read_bytes()).hexdigest() == digest
        if manifest["check"] == "unittest" and not reason:
            assert (root / "judge" / name / "test_private.py").is_file()
            assert (root / "judge" / name / "reference.py").is_file()
        totals[entry["kind"], oracle[name] is None] += 1
        splits[entry["split"]] += 1
        families[entry["kind"], entry["family"]] += 1
        prompts.add(manifest["instruction"])
        digest = hashlib.sha256(manifest["instruction"].encode())
        for file in sorted((directory / "workspace").rglob("*")):
            if file.is_file():
                digest.update(file.relative_to(directory).as_posix().encode() + b"\0" + file.read_bytes())
        fingerprint = digest.hexdigest()
        assert fingerprint not in fingerprints, f"duplicate task input: {name}"
        fingerprints.add(fingerprint)
    assert set(totals) == {(kind, label) for kind in GENERATORS for label in (True, False)}
    assert all(totals[kind, label] == (15 if label else 60) for kind, label in totals)
    assert len(families) == 60 and all(count == REPEATS for count in families.values())
    assert splits == {"dev": 220, "holdout": 80}
    print(json.dumps({"tasks": len(oracle), "unique_prompts": len(prompts), "families": len(families),
                      "splits": dict(splits), "unsolvable": sum(reasons.values()), "causes": dict(reasons)}))


def verify_coding(root):
    from bench.suite import private_coding_passes
    from ping_floyd.coding import test_run

    entries = json.loads((root / "metadata.json").read_text())["cases"]
    repair_cases = [(name, entry) for name, entry in entries.items()
                    if entry["kind"] == "coding" and entry["reason"] is None]

    def check(item):
        name, entry = item
        workspace = root / "tasks" / name / "workspace"
        broken = test_run(workspace)
        return (name, not broken["passed"] and not broken["isolation_error"]
                and private_coding_passes(root, name, entry, reference=True))

    with ThreadPoolExecutor(max_workers=4) as pool:
        failed = [name for name, ok in pool.map(check, repair_cases) if not ok]
    if failed:
        raise ValueError(f"coding reference or bug verification failed: {failed}")
    print(json.dumps({"coding_references_verified": len(repair_cases)}))


def sample_100(source, destination, seed):
    """Create a contest-sized batch with 25 tasks of each type and 20% impossible."""
    if any((source / "tasks").glob("*/answer.txt")):
        raise ValueError("sample from a clean suite before running the agent")
    metadata = json.loads((source / "metadata.json").read_text())["cases"]
    oracle = json.loads((source / "oracle.json").read_text())
    rng = random.Random(seed)
    selected = []
    for kind in GENERATORS:
        for impossible, count in ((False, 20), (True, 5)):
            group = sorted(name for name, entry in metadata.items()
                           if entry["kind"] == kind and (oracle[name] is None) == impossible)
            selected.extend(rng.sample(group, count))
    destination.mkdir(parents=True, exist_ok=False)
    for name in sorted(selected):
        shutil.copytree(source / "tasks" / name, destination / "tasks" / name,
                        ignore=shutil.ignore_patterns("answer.txt", "__pycache__"))
        if (source / "judge" / name).is_dir():
            shutil.copytree(source / "judge" / name, destination / "judge" / name)
    dump(destination / "oracle.json", {name: oracle[name] for name in sorted(selected)})
    dump(destination / "metadata.json",
         {"seed": seed, "cases": {name: metadata[name] for name in sorted(selected)}})
    print(json.dumps({"tasks": len(selected), "unsolvable": sum(oracle[name] is None for name in selected)}))


def archive(root, output):
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as pack:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                info = zipfile.ZipInfo(path.relative_to(root).as_posix(), (2026, 11, 9, 0, 0, 0))
                pack.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", type=Path)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--zip", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--verify-coding", action="store_true", help="run reference and broken sources in Linux")
    parser.add_argument("--sample-from", type=Path, help="take a balanced 100-task batch from a clean suite")
    args = parser.parse_args()
    if args.sample_from:
        if args.audit_only:
            parser.error("--sample-from creates a new suite")
        sample_100(args.sample_from, args.suite, args.seed)
    else:
        if not args.audit_only:
            generate(args.suite, args.seed)
        audit(args.suite)
    if args.verify_coding:
        verify_coding(args.suite)
    if args.zip:
        archive(args.suite, args.zip)


if __name__ == "__main__":
    main()
