"""English adversarial cases with independent answers and private coding tests."""
import argparse
from collections import Counter
from datetime import datetime
from fractions import Fraction
from itertools import combinations
import json
from pathlib import Path
import sqlite3

from .crash import CONFLICT, UNDEFINED, UNKNOWN, table
from .hard import Suite
from .large import archive, markdown, verify_coding

KINDS = ("math", "search", "data_analysis", "coding")


def math_cases(suite):
    for i in range(4):
        intervals = [(start, start + suite.rng.randrange(3, 12))
                     for start in suite.rng.sample(range(-8, 14), 8)]
        answer = Fraction(sum(sum(start <= tick < end for start, end in intervals) >= 2
                              for tick in range(-8, 25)), 4)
        spans = ", ".join(f"[{Fraction(start, 4)}, {Fraction(end, 4)})" for start, end in intervals)
        suite.add("math", "fractional_overlap", i,
                  f"During the following half-open intervals measured in hours, one pump per interval runs: {spans}. "
                  "How many hours have at least two pumps running? Return a reduced fraction or an integer.",
                  str(answer), check="exact_str")

        target = 5 + i
        result = (target + 3) // 2
        parity = "even" if target % 2 == 0 else "odd"
        suite.add("math", "inverse_rounding_with_parity", i,
                  f"A crate starts with an {parity} number of items. Add 3, divide by 2, then round down. "
                  f"The result is {result}. What was the original number?", target)

        start = f"2024-02-28T23:{30+i:02}:00-03:00"
        end = f"2024-02-29T04:{10+i*2:02}:00+00:00"
        minutes = int((datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds() // 60)
        suite.add("math", "offset_leap_midnight", i,
                  f"A job began at {start} and finished at {end}. What was its elapsed time in whole minutes?",
                  minutes)

        labels = ["red"] * (3 + i) + ["blue"] * (4 + i) + ["green"] * 2
        draws = list(combinations(labels, 3))
        probability = Fraction(sum(len(set(draw)) == 2 for draw in draws), len(draws))
        suite.add("math", "draw_exactly_two_colors", i,
                  f"A bag contains {3+i} red, {4+i} blue and 2 green tokens. Draw 3 without replacement, "
                  "uniformly. What is the probability of seeing exactly two different colors? "
                  "Give a reduced fraction.", str(probability), check="exact_str")

    impossible = (
        ("integral_ratio_impossible", "A 17-seat cabin is full. There are twice as many adults as children, "
         "and every seat contains one adult or one child. How many children are seated?", CONFLICT),
        ("rounded_divisor_zero", "Compute the remainder when 19 is divided by the integer obtained by "
         "rounding 2/5 down. Give the exact integer remainder.", UNDEFINED),
        ("missing_endpoint_inclusion", "Count integer points in the interval between -4 and 6. The notation "
         "does not specify whether either endpoint is included. Give the unique count.", UNKNOWN),
        ("missing_offset", "A flight departed at 23:50 local time on May 5 and arrived at 01:15 local time "
         "on May 6. Its arrival airport has UTC offset +02:00. The departure offset is unknown. "
         "Give the unique elapsed time in minutes.", UNKNOWN),
    )
    for i, (family, question, reason) in enumerate(impossible):
        suite.add("math", family, i, question, None, reason=reason)


def employer_assets(person, left, right):
    return {
        "kb/person-a.md": markdown(person, {"Employer": "Arbor Lab"}),
        "kb/person-b.md": markdown(person, {"Employer": "Beacon Lab"}),
        "kb/arbor.md": markdown("Arbor Lab", {"Headquarters": left}),
        "kb/beacon.md": markdown("Beacon Lab", {"Headquarters": right}),
        "format.txt": "Repeated person headings list exhaustive alternatives for the single employer.\n",
    }


def search_cases(suite):
    for i, city in enumerate(("Tallinn", "Accra", "Seoul", "Montevideo")):
        person = f"Riley Kent {i}"
        question = f"According to kb/, In which city is the headquarters of {person}'s employer?"
        suite.add("search", "convergent_employers", i, question, city, check="exact_str",
                  assets=employer_assets(person, city, city))

        entity = f"OP-{suite.rng.randrange(1000, 9999)}"
        year = 1971 + i
        suite.add("search", "case_distinguishes_entity_ids", i,
                  f"According to kb/, In which year was {entity} born?", year, assets={
                      "format.txt": "People are identified by case-sensitive heading IDs.\n",
                      "kb/exact.md": markdown(entity, {"Born": year}),
                      "kb/lower.md": markdown(entity.lower(), {"Born": 2003 + i}),
                  })

        prefix = f"permit-{i}"
        cutoff = "2025-03-18"
        question = (f"Which permit code for {prefix} was in effect on {cutoff}? Use notices/. "
                    "A notice applies from its Effective date inclusive. Choose the greatest Effective date "
                    "not after the cutoff, irrespective of when the notice was issued.")
        notices = {
            "notices/a.txt": f"Project: {prefix}\nIssued: 2025-03-15\nEffective: 2025-03-16\nPermit code: OLD-{i}\n",
            "notices/b.txt": f"Project: {prefix}\nIssued: 2025-02-09\nEffective: {cutoff}\nPermit code: LIVE-{i}\n",
            "notices/c.txt": f"Project: {prefix}\nIssued: 2025-03-17\nEffective: 2025-03-20\nPermit code: FUTURE-{i}\n",
        }
        suite.add("search", "effective_date_not_issue_date", i, question, f"LIVE-{i}",
                  check="exact_str", assets=notices)

        key = f'Station "{i}", north'
        path = f"letters/Case {i}.txt"
        suite.add("search", "quoted_csv_file_redirect", i,
                  f"Parse index.csv as CSV. Use the entry with exact key `{key}` and state release. "
                  "Follow the path in that row and give the service code exactly.",
                  f"SRV-{i}0", check="exact_str", assets={
                      "index.csv": table(("key", "state", "path"), [
                          (key, "draft", "letters/draft.txt"), (key, "release", path),
                          (f"{key}\nbackup", "release", "letters/decoy.txt"),
                      ]),
                      path: f"Service code: SRV-{i}0\n",
                      "letters/draft.txt": f"Service code: DRAFT-{i}\n",
                      "letters/decoy.txt": f"Service code: DECOY-{i}\n",
                  })

    suite.add("search", "divergent_employers", 0,
              "According to kb/, In which city is the headquarters of Riley Kent's employer?", None,
              check="exact_str", reason=UNKNOWN, assets=employer_assets("Riley Kent", "Rome", "Bern"))
    suite.add("search", "release_conflict_same_effective_date", 0,
              "For project Hedge, find the code effective on 2025-05-03. If several notices have begun, "
              "use the latest effective date. The ledger has no secondary priority.",
              None, check="exact_str", reason=UNKNOWN, assets={
                  "ledger.csv": "project,effective,code\nHedge,2025-05-03,AX\nHedge,2025-05-03,AY\n",
              })
    suite.add("search", "missing_translation_link", 0,
              "Use glossary.tsv to translate local code J-05, then use contacts.tsv to find the exact extension. "
              "The translation table is incomplete and codes have no numeric relationship.",
              None, check="exact_int", reason=UNKNOWN, assets={
                  "glossary.tsv": "local\tcentral\nJ-06\tHQ-8\n",
                  "contacts.tsv": "central\textension\nHQ-7\t128\nHQ-8\t172\n",
              })
    suite.add("search", "ambiguous_quoted_csv_redirect", 0,
              "Follow the sole released route for key `W, 2` from index.csv and give its service code. "
              "Multiple released rows have equal priority.",
              None, check="exact_str", reason=UNKNOWN, assets={
                  "index.csv": 'key,state,path\n"W, 2",release,a.txt\n"W, 2",release,b.txt\n',
                  "a.txt": "Service code: X-1\n", "b.txt": "Service code: X-2\n",
              })


def data_cases(suite):
    for i in range(4):
        cents = 50 + 7*i
        suite.add("data_analysis", "same_revision_equivalent_amounts", i,
                  "Use events.csv and accounts.csv. Discard draft rows before selecting the greatest revision per order. "
                  "Join account IDs as exact strings with leading zeroes. For North, add sale cents and subtract refund cents. "
                  "An unregistered account can belong to North or South. Give net cents for North.",
                  cents - 3, assets={
                      "events.csv": "order,account,revision,state,event,cents\n"
                                    f"o1,001,2,posted,sale,00{cents}\no1,001,2,posted,sale,{cents}\n"
                                    "o1,001,3,draft,sale,999\no2,001,1,posted,refund,3\n",
                      "accounts.csv": "account,region\n001,North\n",
                  })

        cut = 2**53 + 1 + i*20
        suite.add("data_analysis", "wide_integer_filter", i,
                  f"In counts.csv, what is the sum of units where key = {cut}?", 37 + i, assets={
                      "counts.csv": f"key,units\n{cut-1},900\n{cut},37\n{cut},{i}\n{cut+1},800\n",
                  })

        winner = f"group_{i}"
        folder = suite.add("data_analysis", "sqlite_null_group_sum", i,
                           "In scores.sqlite, table scores, which team has the highest sum of points?",
                           winner, check="exact_str", assets={
                               "schema.txt": "Use ordinary SQLite SUM. A NULL points cell contributes no value.\n",
                           })
        with sqlite3.connect(folder / "scores.sqlite") as db:
            db.execute("CREATE TABLE scores(team TEXT, points INTEGER)")
            db.executemany("INSERT INTO scores VALUES (?, ?)", [
                (winner, 35 + i), (winner, None), ("runner_up", 14), ("runner_up", 9),
            ])

        rows = [("east", 2+i, 6+i), ("east", 19+i, 27+i), ("west", 40, 40)]
        rate = Fraction(sum(ok for zone, ok, _ in rows if zone == "east"),
                        sum(total for zone, _, total in rows if zone == "east"))
        suite.add("data_analysis", "weighted_rate_not_mean", i,
                  "Use rates.csv. For zone east, divide the total number of successes by the total number "
                  "of attempts across rows. Return the rate as a reduced fraction.",
                  str(rate), check="exact_str", assets={"rates.csv": table(("zone", "successes", "attempts"), rows)})

    suite.add("data_analysis", "unknown_distinct_category", 0,
              "In slots.csv, what is the number of distinct values of state?", None, reason=UNKNOWN, assets={
                  "slots.csv": "slot,state\n1,red\n2,blue\n3,\n",
                  "format.txt": "Each slot has a state. Blank means unrecorded, not empty. "
                                "The possible states are red, blue and green.\n",
              })
    suite.add("data_analysis", "unknown_weighted_denominator", 0,
              "Use rates.csv to divide total successes by total attempts for east. The unknown attempts count "
              "is a real positive number of attempts. Give the exact ratio.", None, reason=UNKNOWN,
              assets={"rates.csv": "zone,successes,attempts\neast,4,9\neast,2,\n"}, check="exact_str")
    suite.add("data_analysis", "empty_rate_denominator", 0,
              "Compute the exact east conversion rate in clicks.csv as the sum of clicks divided by the sum "
              "of impressions.", None, reason=UNDEFINED,
              assets={"clicks.csv": "zone,clicks,impressions\neast,0,0\nwest,4,10\n"}, check="exact_str")
    suite.add("data_analysis", "tie_without_priority", 0,
              "In revenue.csv, which team has the highest sum of cents?", None, reason=UNKNOWN,
              assets={"revenue.csv": "team,cents\na,8\na,2\nb,10\nc,9\n"}, check="exact_str")


def unit_test(function, statements):
    body = "".join(f"        {statement}\n" for statement in statements)
    return f"import unittest\nfrom app import {function}\n\nclass Contract(unittest.TestCase):\n    def test_contract(self):\n{body}\nif __name__ == '__main__':\n    unittest.main()\n"


def coding_cases(suite):
    def add(family, i, function, prompt, broken, reference, public, private):
        suite.add("coding", family, i,
                  f"{prompt} Fix app.py. Run `python3 test_app.py`. Preserve the tests and return DONE.",
                  "tests_pass", check="unittest",
                  assets={"app.py": broken, "test_app.py": unit_test(function, public)},
                  references={"app.py": reference}, hidden=unit_test(function, private))

    for i in range(4):
        values = [7+i, 1+i, -4-i]
        add("clamp_reversed_endpoints", i, "clamp",
            "Implement clamp(x, lo, hi). Treat lo and hi as unordered endpoints; clamp x to the inclusive interval.",
            "def clamp(x, lo, hi):\n    return min(lo, max(hi, x))\n",
            "def clamp(x, lo, hi):\n    if lo > hi:\n        lo, hi = hi, lo\n    return min(hi, max(lo, x))\n",
            [f"self.assertEqual(clamp({8+i}, 2, 6), 6)"],
            [f"self.assertEqual(clamp({3+i}, 8, 1), {3+i})",
             "self.assertEqual(clamp(-12, 8, 1), 1)", "self.assertEqual(clamp(12, 8, 1), 8)"])
        add("first_n_nonpositive", i, "first_n",
            "Implement first_n(values, n). Return a new list of the first n values from a list. "
            "If n is zero or negative return an empty list. Keep original order.",
            "def first_n(values, n):\n    return values[:n + 1]\n",
            "def first_n(values, n):\n    return values[:max(0, n)]\n",
            [f"self.assertEqual(first_n({values}, 2), {values[:2]})"],
            [f"self.assertEqual(first_n({values}, -1), [])",
             f"self.assertEqual(first_n({values}, 20), {values})", "self.assertEqual(first_n([], 5), [])"])
        add("mean_one_shot_and_empty", i, "average",
            "Implement average(values) for a finite iterable of numbers, including a one-shot iterator. "
            "Return the arithmetic mean or None when it is empty.",
            "def average(values):\n    return sum(values) // len(values)\n",
            "def average(values):\n    materialized = list(values)\n"
            "    return sum(materialized) / len(materialized) if materialized else None\n",
            [f"self.assertEqual(average([{i}, {i+3}]), {i+1.5})"],
            [f"self.assertEqual(average(iter({values})), {sum(values)/len(values)!r})",
             "self.assertIsNone(average(iter([])))", "self.assertEqual(average([0, 0]), 0)"])
        token = f"letter{i}"
        add("unicode_casefold_first_spelling", i, "unique_in_order",
            "Implement unique_in_order(values) for strings. Compare via Unicode casefold. Keep only the "
            "first occurrence of each casefolded string, with its original spelling and position.",
            "def unique_in_order(values):\n    return sorted(set(values))\n",
            "def unique_in_order(values):\n    first = {}\n    for value in values:\n"
            "        first.setdefault(value.casefold(), value)\n    return list(first.values())\n",
            [f"self.assertEqual(unique_in_order(['z', {token!r}, 'z']), ['z', {token!r}])"],
            ["self.assertEqual(unique_in_order(['Straße', 'STRASSE', 'É', 'é', 'X']), ['Straße', 'É', 'X'])",
             "self.assertEqual(unique_in_order([]), [])"])

    impossible = (
        ("public_pass_secret_table_absent", "Implement service_code() to return the exact service identifier "
         "from release.txt. No other source is authoritative. The current placeholder passes a smoke test only.",
         "service_code", "def service_code():\n    return 'placeholder'\n", "self.assertIsInstance(service_code(), str)", UNKNOWN),
        ("public_pass_conflicting_return_types", "For input [1], pack(values) must return a built-in list "
         "and must return a built-in tuple for the same call. Both exact type requirements are mandatory.",
         "pack", "def pack(values):\n    return list(values)\n", "self.assertEqual(len(pack([1])), 1)", CONFLICT),
        ("public_pass_policy_unknown", "Implement tariff(quantity) under the table in tariff.csv. "
         "The only test value is quantity zero. Give the correct tariff for all allowed quantities.",
         "tariff", "def tariff(quantity):\n    return 0\n", "self.assertEqual(tariff(0), 0)", UNKNOWN),
        ("public_pass_impossible_inverse", "Implement invert_square(y) as a single-valued inverse of square "
         "on all integers. For every integer n, invert_square(n*n) must equal n exactly.",
         "invert_square", "def invert_square(y):\n    return 0\n", "self.assertEqual(invert_square(0), 0)", CONFLICT),
    )
    for i, (family, prompt, function, source, test, reason) in enumerate(impossible):
        suite.add("coding", family, i, prompt, None, check="unittest", reason=reason,
                  assets={"app.py": source, "test_app.py": unit_test(function, [test])})


def build(root, seed):
    suite = Suite(root, seed, None)
    for generate in (math_cases, search_cases, data_cases, coding_cases):
        generate(suite)
    counts = Counter((entry["kind"], suite.oracle[name] is None) for name, entry in suite.cases.items())
    expected = {(kind, impossible): 4 if impossible else 16 for kind in KINDS for impossible in (False, True)}
    if counts != expected:
        raise ValueError("each type needs 16 solvable and 4 impossible tasks")
    for entry in suite.cases.values():
        entry["split"] = "adversarial"
    suite.finish()
    print(json.dumps({"tasks": len(suite.oracle), "impossible": 16, "seed": seed,
                      "families": len({entry["family"] for entry in suite.cases.values()})}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path)
    parser.add_argument("--seed", type=int, default=20261117)
    parser.add_argument("--verify-coding", action="store_true")
    parser.add_argument("--zip", type=Path)
    args = parser.parse_args()
    build(args.suite, args.seed)
    if args.verify_coding:
        verify_coding(args.suite)
    if args.zip:
        archive(args.suite, args.zip)


if __name__ == "__main__":
    main()
