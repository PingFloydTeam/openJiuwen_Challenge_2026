"""Generate fresh values and paired changes for the robustness comparison."""
import argparse
from pathlib import Path

from .hard import Suite
from .large import archive, markdown, py_tests


def generate(output, seed, shifted=False):
    suite = Suite(output, seed, None)
    unknown = ("missing_information", "two or more answers remain possible")
    conflict = ("logical_error", "the constraints are inconsistent")
    for index in range(1 if shifted else 3):
        width, length, slow, fast, hours = suite.rng.sample(range(4, 21), 5)
        area = (f"We need enough paint for a rectangular panel whose sides are {width} and {length} metres. "
                "Report its surface area in square metres." if shifted else
                f"Find the area in square metres of a {width} by {length} metre rectangle.")
        suite.add("math", "area", index, area, width * length)
        mode = "the same direction" if index % 2 else "opposite directions"
        gap = abs(fast - slow) if index % 2 else fast + slow
        movement = (f"How far apart, in kilometres, are runners after {hours} hours if they leave one point in opposite directions "
                    f"at {slow} and {fast} km/h?" if shifted else
                    f"Two runners leave the same point in {mode} and run at {slow} and {fast} km/h for "
                    f"{hours} hours. How many kilometres separate them?")
        suite.add("math", "movement", index, movement, gap * hours)
        received, sent = 2 + index, 3 + index
        ratio = 2 + index % 2
        starting_left = ratio * length - received
        combined = starting_left + length + sent
        motion = (f"Together two depots started with {combined} boxes. Depot Red took in {received} and depot Blue sent out {sent}. "
                  f"At closing Red held {'twice' if ratio == 2 else 'three times'} as many boxes as Blue. ")
        prompt = "What was depot Blue's starting count?"
        possible = [other for other in range(combined + 1) if other >= sent
                    and combined - other + received == ratio * (other - sent)]
        discrepancy = f"Their closing total was {combined + received - sent + 1}. "
        if shifted:
            motion = (f"Before trading, Red and Blue had {combined} boxes between them. Blue lost {sent} and Red gained {received}. "
                      "That left Red with double Blue's stock. ")
            prompt = "Give the number of boxes Blue started with."
            discrepancy = f"An independent count after trading reported {combined + received - sent + 1} in all. "
        suite.add("math", "inventory", index, motion + prompt, possible[0])
        suite.add("math", "inventory_contradiction", index, motion + discrepancy + prompt, None, reason=conflict)

        person, employer, city = (f"Kai Person-{index}", f"Lab-{index}", ("Turin", "Bergen", "Accra")[index])
        year = 1990 + length
        records = {"kb/person.md": markdown(person, {"Born": year, "Employer": employer}),
                   "kb/lab.md": markdown(employer, {"Headquarters": city})}
        birth = f"From the records under kb/, report the calendar year in which {person} was born." if shifted else \
            f"Return the birth year for {person} from kb/."
        suite.add("search", "birth_year", index, birth, str(year),
                  assets=records, check="exact_str")
        workplace = f"Use kb/ to identify who employs {person}. Give the headquarters city of that employer." if shifted else \
            f"Read kb/ and find the city that hosts the headquarters of the company that employs {person}."
        suite.add("search", "employer_city", index, workplace,
                  city, assets=records, check="exact_str")
        project, code = f"Clover-{index}", f"Q-{width}{length}"
        notices = {"policy.txt": "Only approved notices count. Use the most recent issue date among approved notices.\n",
                   "docs/late.txt": f"Project: {project}\nIssued: 2025-06-08\nStatus: draft\nContact code: draft-only\n",
                   "docs/current.txt": f"Project: {project}\nIssued: 2025-05-08\nStatus: approved\nContact code: {code}\n",
                   "docs/early.txt": f"Project: {project}\nIssued: 2024-05-08\nStatus: approved\nContact code: old-code\n"}
        notice_prompt = (f"Find the contact code for project {project}. Apply the document selection policy in policy.txt to the notices "
                         "under docs/. Return the code for the selected notice.")
        if shifted:
            notice_prompt = (f"Following policy.txt, choose the applicable notice in docs/ for {project}. "
                             "Give its contact code.")
        suite.add("search", "latest_notice", index, notice_prompt, code, assets=notices, check="exact_str")
        suite.add("search", "notice_tie", index, notice_prompt, None, check="exact_str", reason=unknown,
                  assets=notices | {"docs/tie.txt": notices["docs/current.txt"].replace(code, "other-current")})

        ledger = (f"order,account,revision,event,cents\no1,A,1,sale,{width}\no1,A,2,sale,{length*4}\n"
                  f"o2,A,1,refund,{slow}\no3,B,1,sale,{fast}\no4,C,1,sale,{hours}\n")
        request = ("What is net revenue in cents for region East using registry members.tsv and journal events.csv? "
                   "Link rows by account ID, keep the latest revision per order, and count refunds as negative. "
                   "An unlisted account can belong to any region.")
        if shifted:
            request = ("Match accounts in events.csv to their regions in members.tsv. "
                       "For each order keep only its greatest revision. For East, add sale cents and deduct refund cents. "
                       "An account absent from members.tsv may belong to East or West. Report the net cents for East.")
        tabular = {"members.tsv": "account\tregion\nA\tEast\nB\tWest\nC\tEast\n", "events.csv": ledger}
        suite.add("data_analysis", "revisions", index, request, length*4-slow+hours, assets=tabular)
        suite.add("data_analysis", "unlisted_account", index, request, None, reason=unknown,
                  assets=tabular | {"members.tsv": tabular["members.tsv"].replace("C\tEast\n", "")})
        stats = {"rows.tsv": f"part\tweek\tunits\nax\t1\t{width}\nax\t2\t{length}\nby\t2\t{slow}\n"}
        cutoff = "From rows.tsv, sum units after week 1, excluding week 1 itself." if shifted else \
            "In rows.tsv, total units for entries where week is at least 2."
        average = "Return the mean units of the week-one entries in rows.tsv." if shifted else \
            "Select rows in rows.tsv with week equal to 1. What is their average of units?"
        suite.add("data_analysis", "cutoff", index, cutoff,
                  length+slow, assets=stats)
        suite.add("data_analysis", "mean", index, average,
                  width, assets=stats)

        if not shifted:
            code_cases(suite, index, width, length)
    for entry in suite.cases.values():
        entry["split"] = "shifted" if shifted else "robustness"
    suite.finish()


def code_cases(suite, index, width, length):
    def add(function, original, reference, oracle, samples, hidden, specification):
        suite.add("coding", function, index, specification + " Run `python3 test_lib.py` and submit DONE after the fix.",
                  "tests_pass", check="unittest", assets={"lib.py": original,
                  "test_lib.py": py_tests("lib", function, samples, oracle)}, references={"lib.py": reference},
                  hidden=py_tests("lib", function, hidden, oracle))

    reference = "def total(values):\n    result = 0\n    for value in values:\n        result += value\n    return result\n"
    add("total", reference.replace("+=", "-="), reference, sum,
        [([width, -length, 2],), ([],)], [([-8, -3, 2],), ([4, 0],)],
        "Return the signed sum, including negative entries, and use zero for an empty list.")
    reference = "def unique_in_order(values):\n    return list(dict.fromkeys(values))\n"
    add("unique_in_order", reference.replace("list(dict.fromkeys(values))", "sorted(set(values))"), reference,
        lambda values: [v for i, v in enumerate(values) if v not in values[:i]],
        [([max(width, length), min(width, length), max(width, length)],), ([],)],
        [(["b", "a", "a", "c"],), ([0, -1, 0],)], "Remove duplicates in order of first occurrence.")
    reference = "def overlap(a, b):\n    return max(0, min(a[1], b[1]) - max(a[0], b[0]))\n"
    add("overlap", reference.replace("min(a[1], b[1])", "max(a[1], b[1])"), reference,
        lambda a, b: len(set(range(*a)) & set(range(*b))),
        [((0, width + 3), (1, width)), ((0, 2), (4, 6))], [((-5, 4), (-3, 3)), ((0, 5), (5, 7))],
        "Compute the length of intersection of two half-open intervals with integer endpoints.")
    suite.add("coding", "contradictory_contract", index,
              f"Implement score() as a deterministic function. The same call must return {width} and must return {width + 1}. "
              "Both requirements are mandatory.", None, check="unittest",
              reason=("logical_error", "one deterministic call cannot return distinct values"),
              assets={"lib.py": "def score():\n    raise NotImplementedError\n"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=20261124)
    parser.add_argument("--zip", type=Path)
    parser.add_argument("--shifted", action="store_true")
    args = parser.parse_args()
    generate(args.output, args.seed, args.shifted)
    if args.zip:
        archive(args.output, args.zip)


if __name__ == "__main__":
    main()
