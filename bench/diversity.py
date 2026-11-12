"""New family-separated English controls, with independent expected results."""
import argparse
from collections import Counter
import csv
from io import StringIO
import json
from pathlib import Path

from .hard import Suite
from .large import archive


def table(columns, rows):
    stream = StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(columns)
    writer.writerows(rows)
    return stream.getvalue()


def tests(function, examples, extra=""):
    body = "".join(f"        self.assertEqual({function}(*{args!r}), {answer!r})\n" for args, answer in examples)
    return (f"import unittest\nfrom app import {function}\nclass Checks(unittest.TestCase):\n"
            f"    def test_values(self):\n{body}{extra}\nif __name__ == '__main__': unittest.main()\n")


def build(destination, split, seed):
    s = Suite(destination, seed, None)
    add = s.add
    for i in range(4):
        if split == "development":
            x, y = 3 + i, 7 - i
            add("math", "bounded_purchase", i,
                f"Buy x small packs at 3 credits and y large packs at 5 credits. There are {x+y} packs "
                f"and the bill is {3*x+5*y} credits. x and y are whole numbers from 0 through 20. How many small packs?", x)
            low, high = 10+i, 70+3*i
            value = sum((n % 3 == 0) != (n % 5 == 0) for n in range(low, high+1))
            add("math", "exclusive_divisibility", i,
                f"Among integers from {low} through {high} inclusive, how many are divisible by either 3 or 5, but not both?", value)
            rows = [{"person": {"id": f"P{i}"}, "active": True, "site": {"code": f"LIVE-{i}"}},
                    {"person": {"id": f"p{i}"}, "active": True, "site": {"code": "CASE-DECOY"}},
                    {"person": {"id": f"P{i}"}, "active": False, "site": {"code": "OLD"}}]
            s.rng.shuffle(rows)
            add("search", "nested_contact", i,
                f"Read contacts.json. In directory, select the active person with case-sensitive id P{i}. "
                "Give the site code. Every matching record must agree.", f"LIVE-{i}", check="exact_str",
                assets={"contacts.json": json.dumps({"directory": rows})})
            records = [{"key": f"K{i}", "revision": 4, "state": "approved", "code": f"NEW-{i}"},
                       {"key": f"K{i}", "revision": 2, "state": "approved", "code": "OLD"},
                       {"key": f"K{i}", "revision": 8, "state": "draft", "code": "DRAFT"}]
            s.rng.shuffle(records)
            add("search", "jsonl_revision", i,
                f"events.jsonl contains one JSON object per line. For key K{i}, choose the approved record with "
                "the greatest revision. Return its code; do not break ties arbitrarily.", f"NEW-{i}", check="exact_str",
                assets={"events.jsonl": "\n".join(json.dumps(row) for row in records)+"\n"})
            values = [19+i, 19+i, 7, 13+i, -2]
            s.rng.shuffle(values)
            add("data_analysis", "second_distinct_value", i,
                "Use scores.csv. Return the second highest distinct integer score. Repeated scores count only once.",
                sorted(set(values), reverse=True)[1], assets={"scores.csv": table(["score"], [[v] for v in values])})
            rows = [("A", 1, "posted", 90), ("A", 2, "void", 90), ("B", 1, "posted", 7),
                    ("B", 2, "posted", 20+i), ("C", 1, "draft", 40), ("C", 3, "posted", 30+i)]
            s.rng.shuffle(rows)
            add("data_analysis", "latest_then_status", i,
                "In journal.csv first keep the largest sequence for each id, then retain state posted. "
                "What is the total cents? A newer void or draft replaces an older posted record.", 50+2*i,
                assets={"journal.csv": table(["id", "sequence", "state", "cents"], rows)})
            reference = "def segments(values, width):\n    if width <= 0:\n        raise ValueError('width')\n    values = list(values)\n    return [values[j:j+width] for j in range(len(values)-width+1)]\n"
            broken = reference.replace("len(values)-width+1", "len(values)-width")
            public = tests("segments", [(([1, 2, 3+i], 2), [[1, 2], [2, 3+i]])])
            hidden = tests("segments", [(([], 1), []), (([1, 2], 3), []), (([2, 4], 1), [[2], [4]])],
                           "        self.assertEqual(segments(iter([1,2,3]),2), [[1,2],[2,3]])\n        with self.assertRaises(ValueError): segments([1], 0)\n")
            add("coding", "iterable_windows", i,
                "Repair segments(values, width) in app.py. Return every consecutive window of width as a list of lists. "
                "Accept any finite iterable including a one-shot iterator. Reject width <= 0 with ValueError. "
                "Do not mutate inputs. Run `python3 test_app.py`. Return DONE.", "tests_pass", check="unittest",
                assets={"app.py": broken, "test_app.py": public}, references={"app.py": reference}, hidden=hidden)
            reference = "def keep_recent(values):\n    values = list(values)\n    last = {v.casefold(): i for i,v in enumerate(values)}\n    return [v for i,v in enumerate(values) if last[v.casefold()] == i]\n"
            public = tests("keep_recent", [(([f"A{i}", f"a{i}", "B"],), [f"a{i}", "B"])])
            hidden = tests("keep_recent", [((["Z", "a", "z", "A"],), ["z", "A"]), ((["Maße", "MASSE", "ß", "SS"],), ["MASSE", "SS"]), (([],), [])])
            add("coding", "last_casefold_occurrence", i,
                "Repair keep_recent(values). Compare strings by Unicode casefold; keep the LAST occurrence of each "
                "equivalence class, in order of the retained positions, with original spelling. Accept finite iterables. "
                "Do not mutate inputs. Run `python3 test_app.py`. Return DONE.", "tests_pass", check="unittest",
                assets={"app.py": "def keep_recent(values):\n    return list(dict.fromkeys(values))\n", "test_app.py": public},
                references={"app.py": reference}, hidden=hidden)
        else:
            x,y=2+i,8+i
            add("math", "bounded_nonlinear", i,
                f"Positive integers a <= b <= 20 have product {x*y} and sum {x+y}. Return a*a+b*b.", x*x+y*y)
            cap=17+3*i
            optimum=max(4*a+7*b for a in range(9) for b in range(9) if 3*a+5*b<=cap)
            add("math", "integer_optimization", i,
                f"Choose integer quantities x,y between 0 and 8 inclusive. Spend 3*x+5*y <= {cap}. "
                "What is the largest achievable benefit 4*x+7*y? Return the benefit, not the quantities.", optimum)
            rows=[{"group":"east","site":{"city":"Lima"}}, {"group":"east","site":{"city":"Lima"}},
                  {"group":"east","site":{"city":f"Town{i}"}}, {"group":"west","site":{"city":"Rome"}}]
            add("search", "nested_distinct_count", i,
                "In the JSON array locations.json, count distinct site.city values among records whose group is east. "
                "All matching cities are known. Return only the count.", 2, assets={"locations.json":json.dumps(rows)})
            rows=[{"team/id":f"T/{i}","office~code":f"DEST-{i}"},{"team/id":f"T-{i}","office~code":"WRONG"}]
            add("search", "escaped_json_fields", i,
                f"mapping.json contains entries. The field names literally contain slash and tilde. For team/id equal "
                f"to T/{i}, return office~code exactly. Require agreement of all matches.", f"DEST-{i}", check="exact_str",
                assets={"mapping.json":json.dumps({"entries":rows})})
            accounts=[("001",12+i),("01",7+i),("1",5+i),("Z",9+i)]
            sales=[("001","2025-02-01"),("001","2025-02-02"),("1","2024-01-01")]
            add("data_analysis", "anti_join_exact_ids", i,
                "accounts.csv gives id and cents; sales.csv gives account and day. Sum cents for accounts with NO sale "
                "on or after 2025-01-01. IDs are exact strings: 001, 01 and 1 are different. Do not multiply accounts by sale rows.",
                21+3*i, assets={"accounts.csv":table(["id","cents"],accounts),"sales.csv":table(["account","day"],sales)})
            rows=[("a",3+i),("b",6+i),("a",5+i),("b",-2),("c",1)]
            totals={key:sum(v for k,v in rows if k==key) for key,_ in rows}
            add("data_analysis", "group_peak", i,
                "Read measurements.csv. Add signed amount within each group, then report the uniquely greatest group total "
                "as an integer. Return the total rather than the group name.", max(totals.values()),
                assets={"measurements.csv":table(["group","amount"],rows)})
            reference="def decode_pair(text):\n    key, sep, value = text.partition(':')\n    if not sep or not key.strip():\n        raise ValueError('pair')\n    return [key.strip(), value.strip()]\n"
            public=tests("decode_pair", [((f" k{i} : a:b ",),[f"k{i}","a:b"])])
            hidden=tests("decode_pair", [(("a:",),["a",""]), (("ä : x:y:z",),["ä","x:y:z"])],
                         "        for value in ['missing', ':empty', '  :x']:\n            with self.assertRaises(ValueError): decode_pair(value)\n")
            add("coding", "split_first_separator", i,
                "Repair decode_pair(text) in app.py. Split on the FIRST colon only. Strip both parts and return a two-item "
                "list. An empty value is allowed; missing colon or empty stripped key raises ValueError. "
                "Run `python3 test_app.py`. Return DONE.","tests_pass",check="unittest",
                assets={"app.py":"def decode_pair(text):\n    key,value=text.split(':')\n    return [key.strip(),value.strip()]\n","test_app.py":public},
                references={"app.py":reference},hidden=hidden)
            reference="def compress(spans):\n    result=[]\n    for a,b in sorted(spans):\n        if result and a <= result[-1][1]:\n            result[-1][1]=max(result[-1][1],b)\n        else:\n            result.append([a,b])\n    return result\n"
            public=tests("compress", [(([[i,i+2],[i+2,i+4]],),[[i,i+4]])])
            hidden=tests("compress", [(([[4,5],[1,3],[2,6]],),[[1,6]]), (([],),[]), (([[1,1],[1,2]],),[[1,2]])],
                         "        values=[[4,5],[1,2]]\n        compress(values)\n        self.assertEqual(values, [[4,5],[1,2]])\n        self.assertEqual(compress(iter([[1,2],[2,4]])), [[1,4]])\n")
            add("coding", "merge_touching_intervals", i,
                "Repair compress(spans) in app.py. Merge overlapping OR touching intervals, sorted by start. "
                "Return lists [start,end]. Inputs have start <= end and may arrive in any order as a finite iterable. "
                "Do not mutate inputs. Run `python3 test_app.py`. Return DONE.","tests_pass",check="unittest",
                assets={"app.py":reference.replace("a <= result[-1][1]","a < result[-1][1]"),"test_app.py":public},
                references={"app.py":reference},hidden=hidden)

    impossible = ({
        "math": [("free_count","Two nonnegative integers sum to 12. What is the first integer?",{}),
                 ("inconsistent_counts","Integer counts x,y obey x+y=8 and 2*x+2*y=17. Give x.",{})],
        "search": [("conflicting_json","Use contacts.json. For id X return the unique code. No record has priority.",
                    {"contacts.json":'[{"id":"X","code":"A"},{"id":"X","code":"B"}]'}),
                   ("missing_json_field","Use contacts.json. Give the site code for id X; that field is absent and no other source exists.",
                    {"contacts.json":'[{"id":"X","site":{}}]'})],
        "data_analysis": [("tied_aggregate","Using scores.csv, name the sole group with greatest total value. Ties have no priority.",
                           {"scores.csv":"group,value\na,5\na,2\nb,7\n"}),
                          ("unknown_amount","In money.csv, a blank cents cell is an unknown amount, not zero. Give the exact total cents.",
                           {"money.csv":"id,cents\na,12\nb,\n"})],
        "coding": [("absent_conversion","Implement convert(x) according to mapping.json. The missing mapping is the only specification.",
                     {"app.py":"def convert(x):\n    return x\n"}),
                    ("noninjective_contract","For every integer z, undo(z*z) must equal z exactly.",
                     {"app.py":"def undo(x):\n    return 0\n"})],
    } if split=="development" else {
        "math": [("unbounded_optimum","A nonnegative integer x has no upper bound. Give the largest value of 7*x.",{}),
                 ("indivisible_cohort","Exactly two thirds of a cohort of eight people are adults. Every person is an adult or a child. How many adults?",{})],
        "search": [("tied_jsonl_latest","In events.jsonl return code for key X at greatest revision. There is no secondary priority.",
                    {"events.jsonl":'{"key":"X","revision":9,"code":"A"}\n{"key":"X","revision":9,"code":"B"}\n'}),
                   ("duplicate_json_keys","The file has a repeated code key with conflicting values. No duplicate-key precedence is defined. What is the unique code for X in records.json?",
                    {"records.json":'[{"id":"X","code":"A","code":"B"}]'})],
        "data_analysis": [("absent_join_key","Use entries.csv and accounts.csv to sum cents for the owner of each entry. Entries have no account ID, and no mapping is available.",
                           {"entries.csv":"cents\n10\n20\n","accounts.csv":"id,owner\na,A\nb,B\n"}),
                          ("undefined_ratio","Return total clicks / total views from rates.csv as an exact number.",
                           {"rates.csv":"clicks,views\n0,0\n0,0\n"})],
        "coding": [("conflicting_types","f() must return a built-in dict and must return a built-in tuple for the same call. Both exact type requirements are mandatory.",
                     {"app.py":"def f():\n    return {}\n"}),
                    ("missing_tariff","Implement price(n) using the table in tariffs.csv. There is no alternative source for the missing table.",
                     {"app.py":"def price(n):\n    return 0\n"})],
    })
    for kind,cases in impossible.items():
        for i,(family,question,assets) in enumerate(cases):
            add(kind,family,i,question,None,assets=assets,check="unittest" if kind=="coding" else "exact_str",
                reason=("missing_information","No unique valid answer under the explicitly stated conditions"))
    for entry in s.cases.values():
        entry["split"]=split
    s.finish()
    assert Counter((entry["kind"],s.oracle[name] is None) for name,entry in s.cases.items()) == Counter(
        {(kind,impossible):2 if impossible else 8 for kind in ("math","search","data_analysis","coding") for impossible in (False,True)})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("destination",type=Path)
    parser.add_argument("--seed",type=int,default=20261121)
    args=parser.parse_args()
    for split in ("development","holdout"):
        build(args.destination/split,split,args.seed)
        archive(args.destination/split,args.destination/f"{split}.zip")


if __name__=="__main__": main()
