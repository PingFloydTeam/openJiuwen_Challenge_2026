"""Generalization checks independent of benchmark answer files."""
from fractions import Fraction
from pathlib import Path
import tempfile
import unittest
import json

import httpx

from bench import policy_checks
from bench.policy_checks import write_task
from ping_floyd.coding import Coding
from ping_floyd.domain import InvalidInput, NoUniqueAnswer
from ping_floyd.finite import finite_search
from ping_floyd.records import records_catalog, records_query
from ping_floyd.relational import sql_query


class DiversityChecks(unittest.TestCase):
    def test_finite_counts_have_closed_form(self):
        for n in range(1,20):
            self.assertEqual(finite_search({"x":[0,n],"y":[0,n]},["x+y <= "+str(n)],mode="count"), (n+1)*(n+2)//2)
            self.assertEqual(Fraction(finite_search({"x":[0,n]},[],"x","sum")),n*(n+1)//2)
        self.assertEqual(finite_search({"x":[1,10],"y":[1,10]},["x*y=12","x+y=7"],"x*x+y*y"),"25")
        with self.assertRaises(NoUniqueAnswer):
            finite_search({"x":[1,4]},[],"x")

    def test_finite_rejects_undefined_and_excessive_work_without_execution(self):
        with self.assertRaises(NoUniqueAnswer): finite_search({"x":[0,2]},[],"1/x")
        self.assertEqual(finite_search({"x":[0,2]},["x != 0 and 1/x < 1"],"x"),"2")
        for target in ["__import__('os')", "x.real", "((x**6)**6)**6", "[x][0]"]:
            with self.assertRaises(ValueError): finite_search({"x":[1000,1000]},[],target)
        with self.assertRaises(ValueError): finite_search({"x":[0,1000],"y":[0,1000]},[],mode="count")

    def test_json_nested_paths_type_identity_and_tied_ranks(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/"r.json").write_text(json.dumps({"rows":[
                {"id":"01","active":True,"r":2,"a/b":{"x~y":"LIVE"}},
                {"id":"01","active":True,"r":1,"a/b":{"x~y":"OLD"}},
                {"id":1,"active":True,"r":9,"a/b":{"x~y":"WRONG"}}]}))
            args=dict(file="r.json",path="/rows",filters={"/id":"01"},op="argmax",field="/a~1b/x~0y",order_by="/r")
            self.assertEqual(records_query(root,**args),"LIVE")
            catalog=records_catalog(root,"r.json")
            self.assertEqual(catalog["/rows"]["records"],3)
            self.assertIn("/a~1b/x~0y",catalog["/rows"]["fields"])
            self.assertEqual(records_query(root,"r.json",path="/rows",filters={"/active":1},op="count"),0)
            (root/"r.jsonl").write_text('{"k":"X","rank":2,"v":"A"}\n{"k":"X","rank":2,"v":"B"}\n')
            with self.assertRaises(NoUniqueAnswer):
                records_query(root,"r.jsonl",filters={"/k":"X"},op="argmax",field="/v",order_by="/rank")
            (root/"single.json").write_text('[{"nested":[{"id":"x"}]}]')
            self.assertEqual(list(records_catalog(root,"single.json")),[""])

    def test_json_duplicate_keys_unknowns_and_huge_numbers_are_not_guessed(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for text in ['[{"x":1,"x":2}]','[{"x":1e1000000}]','[{"x":NaN}]']:
                (root/"r.json").write_text(text)
                with self.assertRaises(InvalidInput): records_query(root,"r.json",op="sum",field="/x")
            (root/"r.json").write_text('[{"x":0.1},{"x":0.2}]')
            self.assertEqual(records_query(root,"r.json",op="sum",field="/x"),"3/10")
            (root/"r.json").write_text('[{"x":1},{}]')
            with self.assertRaises(NoUniqueAnswer): records_query(root,"r.json",op="sum",field="/x")

    def test_sql_cannot_hide_unknown_inputs_or_break_text_ties(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/"a.csv").write_text("id,cents\na,4\nb,\n")
            with self.assertRaises(NoUniqueAnswer):
                sql_query(root,"","SELECT SUM(cents) FROM a",tables={"a":"a.csv"},integers={"a":["cents"]},require_complete=True)
            with self.assertRaises(NoUniqueAnswer):
                sql_query(root,"","SELECT id FROM a ORDER BY id LIMIT 1",tables={"a":"a.csv"})

    def test_invariants_follow_unfamiliar_function_names_and_public_literals(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/"source";stage=root/"stage";source.mkdir();stage.mkdir()
            (source/"app.py").write_text("def reorder(items):\n    items.sort()\n    return items\n")
            public="import unittest\nfrom app import reorder\nclass T(unittest.TestCase):\n    def test_ok(self):\n        self.assertEqual(reorder([1,2]), [1,2])\nif __name__ == '__main__': unittest.main()\n"
            (source/"test_app.py").write_text(public)
            coding=Coding(source,stage,"reorder(items) accepts lists in any order. Do not mutate inputs.")
            self.assertFalse(coding.test()["passed"])
            self.assertTrue(coding.replace_body("app.py","reorder","return sorted(items)")["passed"])
            self.assertEqual((source/"test_app.py").read_text(),public)
            self.assertEqual(coding.inspect_code("app.py")[0]["name"],"reorder")

    def test_protocol_metadata_is_recovered_without_changing_tool_arguments(self):
        from openjiuwen.core.common.logging.log_config import configure_log_config
        configure_log_config({"level": "WARNING", "output": ["console"], "log_path": "/tmp/jiuwen-logs"})
        from ping_floyd.agent import parse_action
        result=parse_action('{"tool":"calc","args":{"expr":"2+3","final":true}}')
        self.assertEqual(result,{"tool":"calc","args":{"expr":"2+3"},"final":True})
        with self.assertRaises(ValueError):parse_action('{"tool":"calc","final":false,"args":{"expr":"2","final":true}}')


class DiscoveryChecks(unittest.IsolatedAsyncioTestCase):
    execute=policy_checks.OfflinePolicyChecks.execute

    async def test_inverse_cannot_be_confirmed_by_reversing_rounded_arithmetic(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            write_task(root,"inverse","A counter starts with an even number of items. Add 3, divide by 2, then round down. The result is 5. What was the original number?",2)
            actions=iter([{"tool":"calc","args":{"expr":"5*2-3"},"final":True},
                          {"tool":"integer_inverse","args":{"operations":[["add",3],["floor_div",2]],"target":5},"final":True}])
            def handler(request):
                return httpx.Response(200,json={"choices":[{"message":{"content":json.dumps(next(actions))},"finish_reason":"stop"}],
                                               "usage":{"prompt_tokens":100,"completion_tokens":40}})
            await self.execute(root,handler)
            self.assertEqual((root/"inverse/answer.txt").read_text(),"8\n")

    async def test_ratio_cannot_be_confirmed_by_inexact_sql(self):
        for data, answer in [("zone,hits,tries\neast,6,9\n", "2/3\n"),
                             ("zone,hits,tries\neast,6,9\neast,2,\n",None)]:
            with self.subTest(data=data), tempfile.TemporaryDirectory() as folder:
                root=Path(folder)
                write_task(root,"ratio","Use rates.csv to divide total hits by total tries for east. Give the exact ratio.",2,"data_analysis","exact_str")
                workspace=root/"ratio/workspace";workspace.mkdir();(workspace/"rates.csv").write_text(data)
                actions=iter([{"tool":"sql","args":{"tables":{"t":"rates.csv"},"integers":{"t":["hits","tries"]},
                                          "statement":"SELECT SUM(hits)/SUM(tries) FROM t WHERE zone='east'"},"final":True},
                              {"tool":"query","args":{"file":"rates.csv","op":"ratio_sums","column":"hits","factor":"tries","filters":[["zone","=","east"]]},"final":True},
                              {"skip":"unknown denominator"}])
                def handler(request):
                    return httpx.Response(200,json={"choices":[{"message":{"content":json.dumps(next(actions))},"finish_reason":"stop"}],
                                                   "usage":{"prompt_tokens":100,"completion_tokens":40}})
                await self.execute(root,handler,budget=2000)
                path=root/"ratio/answer.txt"
                self.assertEqual(path.read_text() if path.exists() else None,answer)

    async def test_structured_tools_work_through_real_openjiuwen(self):
        cases = [
            ("search", "In people.json return the site code for id X. All matching records must agree.",
             "people.json", '[{"id":"X","site":{"code":"NORTH"}},{"id":"Y","site":{"code":"SOUTH"}}]',
             {"tool":"records","args":{"file":"people.json","filters":{"/id":"X"},"op":"unique","field":"/site/code"},"final":True}, "NORTH"),
            ("data_analysis", "Return the second largest distinct integer value from values.csv.",
             "values.csv", "value\n8\n8\n11\n-1\n",
             {"tool":"sql","args":{"tables":{"t":"values.csv"},"integers":{"t":["value"]},
                 "statement":"SELECT DISTINCT value FROM t ORDER BY value DESC LIMIT 1 OFFSET 1"},"final":True}, "8"),
        ]
        for kind, instruction, filename, data, action, answer in cases:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as folder:
                root=Path(folder)
                write_task(root,"structured",instruction,2,kind=kind,check="exact_str")
                workspace=root/"structured/workspace"; workspace.mkdir()
                (workspace/filename).write_text(data)
                def handler(request):
                    payload=json.loads(request.content)
                    self.assertIn(action["tool"],payload["messages"][0]["content"])
                    return httpx.Response(200,json={"choices":[{"message":{"content":json.dumps(action)},"finish_reason":"stop"}],
                                                   "usage":{"prompt_tokens":100,"completion_tokens":40}})
                await self.execute(root,handler,extra=("--tool-profile","compact"))
                self.assertEqual((root/"structured/answer.txt").read_text(),answer+"\n")

    async def test_discovery_recovers_a_tool_hidden_by_wording(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            write_task(root,"opaque","Restore the starting inventory from the supplied operations.",2)
            actions=iter([{"tool":"discover","args":{"names":["integer_inverse"]}},
                          {"tool":"integer_inverse","args":{"operations":[["add",3]],"target":8},"final":True}])
            def handler(request):
                return httpx.Response(200,json={"choices":[{"message":{"content":json.dumps(next(actions))},"finish_reason":"stop"}],"usage":{"prompt_tokens":100,"completion_tokens":30}})
            events=await self.execute(root,handler,extra=("--tool-profile","compact"))
            self.assertEqual((root/"opaque/answer.txt").read_text(),"5\n")
            self.assertTrue(any(e.get("tool")=="discover" for e in events))
