"""Regression cases for evidence projection, equivalent plans and deadlines."""
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock,patch

from openjiuwen.core.common.logging.log_config import configure_log_config
configure_log_config({'level':'WARNING','output':['console'],'log_path':'/tmp/jiuwen-logs'})

from bench import policy_checks
from bench.policy_checks import write_task
from bench.review_checks import response
from ping_floyd.agent import Records,requested_fact,run_task,WORKFLOW_EXECUTE_TIMEOUT
from ping_floyd.domain import NoUniqueAnswer,Task
from ping_floyd.pipeline import value,column,record_plan


class RecoveryTools(unittest.TestCase):
    def test_nested_alias_preserves_literal_keys_unknowns_and_numeric_types(self):
        self.assertEqual(value({'j':{'code':'A'}},'j/code'),'A')
        self.assertEqual(value({'j':{'code':'A'},'j/code':'B'},'j/code'),'B')
        with self.assertRaises(NoUniqueAnswer):value({'j':{'code':'A'},'j/code':None},'j/code')
        self.assertEqual(column({'j':{'revision':'10'}},'j/revision',{'/j/revision':'integer'}),10)
        with self.assertRaises(ValueError):column({'j':{'revision':'bad'}},'j/revision',{'/j/revision':'integer'})

    def test_filter_alias_conflicts_are_rejected(self):
        args=Records.model_validate({'file':'a.json','filter':{'person':'003'}})
        self.assertEqual(args.filters,{'/person':'003'})
        with self.assertRaises(ValueError):Records.model_validate({'file':'a.json','filter':{'x':1},'filters':{'x':2}})

    def test_projection_is_not_count_or_general_prose(self):
        self.assertEqual(requested_fact('Inspect rows.json and return the unique badge, requiring agreement.'),'field:badge')
        self.assertIsNone(requested_fact('Count rows and return the count.'))
        self.assertIsNone(requested_fact('Return a reduced fraction.'))


class RecoveryWorkflow(unittest.IsolatedAsyncioTestCase):
    async def test_cardinality_cannot_answer_a_field_request(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);write_task(root,'q','In rows.json return the unique badge.',3,'search','exact_str')
            w=root/'q/workspace';w.mkdir();(w/'rows.json').write_text('[{"badge":"XZ"},{"badge":"XZ"}]')
            base={'source':{'file':'rows.json'},'field':'badge'}
            actions=iter([{'tool':'record_plan','args':{**base,'op':'count_distinct'},'final':True},
                          {'tool':'record_plan','args':{**base,'op':'unique'},'final':True}])
            events=await policy_checks.OfflinePolicyChecks.execute(self,root,lambda _:response(next(actions)),budget=5000)
            self.assertEqual((root/'q/answer.txt').read_text(),'XZ\n')
            self.assertEqual(sum(e.get('event')=='llm' for e in events),2)

    async def test_relational_link_accepts_complete_plan_and_retains_ties(self):
        for conflict in (False,True):
            with self.subTest(conflict=conflict),tempfile.TemporaryDirectory() as d:
                root=Path(d);write_task(root,'q','Find person P in people.json. Follow dept to departments.csv, then site to sites.json and return badge, requiring agreement on maximum numeric revision ties.',3,'search','exact_str')
                w=root/'q/workspace';w.mkdir()
                (w/'people.json').write_text('[{"person":"P","dept":"02"}]')
                (w/'departments.csv').write_text('dept,site\n02,A\n2,B\n')
                (w/'sites.json').write_text(json.dumps([{'site':'A','rev':'2','badge':'OLD'},{'site':'A','rev':'10','badge':'NEW'}, {'site':'A','rev':'10','badge':'CONFLICT' if conflict else 'NEW'}]))
                args={'source':{'file':'people.json'},'steps':[
                    {'op':'where','filters':[['person','=','P']]},
                    {'op':'join','source':{'file':'departments.csv'},'left':'dept','right':'dept','prefix':'d'},
                    {'op':'join','source':{'file':'sites.json','types':{'rev':'integer'}},'left':'d/site','right':'site','prefix':'s'},
                    {'op':'latest','order_by':'s/rev','by':[]}], 'op':'unique','field':'s/badge'}
                if conflict:
                    with self.assertRaises(NoUniqueAnswer):record_plan(w,**args)
                actions=iter([{'tool':'record_plan','args':args,'final':True},{'skip':'conflicting final badges'}])
                await policy_checks.OfflinePolicyChecks.execute(self,root,lambda _:response(next(actions)),budget=5000,extra=('--agent-only',))
                self.assertEqual((root/'q/answer.txt').exists(),not conflict)
                if not conflict:self.assertEqual((root/'q/answer.txt').read_text(),'NEW\n')

    async def test_sdk_timeout_is_bounded_by_global_deadline(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);write_task(root,'q','Calculate 1+2.',1)
            task=Task.load(root/'q/task.json')
            for seconds in (20,500):
                proxy=SimpleNamespace(deadline=time.monotonic()+seconds)
                mock=AsyncMock(return_value=SimpleNamespace(result={'output':{'saved':False}}))
                with patch('ping_floyd.agent.Runner.run_workflow',mock):
                    await run_task(task,proxy,lambda _:None,10000,5,False)
                actual=mock.call_args.kwargs['envs'][WORKFLOW_EXECUTE_TIMEOUT]
                self.assertGreater(actual,min(seconds,180)-2)
                self.assertLessEqual(actual,min(seconds,180))


if __name__=='__main__':unittest.main()
