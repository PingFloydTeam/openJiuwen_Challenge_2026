"""Submission contract: organizer budget is authoritative, including zero."""
from contextlib import redirect_stderr
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from bench import policy_checks
from bench.review_checks import response
import run_agent


class BudgetOptions(unittest.TestCase):
    def parse(self, folder, environment, *args):
        with patch.dict(os.environ):
            os.environ.pop('BUDGET',None)
            if environment is not None:os.environ['BUDGET']=environment
            with patch('sys.argv',['run_agent.py','--tasks-dir',str(folder),*args]):
                return run_agent.options()

    def test_environment_accepts_variable_and_zero_budgets(self):
        with tempfile.TemporaryDirectory() as d:
            for budget in (0,1,137,5000,17500,100000,250000):
                with self.subTest(budget=budget):
                    args=self.parse(d,str(budget))
                    self.assertEqual(args.budget,budget)
                    self.assertEqual(args.budget_source,'BUDGET')

    def test_cli_cannot_raise_environment_ceiling(self):
        with tempfile.TemporaryDirectory() as d:
            for env,cli,expected in [('137','100000',137),('17500','5000',5000),('0','100000',0)]:
                self.assertEqual(self.parse(d,env,'--budget',cli).budget,expected)

    def test_local_budget_without_environment(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(self.parse(d,None,'--budget','250').budget,250)
            self.assertEqual(self.parse(d,None,'--offline').budget,0)
            with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                self.parse(d,None)

    def test_invalid_environment_is_not_masked_by_cli(self):
        with tempfile.TemporaryDirectory() as d:
            for value in ('','-1','5.5','unknown'):
                with self.subTest(value=value),redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                    self.parse(d,value,'--budget','100000')
            with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                self.parse(d,'100','--budget=-1')


class BudgetWorkflow(unittest.IsolatedAsyncioTestCase):
    async def test_environment_caps_real_scheduler_and_preserves_free_answers(self):
        # Mock only HTTP; keep real openJiuwen, tools, reservations and saved output.
        for budget in (0,1,137,5000,17500,100000,250000):
            with self.subTest(budget=budget),tempfile.TemporaryDirectory() as d:
                root=Path(d);policy_checks.write_task(root,'local','Calculate 2+3.',1)
                for i in range(8):
                    policy_checks.write_task(root,f'fallback-{i}',f'Budget case {i}. Use a tool to return seven.',2)
                calls=[]
                def handler(req):
                    calls.append(json.loads(req.content))
                    return response({'tool':'calc','args':{'expr':'7'},'final':True})
                with patch.dict(os.environ,{'BUDGET':str(budget)}):
                    events=await policy_checks.OfflinePolicyChecks.execute(self,root,handler,budget=1000000,
                        extra=('--workers','5','--no-auto-retry'))
                start=next(e for e in events if e.get('event')=='run_start');end=events[-1]
                self.assertEqual(start['budget_tenths'],budget*10)
                self.assertEqual(start['budget_source'],'min(BUDGET,--budget)')
                self.assertLessEqual(end['spent_tenths']+end['unknown_reserved_tenths'],budget*10)
                self.assertEqual(end['remaining_tenths'],budget*10-end['spent_tenths']-end['unknown_reserved_tenths'])
                self.assertEqual((root/'local/answer.txt').read_text(),'5\n')
                if budget<=1:self.assertFalse(calls)
                if budget>=5000:self.assertEqual(len(list(root.glob('*/answer.txt'))),9)


if __name__=='__main__':unittest.main()
