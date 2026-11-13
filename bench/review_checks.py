"""Reproductions of independent review findings, with positive safety controls."""
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
import httpx

from ping_floyd.proxy import Budget,Proxy,token_usage
from ping_floyd.coding import Coding,test_run
from ping_floyd.contracts import required_inputs,universal_contract_conflicts
from ping_floyd.repair import try_repair
from ping_floyd.tools import kb_lookup,db_query
from ping_floyd.fast import fast_search
from ping_floyd.pipeline import record_plan
from ping_floyd.knowledge import kb_route
from ping_floyd.domain import NoUniqueAnswer


def response(action,finish='stop',usage=None):
    return httpx.Response(200,json={'choices':[{'message':{'content':json.dumps(action)},'finish_reason':finish}],
                                   'usage':usage or {'prompt_tokens':100,'completion_tokens':20,'total_tokens':120}})


class ReviewTransport(unittest.IsolatedAsyncioTestCase):
    async def check_replies(self,replies):
        budget=Budget(100000);events=[];calls=[]
        p=Proxy('http://test.invalid/v1','test', 'qwen3.8-27b-noreason',budget,events.append,time.monotonic()+10)
        await p.client.aclose()
        def handler(req):
            calls.append(json.loads(req.content));return replies[len(calls)-1]
        p.client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            result=await p.invoke('task',[{'role':'user','content':'test'}],90000,100)
            return result,budget,events,calls
        finally:await p.client.aclose()
    async def test_transient_429_preserves_global_budget_and_retries(self):
        result,budget,events,calls=await self.check_replies([httpx.Response(429,headers={'Retry-After':'0'}),response({'answer':'4'})])
        self.assertEqual(len(calls),2);self.assertFalse(budget.blocked)
        self.assertEqual(budget.spent,300);self.assertGreater(budget.reserved_unknown,0)
        self.assertEqual(budget.in_flight,0)
    async def test_paid_truncation_is_retried_without_losing_usage(self):
        result,budget,events,calls=await self.check_replies([response({'tool':'incomplete'},'length'),response({'answer':'4'})])
        self.assertEqual(len(calls),2);self.assertEqual(budget.spent,600)
        self.assertEqual(budget.reserved_unknown,0);self.assertGreater(calls[1]['max_tokens'],calls[0]['max_tokens'])
    def test_unexplained_total_is_not_trusted(self):
        with self.assertRaises(ValueError):token_usage({'prompt_tokens':100,'completion_tokens':20,'total_tokens':220})
        self.assertEqual(token_usage({'prompt_tokens':100,'completion_tokens':20,'total_tokens':120,'reasoning_tokens':5}),(100,20,5))
        self.assertEqual(token_usage({'prompt_tokens':100,'completion_tokens':20,'total_tokens':125,'reasoning_tokens':5}),(100,25,5))


class ReviewTools(unittest.TestCase):
    def test_old_landlock_abi_keeps_strict_filesystem_mode(self):
        from types import SimpleNamespace
        from ping_floyd.test_worker import restrict_paths
        for abi in (-1,1,2):
            self.assertFalse(restrict_paths(SimpleNamespace(syscall=lambda *_,a=abi:a)))

    def test_record_field_normalization_preserves_ids_and_nested_pointers(self):
        from ping_floyd.agent import Records
        result=Records.model_validate({'file':'a.json','filters':{'person':'007','/meta/code':'03'},'op':'unique','field':'dept'})
        self.assertEqual(result.filters,{'/person':'007','/meta/code':'03'})
        self.assertEqual(result.field,'/dept')

    def test_unsupported_field_does_not_prove_impossibility(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'a.md').write_text('# Ada Reed\nBirth year: 1983\n')
            with self.assertRaises(ValueError) as caught:kb_lookup(root,'Ada Reed',['Born'],strict=True)
            self.assertNotIsInstance(caught.exception,NoUniqueAnswer)
    def test_birth_date_projects_to_year(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'kb').mkdir();(root/'kb/a.md').write_text('# Ada Reed\nBorn: 1983-04-17\n')
            self.assertEqual(fast_search(root,'According to kb/, In which year was Ada Reed born?'),'1983')
    def test_zero_product_ignores_irrelevant_unknown(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'data.csv').write_text('price,units\n,0\n2,3\n')
            self.assertEqual(str(db_query(root,'data.csv','sum_product','price',factor='units')),'6')
            self.assertEqual(record_plan(root,{'file':'data.csv'},[],'sum_product','price','units'),'6')
            (root/'data.csv').write_text('price,units\n,1\n2,3\n')
            with self.assertRaises(NoUniqueAnswer):record_plan(root,{'file':'data.csv'},[],'sum_product','price','units')
    def test_direct_route_does_not_depend_on_dangling_branch(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'a.md').write_text('# A\nRoad: B\nRail: Missing\n');(root/'b.md').write_text('# B\nCity: X\n')
            self.assertEqual(kb_route(root,'A','B',['Road','Rail']),1)
            self.assertEqual(kb_route(root,'A','B',['Road','Rail'],'first'),'B')
    def test_runtime_input_and_stateful_calls_are_not_impossible(self):
        with tempfile.TemporaryDirectory() as d:
            required_inputs(Path(d),'Read integers from numbers.txt. The caller creates this file before invoking the function.')
            with self.assertRaises(NoUniqueAnswer):required_inputs(Path(d),'Return the exact value recorded in absent.txt.')
        universal_contract_conflicts('Implement a stateful counter. On successive calls `assert f() == 1` then `assert f() == 2`.')
        with self.assertRaises(NoUniqueAnswer):universal_contract_conflicts('A pure function must satisfy `assert f() == 1` and `assert f() == 2`.')
    @unittest.skipUnless(sys.platform=='linux','Linux isolation')
    def test_unknown_empty_contract_does_not_accept_generic_repair(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);src=root/'src';stage=root/'stage';src.mkdir();stage.mkdir()
            (src/'m.py').write_text('def average(xs):\n    return sum(xs) // len(xs)\n')
            (src/'test_m.py').write_text('import unittest\nfrom m import average\nclass T(unittest.TestCase):\n def test_average(self): self.assertEqual(average([1,2]),1.5)\nif __name__=="__main__": unittest.main()\n')
            q='Repair average(xs). Return 0 for empty inputs. Run `python3 test_m.py`.'
            c=Coding(src,stage,q);initial=c.test()
            self.assertFalse(initial['passed']);self.assertFalse(try_repair(c,initial['output'],q))
    @unittest.skipUnless(sys.platform=='linux','Linux isolation')
    def test_workspace_files_and_stdlib_imports_with_host_and_network_denied(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);root=base/'stage';root.mkdir();outside=base/'outside.txt';outside.write_text('private-fixture')
            (root/'numbers.txt').write_text('2 3 5');(root/'link.txt').symlink_to(outside)
            tests='''import unittest, socket
from pathlib import Path
class T(unittest.TestCase):
 def test_io(self):
  import tomllib
  self.assertEqual(tomllib.loads('n=7')['n'],7)
  self.assertEqual(sum(map(int,Path('numbers.txt').read_text().split())),10)
  Path('result.txt').write_text('10')
  self.assertEqual(Path('result.txt').read_text(),'10')
 def test_denials(self):
  with self.assertRaises(PermissionError): Path('link.txt').read_text()
  with self.assertRaises(PermissionError): Path('../outside.txt').write_text('bad')
  with self.assertRaises(PermissionError): socket.socket()
'''
            (root/'test_m.py').write_text(tests)
            result=test_run(root);self.assertTrue(result['passed'],result['output'])
            self.assertEqual(outside.read_text(),'private-fixture')


class ReviewWorkflow(unittest.IsolatedAsyncioTestCase):
    async def test_automatic_recovery_only_with_ample_budget(self):
        from bench.policy_checks import OfflinePolicyChecks,write_task,calculator
        for budget,retries in ((5000,0),(25000,1)):
            with self.subTest(budget=budget),tempfile.TemporaryDirectory() as d:
                root=Path(d);write_task(root,'q','Use a tool to calculate three times four.',3)
                models=[]
                def handler(req):
                    model=json.loads(req.content)['model'];models.append(model)
                    return response(calculator('3*4',model=='qwen3.8-27b'))
                events=await OfflinePolicyChecks.execute(self,root,handler,budget=budget,extra=('--agent-only',))
                self.assertEqual(events[-1]['model_retried'],retries)
                self.assertEqual((root/'q/answer.txt').exists(),bool(retries))
                if retries:
                    self.assertEqual(models[-1],'qwen3.8-27b')
                    self.assertEqual((root/'q/answer.txt').read_text(),'12\n')

    async def test_bounded_parity_inverse_accepts_equivalent_solver(self):
        from bench.policy_checks import OfflinePolicyChecks,write_task
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            write_task(root,'q','A process starts with an odd integer between 0 and 20. Multiply it by 3, then add 2 to obtain 23. Return the original integer.',3)
            action={'tool':'finite_search','args':{'domains':{'x':[0,20]},'constraints':['x%2==1','3*x+2==23'],'target':'x','mode':'unique'},'final':True}
            await OfflinePolicyChecks.execute(self,root,lambda _:response(action),budget=5000)
            self.assertEqual((root/'q/answer.txt').read_text(),'7\n')

    async def test_verified_links_expand_entity_scope(self):
        from bench.policy_checks import OfflinePolicyChecks,write_task
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);write_task(root,'q','According to kb/, follow Aster Reed through three Mentor links. Return the Born field of that person.',3,'search','exact_int')
            kb=root/'q/workspace/kb';kb.mkdir(parents=True)
            names=['Aster Reed','Birch Moss','Cedar Pine','Delta Vale']
            for i,name in enumerate(names):
                (kb/f'{i}.md').write_text(f'# {name}\n'+(f'Mentor: {names[i+1]}\n' if i<3 else 'Born: 1981\n'))
            actions=iter([{'tool':'lookup','args':{'entity':names[0],'fields':['Mentor','Mentor','Mentor']}},
                          {'tool':'lookup','args':{'entity':names[3],'fields':['Born']},'final':True}])
            await OfflinePolicyChecks.execute(self,root,lambda _:response(next(actions)),budget=5000,extra=('--agent-only',))
            self.assertEqual((root/'q/answer.txt').read_text(),'1981\n')

    async def test_fallback_accepts_alternative_field_and_prose_evidence(self):
        from bench.policy_checks import OfflinePolicyChecks,write_task
        for prose in (False,True):
            with self.subTest(prose=prose),tempfile.TemporaryDirectory() as d:
                root=Path(d);q='According to kb/, In which year was Ada Reed born?'
                write_task(root,'q',q,3,'search','exact_int');w=root/'q/workspace';(w/'kb').mkdir(parents=True)
                (w/'kb/a.md').write_text('# Ada Reed\n'+('Ada Reed was born in 1983 in Oslo.\n' if prose else 'Birth year: 1983\n'))
                actions=iter([{'tool':'read','args':{'file':'kb/a.md'}},{'answer':'1983'}] if prose else
                             [{'tool':'lookup','args':{'entity':'Ada Reed','fields':['Birth year']},'final':True}])
                await OfflinePolicyChecks.execute(self,root,lambda _:response(next(actions)),budget=5000)
                self.assertEqual((root/'q/answer.txt').read_text(),'1983\n')


if __name__=='__main__':unittest.main()
