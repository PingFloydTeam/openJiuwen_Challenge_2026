"""Independent algebraic, relational, safety and openJiuwen integration checks."""
from fractions import Fraction
from itertools import product
import json
from pathlib import Path
import random
import tempfile
import unittest

from ping_floyd.finite import finite_search
from ping_floyd.pipeline import record_plan
from ping_floyd.records import records_query
from ping_floyd.contracts import contract_tests, literal_assertions, universal_contract_conflicts
from ping_floyd.coding import Coding
from ping_floyd.domain import NoUniqueAnswer


class AlgebraChecks(unittest.TestCase):
    def test_elimination_matches_independent_enumeration(self):
        rng=random.Random(9123)
        for _ in range(30):
            n=rng.randrange(2,12);c=rng.randrange(2,8)
            points=[(x,y,z) for x,y,z in product(range(n+1),repeat=3) if x+2*y==c and 2*z==x+y]
            domains={name:[0,n] for name in 'xyz'};conditions=[f'x+2*y={c}','2*z=x+y']
            self.assertEqual(finite_search(domains,conditions,mode='count'),len(points))
            if points:
                vals=[x*x+z-y for x,y,z in points]
                for mode,expected in [('min',min(vals)),('max',max(vals)),('sum',sum(vals))]:
                    self.assertEqual(Fraction(finite_search(domains,conditions,'x*x+z-y',mode)),expected)
    def test_large_cartesian_space_is_reduced_without_guessing_bounds(self):
        self.assertEqual(finite_search({'x':[0,1000],'y':[0,1000],'z':[0,1000]},['x+y=500','z=x-y','z%7=0'],'x*y','max'),'62500')
        self.assertEqual(finite_search({'x':[0,1000],'y':[0,1000]},['x+y=500','2*x+2*y=1001'],mode='count'),0)
        with self.assertRaises(ValueError):finite_search({'x':[0,1000],'y':[0,1000]},[],mode='count')
    def test_uniqueness_is_of_objective_not_assignment(self):
        self.assertEqual(finite_search({'a':[0,500],'b':[0,500]},['a+b=20'],'3*a+3*b'),'60')
        with self.assertRaises(NoUniqueAnswer):finite_search({'a':[0,500],'b':[0,500]},['a+b=20'],'a')


class PlanChecks(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def write(self,name,obj):
        (self.root/name).write_text(obj if isinstance(obj,str) else json.dumps(obj));return {'file':name}
    def test_filter_precedes_missing_check_and_decimal_arithmetic_is_exact(self):
        src=self.write('a.csv','scope,value,denom\nkeep,0.10,1\nother,,\nkeep,0.20,4\n')
        steps=[{'op':'where','filters':[['scope','=','keep']]}]
        self.assertEqual(record_plan(self.root,src,steps,'ratio_sums','value','denom'),'3/50')
        with self.assertRaises(NoUniqueAnswer):record_plan(self.root,src,[],'sum','value')
    def test_unknown_filter_does_not_poison_definite_nonmatch(self):
        src=self.write('a.json',[{'team':'west'},{'team':'east','n':3}])
        steps=[{'op':'where','filters':[['n','>',0],['team','=','east']]}]
        self.assertEqual(record_plan(self.root,src,steps,'sum','n'),'3')
    def test_latest_order_changes_result(self):
        src=self.write('a.json',[{'id':'a','r':1,'state':'yes','v':8},{'id':'a','r':10,'state':'no','v':99}])
        latest={'op':'latest','by':['id'],'order_by':'r'};where={'op':'where','filters':[['state','=','yes']]}
        self.assertEqual(record_plan(self.root,src,[where,latest],'sum','v'),'8')
        self.assertEqual(record_plan(self.root,src,[latest,where],'sum','v'),'0')
    def test_all_tied_projections_must_agree(self):
        src=self.write('a.json',[{'r':2,'site':'A'},{'r':2,'site':'B'},{'r':1,'site':'C'}])
        self.write('b.json',[{'site':'A','label':'Z'},{'site':'B','label':'Z'},{'site':'C','label':'OLD'}])
        steps=[{'op':'latest','by':[],'order_by':'r'}, {'op':'join','source':{'file':'b.json'},'left':'site','right':'site'}]
        self.assertEqual(record_plan(self.root,src,steps,'unique','/joined/label'),'Z')
        self.write('b.json',[{'site':'A','label':'Z'},{'site':'B','label':'Q'}])
        with self.assertRaises(NoUniqueAnswer):record_plan(self.root,src,steps,'unique','/joined/label')
    def test_antijoin_keeps_leading_zero_identifiers(self):
        src=self.write('a.json',[{'id':'001','v':99},{'id':'01','v':7},{'id':'1','v':9}])
        other=self.write('b.csv','id\n001\n001\n1\n')
        steps=[{'op':'join','source':other,'left':'id','right':'id','how':'anti'}]
        self.assertEqual(record_plan(self.root,src,steps,'sum','v'),'7')
    def test_numeric_csv_revisions_and_group_ties(self):
        src=self.write('a.csv','id,r,v\na,2,100\na,10,2\nb,1,2\n');src['numbers']=['r']
        steps=[{'op':'latest','by':['id'],'order_by':'r'}]
        self.assertEqual(record_plan(self.root,src,steps,'sum','v'),'4')
        with self.assertRaises(NoUniqueAnswer):record_plan(self.root,src,steps,'argmax_sum','v',group='id')
    def test_plan_rejects_unknown_stages_and_outside_files(self):
        src=self.write('a.json',[{'x':2}])
        for steps in [[{'op':'eval','code':'2+2'}],[{'op':'where','filters':[],'ignored':True}]]:
            with self.assertRaises(ValueError):record_plan(self.root,src,steps,'sum','x')
        with self.assertRaises(ValueError):record_plan(self.root,{'file':'../a.json'},[],'sum','x')
        with self.assertRaises(ValueError):records_query(self.root,'a.json',op='sum',field='/x',order_by='/x')


class ProtocolChecks(unittest.TestCase):
    def test_terminal_step_has_unambiguous_canonical_form(self):
        from ping_floyd.agent import RecordPlan
        obj=RecordPlan.model_validate({'source':{'file':'x.json','path':'$'},
            'steps':[{'op':'where','filters':[['id','=','A']]},{'op':'sum','field':'amount'}]})
        self.assertEqual(obj.op,'sum');self.assertEqual(obj.field,'amount')
        self.assertEqual(obj.source['path'],'');self.assertEqual(len(obj.steps),1)
        with self.assertRaises(ValueError):RecordPlan.model_validate({'source':{'file':'x.json'},'op':'count',
            'steps':[{'op':'sum','field':'amount'}]})


class ContractChecks(unittest.TestCase):
    def test_literal_assertions_do_not_execute_text(self):
        self.assertEqual(literal_assertions('Required: `assert alien([1], n=2) == [1, 1]`'), [('alien',[[1]],{'n':2},[1,1])])
        self.assertEqual(literal_assertions('`assert alien(__import__("os")) == 2`'),[])
        with self.assertRaises(NoUniqueAnswer):universal_contract_conflicts('`assert f(1) == 3` and `assert f(1) == 4`')
    def test_probes_and_contract_examples_guard_green_public_tests(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);src=root/'source';stage=root/'stage';src.mkdir();stage.mkdir()
            (src/'app.py').write_text('def alien(x):\n    return abs(x)\n')
            public='import unittest\nfrom app import alien\nclass T(unittest.TestCase):\n    def test_public(self): self.assertEqual(alien(2),2)\nif __name__ == "__main__": unittest.main()\n'
            (src/'test_app.py').write_text(public)
            coding=Coding(src,stage,'Required example: `assert alien(-2) == -2`.')
            self.assertFalse(coding.test()['passed'])
            checks=[{'file':'app.py','function':'alien','args':[0],'expected':0}]
            self.assertTrue(coding.replace_body('app.py','alien','return x',checks=checks)['passed'])
            self.assertEqual((src/'test_app.py').read_text(),public)
            coding.commit();self.assertIn('return x',(src/'app.py').read_text())
            self.assertFalse(list(src.glob('pf_*')))




class ContextChecks(unittest.IsolatedAsyncioTestCase):
    async def test_failed_repair_refreshes_model_source_context(self):
        import httpx
        from bench.policy_checks import OfflinePolicyChecks, write_task
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            write_task(root,'repair','Repair unfamiliar(x) to return x+2 for all integers. Run `python3 test_app.py`. Return DONE.',2,'coding','unittest')
            workspace=root/'repair'/'workspace';workspace.mkdir()
            (workspace/'app.py').write_text('def unfamiliar(x):\n    return x - 1\n')
            (workspace/'test_app.py').write_text('import unittest\nfrom app import unfamiliar\nclass T(unittest.TestCase):\n    def test_public(self): self.assertEqual(unfamiliar(1),3)\nif __name__ == "__main__": unittest.main()\n')
            calls=[]
            def handler(request):
                prompt=json.loads(request.content)['messages'][-1]['content'];calls.append(prompt)
                if len(calls)==1:
                    action={'tool':'replace_body','args':{'file':'app.py','function':'unfamiliar','body':'return x + 1'}}
                else:
                    self.assertIn('return x + 1',prompt)
                    self.assertNotIn('return x - 1',prompt)
                    action={'tool':'replace_body','args':{'file':'app.py','function':'unfamiliar','body':'return x + 2',
                            'checks':[{'file':'app.py','function':'unfamiliar','args':[0],'expected':2}]}}
                return httpx.Response(200,json={'choices':[{'message':{'content':json.dumps(action)},'finish_reason':'stop'}],
                                               'usage':{'prompt_tokens':100,'completion_tokens':40}})
            await OfflinePolicyChecks.execute(self,root,handler,budget=5000)
            self.assertEqual(len(calls),2)
            self.assertEqual((root/'repair'/'answer.txt').read_text(),'DONE\n')

if __name__=='__main__':unittest.main()
