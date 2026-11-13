"""Independent relational oracles, objective guards and hidden-contract checks."""
from collections import defaultdict
from fractions import Fraction
import json
from pathlib import Path
import random
import tempfile
import unittest

from ping_floyd.pipeline import record_plan
from ping_floyd.intent import symbolic_goal, explicit_equations, check_goal
from ping_floyd.memory import TaskMemory
from ping_floyd.contracts import contract_tests
from ping_floyd.domain import NoUniqueAnswer


class TypedPlanChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
    def tearDown(self):
        self.temp.cleanup()
    def source(self, rows, types=None):
        (self.root/'input.json').write_text(json.dumps(rows))
        return {'file':'input.json','types':types or {}}
    def test_datetime_order_uses_instants_and_preserves_equal_instants(self):
        src = self.source([
            {'at':'2026-01-01T01:00:00+03:00','label':'old'},
            {'at':'2025-12-31T23:00:00Z','label':'new'},
            {'at':'2026-01-01T02:00:00+03:00','label':'new'}],{'at':'datetime'})
        self.assertEqual(record_plan(self.root,src,[], 'argmax','label',order_by='at'),'new')
        rows = json.loads((self.root/'input.json').read_text());rows[-1]['label']='conflict'
        src = self.source(rows,{'at':'datetime'})
        with self.assertRaises(NoUniqueAnswer):
            record_plan(self.root,src,[],'argmax','label',order_by='at')
    def test_types_reject_invalid_values_without_guessing(self):
        for kind,value in [('date','2026-02-30'),('datetime','2026-01-01T00:00:00'),('integer','1.2'),('identifier',1)]:
            src = self.source([{'k':value}],{'k':kind})
            with self.assertRaises(ValueError):record_plan(self.root,src,[],'unique','k')
    def test_group_filter_rank_matches_independent_decimal_oracle(self):
        rng=random.Random(7415)
        for _ in range(25):
            rows=[{'key':rng.choice(['001','01','x']), 'amount':str(Fraction(rng.randint(-500,500),100))} for i in range(17)]
            totals=defaultdict(Fraction)
            for row in rows:totals[row['key']]+=Fraction(row['amount'])
            best=max(totals.values());winners=[k for k,v in totals.items() if v==best]
            src=self.source(rows,{'key':'identifier','amount':'decimal'})
            steps=[{'op':'group','by':['key'],'aggregates':{'total':{'op':'sum','field':'amount'},'n':{'op':'count'}}},
                   {'op':'where','filters':[['n','>',0]]}]
            if len(winners)==1:
                self.assertEqual(record_plan(self.root,src,steps,'argmax','key',order_by='total'),winners[0])
            else:
                with self.assertRaises(NoUniqueAnswer):record_plan(self.root,src,steps,'argmax','key',order_by='total')
    def test_group_can_be_composed_twice(self):
        src=self.source([{'a':'x','b':'u','v':1},{'a':'x','b':'v','v':2},{'a':'y','b':'u','v':7}])
        steps=[{'op':'group','by':['a','b'],'aggregates':{'total':{'op':'sum','field':'v'}}},
               {'op':'group','by':['a'],'aggregates':{'maximum':{'op':'max','field':'total'}}}]
        self.assertEqual(record_plan(self.root,src,steps,'sum','maximum'),'9')
    def test_excluded_unknown_date_is_irrelevant(self):
        src=self.source([{'scope':'out','at':'bad'},{'scope':'in','at':'2026-03-04'}],{'at':'date'})
        self.assertEqual(record_plan(self.root,src,[{'op':'where','filters':[['scope','=','in']]}],'unique','at'),'2026-03-04')
    def test_group_rejects_unknown_arguments(self):
        src=self.source([{'a':'x','v':2}])
        for spec in [{'op':'count','field':'v'},{'op':'sum','field':'v','ignore_null':True},{'op':'eval','field':'v'}]:
            with self.assertRaises(ValueError):
                record_plan(self.root,src,[{'op':'group','by':['a'],'aggregates':{'n':spec}}],'sum','n')
    def test_ranking_stages_retain_rows_and_ties(self):
        src=self.source([{'k':'a','n':3,'label':'OK'},{'k':'b','n':3,'label':'OK'},{'k':'c','n':1,'label':'LOW'}])
        self.assertEqual(record_plan(self.root,src,[{'op':'argmax','field':'n','order_by':'n'}],'unique','label'),'OK')
        self.assertEqual(record_plan(self.root,src,[{'op':'argmin','order_by':'n'}],'unique','label'),'LOW')
        with self.assertRaises(ValueError):record_plan(self.root,src,[{'op':'argmax','field':'label','order_by':'n'}],'unique','label')


class GoalChecks(unittest.TestCase):
    def test_guard_blocks_correct_constant_for_wrong_question(self):
        goal=symbolic_goal('Integers x,y are in [0,71]. x+y=71. Return x.')
        self.assertEqual(goal,('x','unique'))
        with self.assertRaises(ValueError):check_goal(goal,'calc',{'expr':'71'})
        with self.assertRaises(ValueError):check_goal(goal,'finite_search',{'target':'x+y','mode':'unique'})
        check_goal(goal,'finite_search',{'target':'x','mode':'unique'})
    def test_equivalent_affine_objective_is_accepted(self):
        goal=symbolic_goal('x+y=71. Return 2*x+2*y.')
        check_goal(goal,'finite_search',{'target':'2*(x+y)','mode':'unique'})
    def test_unknown_language_does_not_become_a_fake_variable(self):
        for question in ['Take the remainder of 8 divided by 3. Return the remainder.',
                         'Return the unique winning project.', 'Find the area in square meters.']:
            self.assertIsNone(symbolic_goal(question))
    def test_fraction_objective(self):
        self.assertEqual(symbolic_goal('x+y=71 and z=x+2*y. Return the maximum x*y/(z+1) as a reduced fraction (integer if denominator 1).'),('x*y/(z+1)','max'))
    def test_task_equations_cannot_be_silently_dropped(self):
        q='x+y=71; z=x+2*y. Return x.'
        required=explicit_equations(q)
        self.assertEqual(len(required),2)
        with self.assertRaises(ValueError):check_goal(symbolic_goal(q),'linear',{'variable':'x','equations':['x=71']},required)
        check_goal(symbolic_goal(q),'finite_search',{'target':'x','constraints':['2*x+2*y=142','2*z=2*x+4*y']},required)
    def test_quantified_sum_needs_constraint_solver(self):
        goal=symbolic_goal('a+b=24. Sum a*a+b over all feasible ordered pairs.')
        self.assertEqual(goal,('a*a+b','sum'))
        with self.assertRaises(ValueError):check_goal(goal,'calc',{'expr':'24'})
    def test_bounds_and_parity_cannot_be_dropped(self):
        q='Integers a,b lie in [0,24]. a+b=24; a is odd. Return a.'
        args={'target':'a','domains':{'a':[0,24],'b':[0,24]},'constraints':['a+b=24','a%2==1']}
        check_goal(symbolic_goal(q),'finite_search',args,explicit_equations(q),q)
        with self.assertRaises(ValueError):check_goal(symbolic_goal(q),'finite_search',{**args,'constraints':['a+b=24']},explicit_equations(q),q)
        with self.assertRaises(ValueError):check_goal(symbolic_goal(q),'finite_search',{**args,'domains':{'a':[1,1],'b':[0,24]}},explicit_equations(q),q)


class PropertyChecks(unittest.TestCase):
    def run_source(self, source, instruction):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'m.py').write_text(source)
            generated=contract_tests(root,{'m.py'},instruction)
            self.assertIsNotNone(generated)
            # Test-only execution of the generator against explicitly authored sources.
            import subprocess,sys
            (root/'check.py').write_text(generated)
            return subprocess.run([sys.executable,'check.py'],cwd=root,capture_output=True,text=True)
    def test_universal_equation_catches_public_example_overfit(self):
        q='For every integer n, unfamiliar(n) must equal 3*n+2 exactly.'
        self.assertNotEqual(self.run_source('def unfamiliar(n): return 5\n',q).returncode,0)
        self.assertEqual(self.run_source('def unfamiliar(n): return 3*n+2\n',q).returncode,0)
    def test_metamorphic_property_catches_duplicate_loss(self):
        q='For every list of integers xs, `assert sorted(alien(xs)) == sorted(xs)`.'
        self.assertNotEqual(self.run_source('def alien(xs): return sorted(set(xs))\n',q).returncode,0)
        self.assertEqual(self.run_source('def alien(xs): return sorted(xs)\n',q).returncode,0)
    def test_natural_language_equations_use_exact_division(self):
        q='For every integer n, divide(n) must equal n/3 exactly.'
        self.assertEqual(self.run_source('from fractions import Fraction\ndef divide(n): return Fraction(n,3)\n',q).returncode,0)
        self.assertNotEqual(self.run_source('def divide(n): return n/3\n',q).returncode,0)
    def test_no_arbitrary_code_from_task(self):
        from ping_floyd.properties import quantified_checks
        self.assertEqual(quantified_checks('For every integer n, `assert __import__("os").system("echo bad") == f(n)`.',{'f':['m']}),[])
    def test_errors_do_not_erase_known_facts(self):
        state=TaskMemory(300)
        state.success('schema','columns: code,revision')
        state.failure('invalid date')
        state.failure('invalid operation')
        self.assertIn('columns: code,revision',state.render())
        self.assertNotIn('invalid date',state.render())
        self.assertLessEqual(len(state.render()),300)


if __name__=='__main__':unittest.main()
