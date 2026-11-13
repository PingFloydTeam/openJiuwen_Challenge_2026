"""New English compositional benchmark; judge data never enters the agent mount.

Development and control use different compositions, not just new random seeds.
Coding public tests are deliberately incomplete; independent private assertions
check general behavior, iterables, signed/large inputs and input preservation.
"""
import argparse
from datetime import datetime, timedelta, timezone
from fractions import Fraction
import json
from pathlib import Path
import re

from .hard import Suite
from .diversity import table, tests


def math_case(s,i,dev):
    a,b,c=s.rng.sample(['u','v','w','p','q','r'],3)
    n=s.rng.randrange(17,40);k=s.rng.randrange(2,5)
    if dev:
        points=[(x,n-x) for x in range(n+1) if x%k==1]
        if i%2:
            answer=sum(x*x+y for x,y in points);mode='Sum';expr=f'{a}*{a}+{b}'
            end=f'Sum {expr} over all feasible ordered pairs.'
        else:
            answer=max(x*y for x,y in points);end=f'Return the maximum {a}*{b}.'
        q=f'{a},{b} are integers in [0,{n}]. {a}+{b}={n}, {a} mod {k}=1. {end}'
        family='dev_feasible_sum' if i%2 else 'dev_bounded_product'
    else:
        family=['fractional_constrained_objective','lattice_count','invariant_objective','quadratic_minimum'][i%4]
        if i%4==0:
            points=[(x,n-x,2*x+n-x) for x in range(n+1) if (n-x)%k==1]
            answer=max(Fraction((x+1)*(y+1),z+1) for x,y,z in points)
            q=f'Integers {a},{b},{c} lie in [0,{3*n}]. {a}+{b}={n}; {c}=2*{a}+{b}; {b} mod {k}=1. Return the maximum ({a}+1)*({b}+1)/({c}+1) as a reduced fraction.'
        elif i%4==1:
            answer=sum(1 for x in range(n+1) for y in range(n+1) if (z:=n-2*x-3*y)>=0 and (x+z)%k==0)
            q=f'Count all ordered integer triples {a},{b},{c} in [0,{n}] such that 2*{a}+3*{b}+{c}={n} and ({a}+{c}) mod {k}=0.'
        elif i%4==2:
            answer=3*n
            q=f'Integers {a},{b},{c} lie in [0,{3*n}]. {a}+{b}={n}; {c}={a}+2*{b}. Return 3*{a}+3*{b}.'
        else:
            answer=min(x*x+(n-x)**2 for x in range(n+1) if x%2==1)
            q=f'Integers {a},{b} lie in [0,{n}]. {a}+{b}={n}; {a} is odd. Return the minimum {a}*{a}+{b}*{b}.'
    s.add('math',family,i,q,str(answer),check='exact_str')


def search_case(s,i,dev):
    token=f'K{s.rng.randrange(1000,9999)}'; answer=f'CODE-{s.rng.randrange(10000,99999)}'
    if dev:
        rows=[{'team':'a','v':1},{'team':'a','v':2},{'team':'b','v':8}]
        assets={'rows.json':json.dumps(rows),'labels.json':json.dumps([{'team':'a','code':answer},{'team':'b','code':'OLD'}])}
        q='In rows.json count rows per team. Select the unique team with the most rows, join team to labels.json and return code. Ties must agree on the final code.'
        family='dev_group_count_join'
        if i%2:
            assets={'events.csv':table(['object','at','code'],[[token,'2026-03-01',answer],[token,'2026-04-01','FUTURE']])}
            q=f'For object {token} in events.csv choose the latest at date no later than 2026-03-15 and return code. Dates use YYYY-MM-DD.'
            family='dev_dated_csv_selection'
    else:
        family=['instant_asof_join','group_count_filter_join','typed_two_hop','tied_instant_projection'][i%4]
        if i%4 in {0,3}:
            rows=[{'id':token,'at':'2026-06-01T01:00:00+03:00','site':'old'},
                  {'id':token,'at':'2026-05-31T23:00:00Z','site':'new'},
                  {'id':token,'at':'2026-06-01T04:00:00+03:00','site':'future'}]
            if i%4==3:rows.append({'id':token,'at':'2026-06-01T01:00:00+02:00','site':'alias'})
            assets={'history.json':json.dumps({'events':rows}),'sites.csv':table(['site','label'],[['old','OLD'],['new',answer],['future','FUTURE'],['alias',answer]])}
            q=f'In history.json /events keep id {token} with at no later than instant 2026-06-01T00:00:00Z. Choose the latest instant, preserving all ties. Offsets indicate time zones, not string priorities. Join site to sites.csv and return label if all final labels agree.'
        elif i%4==1:
            rows=[{'team':'001','state':'ok'},{'team':'001','state':'ok'},{'team':'01','state':'ok'}, {'team':'1','state':'void'}]
            assets={'events.json':json.dumps(rows),'labels.csv':table(['team','label'],[['001',answer],['01','OTHER'],['1','VOID']])}
            q='Filter events.json to state ok. Count records per exact string team, keep teams with count >= 2, then join team to labels.csv. Return the unique label. Leading zeros distinguish teams.'
        else:
            assets={'people.json':json.dumps([{'person':token,'dept':'01'},{'person':token+'x','dept':'1'}]),
                    'departments.csv':table(['dept','site'],[['01','A'],['1','B']]),
                    'sites.json':json.dumps([{'site':'A','rev':2,'code':'OLD'},{'site':'A','rev':10,'code':answer},{'site':'B','rev':30,'code':'OTHER'}])}
            q=f'Find person {token} in people.json. Follow dept to departments.csv using exact string IDs, then site to sites.json. Among that site\'s records select greatest numeric rev and return code, requiring agreement on ties.'
    s.add('search',family,i,q,answer,assets=assets,check='exact_str')


def data_case(s,i,dev):
    rows=[];oracle={}
    for key in ['001','01','1','Q']:
        amounts=[Fraction(s.rng.randint(-20,95),10) for _ in range(s.rng.randint(2,5))]
        oracle[key]=(sum(amounts),len(amounts),max(amounts))
        rows.extend({'bucket':key,'amount':str(v),'state':'use'} for v in amounts)
    rows.append({'bucket':'irrelevant','amount':None,'state':'ignore'});s.rng.shuffle(rows)
    assets={'records.json':json.dumps({'items':rows})}
    if dev:
        answer=sum(v[0] for v in oracle.values() if v[1]>=3)
        q='In records.json /items retain state use, group by exact string bucket, and sum amount and count records per bucket. Keep buckets with count >= 3 and return the exact sum of their totals as a reduced fraction.'
        family='dev_group_having_total'
    elif i%4==0:
        answer=sum(v[2] for v in oracle.values())
        q='In records.json /items retain state use. Compute the maximum exact amount within EACH bucket, then return the sum of those maxima as a reduced fraction. Bucket IDs are strings.'
        family='sum_group_maxima'
    elif i%4==1:
        answer=sum(v[0] for v in oracle.values() if v[0]>0)
        q='In records.json /items retain state use. Sum exact amount per bucket, keep groups whose total is strictly positive, and sum the retained totals. Return a reduced fraction.'
        family='positive_group_totals'
    elif i%4==2:
        keep=['001','Q'];assets['active.csv']=table(['bucket'],[[k] for k in keep])
        answer=sum(oracle[k][2] for k in keep)
        q='Join records.json /items to active.csv on exact string bucket, retain state use, compute max amount per surviving bucket, then sum those maxima. Return an exact reduced fraction. Unknown values in excluded rows are irrelevant.'
        family='joined_group_extrema'
    else:
        answer=Fraction(sum(v[0] for v in oracle.values()),sum(v[1] for v in oracle.values()))
        q='In records.json /items retain state use. For each bucket compute total amount and number of records. Return sum of group totals divided by sum of group counts, exactly as a reduced fraction. Do not average group means.'
        family='weighted_group_mean'
    s.add('data_analysis',family,i,q,str(answer),assets=assets,check='exact_str')


def coding_case(s,i,dev):
    fn=f'process_{s.rng.randrange(1000,9999)}';a=s.rng.randrange(2,8);b=s.rng.randrange(-8,9)
    family=i%(2 if dev else 4)
    if dev:
        expr=f'{a}*n+{b}' if family==0 else f'n*n+{a}'
        body=f'return {expr}';broken=f'return {a}+{b}' if family==0 else f'return {a}+1'
        q=f'Implement {fn}(n). For every integer n, {fn}(n) must equal {expr} exactly.'
        public=[((1,),eval(expr,{'__builtins__':{}},{'n':1}))]
        private=f'for n in [-10000000000000001,-73,-4,0,2,19,10000000000000001]:\n            self.assertEqual(fn(n), {expr})'
        label='dev_universal_affine' if family==0 else 'dev_universal_quadratic'
    elif family==0:
        body=f'return n % {a}';broken=f'return abs(n) % {a}'
        q=f'Implement {fn}(n) for all signed Python integers. Return its Euclidean remainder modulo {a}, always from 0 through {a-1}. For every integer n, `assert {fn}(n+{a}) == {fn}(n)`. For every integer n, `assert 0 <= {fn}(n) < {a}`.'
        public=[((1,),1)];private=f'for n in [-10**40,-101,-{a},-1,0,2,10**40]:\n            self.assertEqual(fn(n), n % {a})';label='signed_periodic_remainder'
    elif family==1:
        body='return sorted(values)';broken='return sorted(set(values))'
        q=f'Implement {fn}(values) to return a new sorted list in nondecreasing order containing exactly all input integers, including multiplicities. Accept a finite iterable. Do not mutate inputs. For every list of integers xs, `assert sorted({fn}(xs)) == sorted(xs)`.'
        public=[(([2,1],),[1,2])]
        private='for xs in [[],[4,4,1],[-3,2,-3,0],[9]*17,list(range(47,-1,-1))]:\n            original=list(xs)\n            self.assertEqual(fn(xs), sorted(xs))\n            self.assertEqual(xs,original)\n            self.assertEqual(fn(iter(xs)), sorted(xs))';label='multiplicity_and_iterators'
    elif family==2:
        body=f'return (n + {a} - 1) // {a}';broken=f'return int(n / {a}) + int(n % {a} != 0)'
        q=f'Implement {fn}(n) as the exact mathematical ceiling of n/{a} for every signed integer, including arbitrarily large integers. For every integer n, `assert ({fn}(n)-1)*{a} < n <= {fn}(n)*{a}`.'
        public=[((1,),1)];private=f'for n in [-10**70-3,-19,-1,0,1,19,10**70+3]:\n            self.assertEqual(fn(n), -((-n)//{a}))';label='integer_ceiling_bounds'
    else:
        body=f'return {a}*n*n+{b}*n';broken=f'return ({a}+{b})*n'
        q=f'Implement {fn}(n). It must return 0 at n=0. For every integer n, `assert {fn}(n+1) - {fn}(n) == {a}*(2*n+1)+{b}`. The recurrence and the value at zero define the function on all integers.'
        public=[((1,),a+b)];private=f'for n in [-10**20,-29,-2,0,2,53,10**20]:\n            self.assertEqual(fn(n), {a}*n*n+{b}*n)';label='bidirectional_recurrence'
    arg='values' if not dev and family==1 else 'n'
    if not dev and (i//4)%2:
        # Some control tasks express the same full contract entirely in prose.
        # The new property parser must not be necessary for every coding success.
        if family==3:
            q=f'Implement {fn}(n) on all signed integers. At zero the result is zero. Increasing the input from n to n+1 increases the result by {a}*(2*n+1)+{b}. Return exact integers, including for very large inputs.'
        else:
            q=re.sub(r' For every (?:integer n|list of integers xs), `assert [^`]+`\.', '',q)
    source=f'def {fn}({arg}):\n    {broken}\n';reference=f'def {fn}({arg}):\n    {body}\n'
    hidden=f'import unittest\nfrom app import {fn} as fn\nclass Private(unittest.TestCase):\n    def test_contract(self):\n        {private}\nif __name__ == "__main__": unittest.main()\n'
    q+=' Run `python3 test_app.py`. Return DONE.'
    s.add('coding',label,i,q,'DONE',assets={'app.py':source,'test_app.py':tests(fn,public)},references={'app.py':reference},hidden=hidden,check='unittest')


def impossible(s,i,kind):
    n=s.rng.randint(12,97)
    assets={}
    if kind=='math':
        q=f'Integers u,v are in [0,{n}]. u+v={n}. Return u.' if i%2 else f'Integers u,v are in [0,{n}]. u+v={n}; 2*u+2*v={2*n+1}. Return u.'
    elif kind=='search':
        assets={'events.json':json.dumps([{'time':'2026-06-01T01:00:00+03:00','code':'A'}, {'time':'2026-05-31T22:00:00Z','code':'B'}])}
        q='Select the latest instant in events.json using the timezone offsets. Return its code. All equally late records must agree; no further priority exists.'
        if i%3==1:
            assets={'index.json':json.dumps([{'id':'P','site':'A'},{'id':'P','site':'B'}]),'sites.json':json.dumps([{'site':'A','code':'ONE'},{'site':'B','code':'TWO'}])}
            q='Follow every site for id P in index.json to sites.json. Return the unique code. Both index records have equal authority and neither may be discarded.'
        elif i%3==2:
            assets={'people.json':json.dumps([{'id':'P','year':1981}])}
            q='Return the exact day and month of birth for id P using people.json. Only the year is recorded. No other information is available.'
    elif kind=='data_analysis':
        assets={'data.csv':table(['group','value'],[['A',n],['A',''],['B',1]])}
        q='Using data.csv, sum value per group, then return the sum of those group totals. A blank value is unknown and may be any integer; no other data exists.'
        if i%3==1:
            assets={'data.csv':table(['num','den'],[[n,1],[1,-1]])}
            q='Return sum(num) divided by sum(den) in data.csv as an exact rational number. Division by zero has no defined value.'
        elif i%3==2:
            assets={'data.csv':table(['amount','currency'],[[n,'USD'],[1,'EUR']])}
            q='Return the exact total amount in USD from data.csv. No exchange rate or conversion policy is provided.'
    else:
        fn=f'map_{n}'
        q=f'Implement {fn}(n) using the mapping in absent_policy.json. This missing file is the only specification. Run `python3 test_app.py`.'
        if i%2==0:q=f'Implement {fn}(n). For every integer n, {fn}(n) must equal n+{n} exactly. For every integer n, {fn}(n+1) must equal n+{n} exactly. Both obligations are mandatory. Run `python3 test_app.py`.'
        assets={'app.py':f'def {fn}(n):\n    return n+{n}\n','test_app.py':tests(fn,[((1,),n+1)])}
    s.add(kind,'insufficient_'+kind,i,q,None,assets=assets,check='unittest' if kind=='coding' else 'exact_str',reason=('underdetermined_or_inconsistent','No unique required result follows from the provided inputs'))


def build(root,dev,seed):
    s=Suite(root,seed,None)
    for kind,fn in [('math',math_case),('search',search_case),('data_analysis',data_case),('coding',coding_case)]:
        for i in range(8 if dev else 19):fn(s,i,dev)
        for i in range(2 if dev else 6):impossible(s,i,kind)
    s.finish()
    print(json.dumps({'tasks':len(s.oracle),'unsolvable':sum(x is None for x in s.oracle.values()),'seed':seed,'development':dev}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--development',action='store_true');p.add_argument('--seed',type=int,default=2026111203)
    a=p.parse_args();build(a.root,a.development,a.seed)
