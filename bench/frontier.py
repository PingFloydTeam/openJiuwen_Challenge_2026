"""English compositional stress test; no solver imports and no answers in task inputs.

Development: 40 tasks, 20% impossible. Transfer: 100 tasks, 24% impossible.
Transfer uses distinct solved families; variants change data, ordering and rewards.
This is author-designed transfer, not a blind external estimate of jury accuracy.
"""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import random
from .hard import Suite
from .diversity import table, tests


def add_math(s, i, family, dev):
    n = s.rng.randrange(180, 410)
    a, b = s.rng.randrange(2, 6), s.rng.randrange(6, 10)
    m = s.rng.choice([7, 11, 13, 17]); residue = s.rng.randrange(m)
    if dev:
        if family == 0:
            feasible=[(x,n-x,a*x+b*(n-x)) for x in range(n+1) if x*(n-x)%m==residue]
            if not feasible: return add_math(s,i,family,dev)
            answer=max(x*y-z for x,y,z in feasible)
            question=(f'Integer quantities x,y are between 0 and {n}, and z is between 0 and {b*n}. '
                      f'x+y={n}; z={a}*x+{b}*y; (x*y) mod {m}={residue}. '
                      'What is the maximum achievable x*y-z?')
            name='quadratic_inventory'
        else:
            answer=sum((x*x+(n-x))%m==residue for x in range(n+1))
            question=(f'Count ordered integer triples (x,y,z). Each coordinate is from 0 through {b*n}. '
                      f'x+y={n}; z={a}*x+{b}*y; (x*x+y) mod {m}={residue}.')
            name='count_affine_residue'
    else:
        if family==0:
            candidates=[(x,n-x) for x in range(n+1) if (x*(n-x))%m==0 and x%2==1]
            answer=min((x-y)**2+3*x for x,y in candidates)
            question=(f'x,y,z are integers in [0,{b*n}]. They obey 2*x+2*y={2*n}, '
                      f'z={a}*x+{b}*y, x is odd, and x*y is divisible by {m}. '
                      'Find the smallest (x-y)**2+3*x.')
            name='constrained_minimum'
        elif family==1:
            answer=sum(x*(n-x) for x in range(n+1) if x%m==residue)
            question=(f'x,y,z are integers from 0 through {b*n}. x+y={n}, '
                      f'2*z=2*x+4*y, x mod {m}={residue}. '
                      'Sum x*y over ALL feasible ordered triples, counting each triple once.')
            name='sum_over_feasible_set'
        elif family==2:
            answer=sum((n-3*x)>=0 and (2*x+(n-3*x))%2==0 for x in range(n+1))
            question=(f'How many ordered integer triples x,y,z in [0,{3*n}] satisfy '
                      f'3*x+y={n} and 2*z=2*x+y? Count only integer solutions, not rational ones.')
            name='integer_lattice_parity'
        else:
            answer=max(Fraction(x*(n-x),a*x+b*(n-x)+1) for x in range(n+1))
            question=(f'x,y,z are integers in [0,{b*n}]. x+y={n} and z={a}*x+{b}*y. '
                      'Return the maximum x*y/(z+1) as a reduced fraction (integer if denominator 1).')
            name='rational_objective'
    s.add('math',name,i,question,answer if type(answer) is int else str(answer),check='exact_int' if type(answer) is int else 'exact_str')


def add_search(s,i,family,dev):
    key=f'P{s.rng.randrange(10000,99999)}'; code=f'R-{s.rng.randrange(10000,99999)}'
    assets={}
    if dev:
        if family==0:
            rows=[{'id':key,'seq':1,'state':'approved','site':'wrong'},
                  {'id':key,'seq':10,'state':'approved','site':'right'},
                  {'id':key+'x','seq':100,'state':'approved','site':'wrong'}]
            s.rng.shuffle(rows)
            assets={'events.json':json.dumps(rows),'sites.json':json.dumps([{'site':'right','label':code},{'site':'wrong','label':'OLD'}])}
            q=f'In events.json select id {key}, then its greatest numeric seq. Join site to site in sites.json and return label. All tied matches must agree.'
            name='latest_json_join'
        else:
            rows=[{'id':key,'state':'approved','seq':4,'code':code}, {'id':key,'state':'draft','seq':12,'code':'DRAFT'},
                  {'id':key,'state':'approved','seq':1,'code':'OLD'}]
            s.rng.shuffle(rows); assets={'events.json':json.dumps(rows)}
            q=f'In events.json for id {key}, FIRST discard non-approved records, THEN choose maximum numeric seq. Return code, requiring all tied records to agree.'
            name='filtered_json_rank'
    else:
        if family==0:
            rows=[{'id':key,'revision':10,'state':'void','office':'bad'},
                  {'id':key,'revision':1,'state':'posted','office':'bad'},
                  {'id':'peer','revision':1,'state':'posted','office':'good'},
                  {'id':'peer','revision':9,'state':'posted','office':'good'}]
            s.rng.shuffle(rows)
            assets={'history.json':json.dumps({'rows':rows}),'offices.csv':table(['office','label'],[['good',code],['bad','DECOY']])}
            q='In history.json /rows first keep the greatest numeric revision PER id, then keep state posted. Join office with offices.csv. Return the unique label of the surviving offices. Preserve ties.'
            name='per_entity_latest_then_status'
        elif family==1:
            people=[{'pid':'001','location':'a'}, {'pid':'01','location':'b'}, {'pid':'1','location':'c'}]
            active=[{'pid':'001'},{'pid':'1'}]
            assets={'people.json':json.dumps(people),'active.json':json.dumps(active),'places.json':json.dumps([{'location':'a','label':'BAD'},{'location':'b','label':code},{'location':'c','label':'BAD'}])}
            q='Find people in people.json whose exact string pid has NO match in active.json. Join their location to places.json and return the unique label. 001, 01, 1 are distinct. Require agreement.'
            name='json_antijoin_then_lookup'
        elif family==2:
            rows=[{'owner':key,'effective':'2026-01-01','site':'old'}, {'owner':key,'effective':'2026-06-01','site':'new'},
                  {'owner':key,'effective':'2027-01-01','site':'future'}]
            s.rng.shuffle(rows)
            assets={'assignments.json':json.dumps(rows),'sites.csv':table(['site','label'],[['old','OLD'],['new',code],['future','FUTURE']])}
            q=f'For owner {key}, keep assignments.json rows with effective <= 2026-11-01, choose the greatest effective value, then join site to sites.csv. Return the unique label. Dates are ISO dates; preserve all ties.'
            name='asof_json_csv_join'
        else:
            rows=[{'id':key,'revision':7,'site':'a'},{'id':key,'revision':7,'site':'b'},{'id':key,'revision':2,'site':'old'}]
            s.rng.shuffle(rows)
            assets={'events.json':json.dumps(rows),'sites.json':json.dumps([{'site':'a','label':code},{'site':'b','label':code},{'site':'old','label':'OLD'}])}
            q=f'For id {key} in events.json retain ALL greatest revision records. Follow every site to sites.json. Intermediate sites may differ. Return label if ALL final labels agree.'
            name='projected_agreement_after_ties'
    s.add('search',name,i,q,code,assets=assets,check='exact_str')


def add_data(s,i,family,dev):
    base=s.rng.randrange(11,97); price=f'{base//10}.{base%10}7'
    if dev:
        if family==0:
            rows=[['keep','1.20','2'],['keep',price,'3'],['other','','5']]; s.rng.shuffle(rows)
            answer=Fraction('1.20')*2+Fraction(price)*3
            q="In sales.csv retain segment keep, then sum price * units exactly. Blank prices are unknown; other segments are irrelevant. Return a reduced fraction or integer."
            assets={'sales.csv':table(['segment','price','units'],rows)}; name='selected_exact_product'
        else:
            rows=[{'id':'a','rev':1,'n':99},{'id':'a','rev':9,'n':base},{'id':'b','rev':3,'n':5}]
            s.rng.shuffle(rows);answer=base+5
            assets={'counts.json':json.dumps(rows)}
            q='In counts.json keep the greatest numeric rev per id, then sum n. Each id has one maximum. Return the total.';name='json_grouped_latest_sum'
    else:
        if family==0:
            rows=[['a','1','1.00','1'],['a','10',price,'3'],['b','1','2.30','2']];s.rng.shuffle(rows)
            answer=Fraction(price)*3+Fraction('2.30')*2
            assets={'orders.csv':table(['id','revision','price','units'],rows)}
            q='In orders.csv retain maximum numeric revision per id, then sum price * units. Decimal prices are exact. Return a reduced fraction or integer.';name='decimal_latest_product'
        elif family==1:
            rows=[['yes','1.25','2.50'],['yes',price,'3.25'],['no','','']];s.rng.shuffle(rows)
            answer=(Fraction('1.25')+Fraction(price))/Fraction('5.75')
            assets={'rates.csv':table(['include','hits','views'],rows)}
            q='In rates.csv only include=yes rows matter. Return total hits divided by total views as an exact reduced fraction. Blank cells are unknown, not zero.';name='filtered_exact_ratio'
        elif family==2:
            assets={'accounts.json':json.dumps([{'id':'001','cents':base},{'id':'01','cents':7},{'id':'1','cents':13}]),
                    'paid.csv':table(['id'],[['001'],['001'],['1']])}
            answer=7
            q='Sum cents in accounts.json for accounts with NO exact string id match in paid.csv. Repeated payment rows do not multiply accounts. Leading zeros are significant.';name='heterogeneous_antijoin_sum'
        else:
            amount=Fraction(price)
            rows=[['X',price],['X','0.01'],['Y',price],['Z','-9.00']];s.rng.shuffle(rows)
            assets={'ledger.csv':table(['team','amount'],rows)};answer='X'
            q='In ledger.csv sum exact decimal amount per team. Return the name of the unique team with largest sum; no tie-break is authorized.';name='exact_decimal_group_winner'
    s.add('data_analysis',name,i,q,str(answer),assets=assets,check='exact_str')


def add_code(s,i,family,dev):
    fn=f'operation_{s.rng.randrange(100,999)}'
    if dev:
        if family==0:
            body="    if width <= 0:\n        raise ValueError('width')\n    values=list(values)\n    return [values[j:j+width]+[None]*max(0,j+width-len(values)) for j in range(0,len(values),width)]\n"
            broken=body.replace('if width <= 0:', 'if width < 0:').replace('+[None]*max(0,j+width-len(values))','')
            examples=[(([1,2,3],2),[[1,2],[3,None]]),(([],3),[]),(([1],3),[[1,None,None]])]
            desc='Partition a finite iterable into width-sized lists; pad the last block with None. Empty input returns []. width <= 0 raises ValueError. Do not mutate inputs.';name='padded_blocks'
        else:
            body="    if b == 0:\n        raise ZeroDivisionError('b')\n    return -((-a)//b)\n";broken="    return a//b\n"
            examples=[((7,3),3),((-7,3),-2),((7,-3),-2),((-7,-3),3)]
            desc='Return the mathematical ceiling of a/b for any signed integers a,b, exactly without float conversion. b=0 raises ZeroDivisionError.';name='signed_ceiling'
    else:
        if family==0:
            body="    pairs=list(pairs)\n    if any(w <= 0 for _,w in pairs):\n        raise ValueError('weight')\n    if not pairs:\n        raise ValueError('empty')\n    total=sum(w for _,w in pairs)\n    running=0\n    for x,w in sorted(pairs):\n        running+=w\n        if 2*running >= total:\n            return x\n"
            broken=body.replace('2*running >= total','2*running > total')
            examples=[(([[1,1],[3,1]],),1),(([[9,1],[2,3],[4,1]],),2),(([[7,2]],),7)]
            desc='Return the lower weighted median: sort value/weight pairs by value and choose the first value whose cumulative weight is at least half the total. Weights must be positive; empty input or any nonpositive weight raises ValueError. Accept a finite iterable. Do not mutate inputs.';name='weighted_median'
        elif family==1:
            body="    result=[]\n    for value,count in runs:\n        if type(count) is not int or count < 0:\n            raise ValueError('count')\n        result.extend([value]*count)\n    return result\n";broken=body.replace('count < 0','count <= 0')
            examples=[(([["a",2],["b",0],["c",1]],),['a','a','c']),(([],),[]),(([[None,2]],),[None,None])]
            desc='Expand (value,count) pairs in order into a list. Zero counts add nothing. Counts must have exact type int (bool is invalid) and be nonnegative; otherwise raise ValueError. Accept finite iterables. Do not mutate inputs.';name='validated_run_expansion'
        elif family==2:
            body="    values=list(values)\n    if not values:\n        return []\n    k %= len(values)\n    return values[k:]+values[:k]\n";broken=body.replace('k %= len(values)','k = min(k,len(values))')
            examples=[(([1,2,3],4),[2,3,1]),(([1,2,3],-1),[3,1,2]),(([],9),[])]
            desc='Rotate a finite iterable left by any signed integer k, reducing k modulo length. Return a list; empty input gives []. Do not mutate inputs.';name='modular_iterable_rotation'
        else:
            body="    result=[]\n    for x in values:\n        if result and x == result[-1][0]:\n            result[-1][1]+=1\n        else:\n            result.append([x,1])\n    return result\n";broken=body.replace('x == result[-1][0]','x == result[0][0]')
            examples=[((['a','b','b','a'],),[['a',1],['b',2],['a',1]]),(([],),[]),(([[1],[1],[2]],),[[[1],2],[[2],1]])]
            desc='Run-length encode consecutive equal values into [value,count] lists. Compare using equality, so values may be unhashable. Preserve order; separate non-adjacent equal runs. Accept finite iterables. Do not mutate inputs.';name='equality_run_encoding'
    signature=('values, width' if dev and family==0 else 'a, b' if dev else 'pairs' if family==0 else 'runs' if family==1 else 'values, k' if family==2 else 'values')
    reference=f'def {fn}({signature}):\n'+body
    source=f'def {fn}({signature}):\n'+broken
    public=tests(fn,examples[:1])
    extra=''
    if dev and family==0: extra=f'        with self.assertRaises(ValueError): {fn}([1],0)\n'
    elif not dev and family==0: extra=f'        with self.assertRaises(ValueError): {fn}([])\n        with self.assertRaises(ValueError): {fn}([[1,0]])\n'
    elif not dev and family==1: extra=f'        for n in [-1,True,1.5]:\n            with self.assertRaises(ValueError): {fn}([["a",n]])\n'
    # Independent private examples: expand/sort for medians, indexing for
    # rotation, itertools for grouping. No candidate tools or reference imports.
    private_examples=list(examples)
    if not dev:
        from itertools import groupby, repeat, chain
        for _ in range(24):
            if family==0:
                pairs=[[s.rng.randrange(-8,9),s.rng.randrange(1,5)] for _ in range(s.rng.randrange(1,8))]
                expanded=sorted(chain.from_iterable(repeat(x,w) for x,w in pairs))
                private_examples.append(((pairs,),expanded[(len(expanded)-1)//2]))
            elif family==1:
                runs=[[s.rng.choice(['x','y',None]),s.rng.randrange(0,5)] for _ in range(s.rng.randrange(0,8))]
                private_examples.append(((runs,),list(chain.from_iterable(repeat(x,n) for x,n in runs))))
            elif family==2:
                values=[s.rng.randrange(-9,10) for _ in range(s.rng.randrange(0,9))];k=s.rng.randrange(-100,101)
                private_examples.append(((values,k),[values[(j+k)%len(values)] for j in range(len(values))]))
            else:
                values=[[s.rng.randrange(3)] for _ in range(s.rng.randrange(0,12))]
                private_examples.append(((values,),[[v,len(list(g))] for v,g in groupby(values)]))
        first_args, first_answer=examples[0]
        extra+=f"        self.assertEqual({fn}(iter({first_args[0]!r})" + ''.join(', '+repr(x) for x in first_args[1:]) + f"), {first_answer!r})\n"
    hidden=tests(fn,private_examples,extra)
    assertion=' '.join(f'`assert {fn}(*{args!r}) == {expected!r}`' for args,expected in [])
    # Literal contract examples are separate from public tests, as in real specs.
    args,expected=examples[-1]
    assertion=f'`assert {fn}('+', '.join(repr(v) for v in args)+f') == {expected!r}`'
    q=f'Repair {fn}({signature}) in app.py. {desc} Required example: {assertion}. Run `python3 test_app.py`. Return DONE.'
    s.add('coding',name,i,q,'tests_pass',check='unittest',assets={'app.py':source,'test_app.py':public},references={'app.py':reference},hidden=hidden)


def impossible(s,i,kind,dev):
    v=s.rng.randrange(10,90)
    if kind=='math':
        q=(f'Integers x,y,z lie in [0,{v*4}]. x+y={v}; z=x+2*y. Return the unique value of x. No other constraints apply.' if i%2==0
           else f'Integers x,y,z lie in [0,{v*4}]. x+y={v}; 2*x+2*y={2*v+1}; z=x. Return x.')
        assets={}
    elif kind=='search':
        q='In events.json keep the greatest revision for id P, follow ALL site links to sites.json, and return the unique label. No secondary priority exists.'
        assets={'events.json':json.dumps([{'id':'P','revision':v,'site':'a'},{'id':'P','revision':v,'site':'b'}]),
                'sites.json':json.dumps([{'site':'a','label':'ONE'},{'site':'b','label':'TWO'}])}
    elif kind=='data_analysis':
        q='For region east in rates.csv, return the exact sum of amount. Blank amount means unknown, not zero. No other data exist.'
        assets={'rates.csv':table(['region','amount'],[['east',str(v)],['east',''],['west','3']])}
    else:
        fn=f'convert_{v}'
        if i%2:
            q=f'Implement {fn}(n) using the mapping in missing.json. This absent mapping is the only specification. Run `python3 test_app.py`. Return DONE.'
        else:
            q=f'Repair {fn}(n). It is deterministic. Both required examples must hold: `assert {fn}(1) == {v}` and `assert {fn}(1) == {v+1}`. Run `python3 test_app.py`. Return DONE.'
        assets={'app.py':f'def {fn}(n):\n    return {v}\n','test_app.py':tests(fn,[((1,),v)])}
    s.add(kind,'impossible_'+kind,i,q,None,assets=assets,reason=('missing_or_contradictory','No unique answer follows from supplied information'),check='unittest' if kind=='coding' else 'exact_str')


def build(root,dev,seed):
    s=Suite(root,seed,None)
    for kind,fn in [('math',add_math),('search',add_search),('data_analysis',add_data),('coding',add_code)]:
        for i in range(8 if dev else 19): fn(s,i,i%(2 if dev else 4),dev)
        for i in range(2 if dev else 6): impossible(s,i,kind,dev)
    s.finish()
    print(json.dumps({'root':str(root),'tasks':len(s.cases),'impossible':sum(v is None for v in s.oracle.values())}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--development',action='store_true');p.add_argument('--seed',type=int,default=871293)
    a=p.parse_args();build(a.root,a.development,a.seed)
