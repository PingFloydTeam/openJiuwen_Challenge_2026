"""Paired Docker experiment. Credentials stay in memory and child environment."""
import argparse
import getpass
import json
import os
from pathlib import Path
from types import SimpleNamespace

from .validation import run


def main():
    p=argparse.ArgumentParser()
    p.add_argument('suite',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--baseline',default='ping-floyd:intelligence-baseline-arm')
    p.add_argument('--candidate',default='ping-floyd:intelligence-v1-arm')
    p.add_argument('--judge',default='ping-floyd:intelligence-v1-bench-arm')
    p.add_argument('--variants',nargs='+',choices=['baseline','candidate'],default=['baseline','candidate'])
    p.add_argument('--budget',type=int,default=100000);p.add_argument('--seconds',type=int,default=1600)
    p.add_argument('--offline',action='store_true')
    a=p.parse_args()
    if not a.offline:
        if not os.getenv('LLM_API_KEY'):
            os.environ['LLM_API_KEY']=getpass.getpass('NeuralDeep key (not saved): ')
        os.environ.setdefault('LLM_BASE_URL','https://api.neuraldeep.ru/v1')
    results=[]
    for variant in a.variants:
        out=a.output/variant
        if out.exists():
            raise ValueError(f'output already exists: {out}')
        print(json.dumps({'starting':variant,'budget':a.budget}),flush=True)
        run(SimpleNamespace(suite=a.suite,output=out,image=getattr(a,variant),bench_image=a.judge,
                            seconds=a.seconds,model='qwen3.8-27b-noreason',budget=a.budget,
                            live=not a.offline,env_file=None))
        results.append({'variant':variant,**json.loads((out/'summary.json').read_text())})
    (a.output/'comparison.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()
