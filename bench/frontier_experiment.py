"""Paired live experiments with a key held only in process memory."""
import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
RUNNER=ROOT/'bench'/'portable_run.py'


def digest(root):
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts}


def snapshot(destination):
    destination.mkdir(parents=True,exist_ok=False)
    for n in ['run_agent.py','Dockerfile','requirements.txt','requirements.in']:
        shutil.copy2(ROOT/n,destination/n)
    shutil.copytree(ROOT/'ping_floyd',destination/'ping_floyd',ignore=shutil.ignore_patterns('__pycache__'))
    destination.with_suffix('.sha256.json').write_text(json.dumps(digest(destination),indent=2)+'\n')


def events(path):
    result=[]
    for line in path.read_text().splitlines():
        try: item=json.loads(line)
        except ValueError: continue
        if isinstance(item,dict):result.append(item)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('split');p.add_argument('--candidate',default='candidate-v1');p.add_argument('--budget',type=int,default=100000);p.add_argument('--repeat',default='r1');p.add_argument('--variants',nargs='+',default=['baseline','candidate']);p.add_argument('--offline',action='store_true')
    a=p.parse_args();root=a.root.resolve();suite=root/a.split
    candidate=root/a.candidate
    if 'candidate' in a.variants and not candidate.exists(): snapshot(candidate)
    env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
    if not a.offline:
        env['LLM_API_KEY']=getpass.getpass('NeuralDeep API key (not stored): ')
        env['LLM_BASE_URL']='https://api.neuraldeep.ru/v1'
    summaries=[]
    for variant in a.variants:
        source=root/'baseline' if variant=='baseline' else candidate
        label=f'{a.split}-{variant}-{a.candidate if variant=="candidate" else "original"}-{a.budget}-{a.repeat}'
        output=root/label
        before=digest(source)
        cmd=[sys.executable,str(RUNNER),'run',str(suite),'--source',str(source),'--output',str(output),'--budget',str(a.budget)]
        if a.offline:cmd.append('--offline')
        print('Starting '+label,flush=True)
        r=subprocess.run(cmd,env=env)
        # Oracle and private tests are copied only AFTER the agent exits.
        shutil.copy2(suite/'oracle.json',output/'oracle.json');shutil.copy2(suite/'metadata.json',output/'metadata.json')
        if (suite/'judge').exists():shutil.copytree(suite/'judge',output/'judge')
        with (output/'grade.jsonl').open('w') as f:
            g=subprocess.run([sys.executable,str(RUNNER),'grade',str(output)],stdout=f,check=True)
        grade=events(output/'grade.jsonl')[-1]
        run=events(output/'run.jsonl'); last=next((x for x in reversed(run) if x.get('event')=='run_end'),{})
        original_tests={str(f.relative_to(suite/'tasks')):hashlib.sha256(f.read_bytes()).hexdigest() for f in (suite/'tasks').glob('*/workspace/test*.py')}
        tests_ok=all(hashlib.sha256((output/'tasks'/n).read_bytes()).hexdigest()==h for n,h in original_tests.items())
        summary={'label':label,'agent_exit':r.returncode,'grade':grade,'agent_summary':last,
                 'llm_events':sum(x.get('event')=='llm' for x in run),'public_tests_unchanged':tests_ok,
                 'source_unchanged':before==digest(source),'runtime':json.loads((output/'runtime.json').read_text())}
        (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');summaries.append(summary)
        print(json.dumps(summary),flush=True)
    (root/f'results-{a.split}-{a.candidate}-{a.budget}-{a.repeat}.json').write_text(json.dumps(summaries,indent=2)+'\n')


if __name__=='__main__':main()
