"""Exercise the actual contest image entrypoints with variable BUDGET and OUT."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from .experiments import docker_prefix


def smoke(image, output):
    output=output.resolve();output.mkdir(parents=True,exist_ok=False)
    results=[]
    for budget,override in [(0,False),(137,True),(17500,False),(250000,True)]:
        root=output/str(budget);tasks=root/'tasks collection';out=root/'output directory'
        tasks.mkdir(parents=True);out.mkdir()
        for name,instruction,reward in [('a','Calculate 2+3.',1),('b','Calculate 6*7.',2)]:
            (tasks/name).mkdir()
            (tasks/name/'task.json').write_text(json.dumps({'task_id':name,'task_type':'math',
                'instruction':instruction,'reward':reward,'check':'exact_int'}))
        protected={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in tasks.glob('*/task.json')}
        cmd=[*docker_prefix('none'),'-v',f'{tasks}:/data/tasks:rw','-v',f'{out}:/out:rw',
             '-e',f'BUDGET={budget}','-e','OUT=/out/answers.txt']
        if override:
            cmd+=['--workdir','/app','--entrypoint','python',image,'/app/run_agent.py','--tasks-dir','/data/tasks']
        else:
            cmd+=['-e','TASKS=/data/tasks',image]
        if budget==137:cmd+=['--budget','100000']
        done=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=60)
        (root/'run.log').write_text(done.stdout)
        events=[json.loads(l) for l in done.stdout.splitlines() if l.startswith('{')]
        start=next(e for e in events if e.get('event')=='run_start')
        end=next(e for e in events if e.get('event')=='run_end')
        assert done.returncode==0,(budget,done.returncode)
        assert start['budget_tenths']==budget*10,(budget,start)
        assert end['spent_tenths']==end['unknown_reserved_tenths']==0
        assert end['remaining_tenths']==budget*10 and end['submitted']==2 and end['out_complete']
        assert (out/'answers.txt').read_text()=='a\t5\nb\t42\n'
        assert (tasks/'a/answer.txt').read_text()=='5\n' and (tasks/'b/answer.txt').read_text()=='42\n'
        assert all(hashlib.sha256(p.read_bytes()).hexdigest()==digest for p,digest in protected.items())
        assert not any(e.get('event')=='llm' for e in events)
        results.append({'budget':budget,'entrypoint':'organizer override' if override else 'image default',
                        'budget_source':start['budget_source'],'passed':True,'api_calls':0})
    (output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);parser.add_argument('--image',required=True)
    args=parser.parse_args();smoke(args.image,args.output)
