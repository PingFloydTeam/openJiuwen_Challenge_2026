"""Exercise the actual contest image entrypoints with variable BUDGET and OUT."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from .experiments import docker_prefix


def smoke(image, output, cpus=4):
    output=output.resolve();output.mkdir(parents=True,exist_ok=False)
    results=[]
    for budget,override in [(0,False),(137,True),(17500,False),(250000,True)]:
        root=output/str(budget);tasks=root/'tasks collection';out=root/'output directory'
        tasks.mkdir(parents=True);out.mkdir()
        # Match the organizer promise: mounts writable even with cap-drop ALL.
        tasks.chmod(0o777);out.chmod(0o777)
        for name,instruction,reward in [('a','Calculate 2+3.',1),('b','Calculate 6*7.',2)]:
            (tasks/name).mkdir()
            (tasks/name).chmod(0o777)
            (tasks/name/'task.json').write_text(json.dumps({'task_id':name,'task_type':'math',
                'instruction':instruction,'reward':reward,'check':'exact_int'}))
        # Qualification coding example, with expected removed before agent access.
        shutil.copytree(Path(__file__).resolve().parents[1]/'tests/fixtures/coding',tasks/'c')
        for directory in [tasks/'c',*(p for p in (tasks/'c').rglob('*') if p.is_dir())]:
            directory.chmod(0o777)
        protected={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in tasks.rglob('*')
                   if p.is_file() and (p.name=='task.json' or p.name.startswith('test'))}
        prefix=docker_prefix('none')
        prefix[prefix.index('--cpus')+1]=str(cpus)
        cmd=[*prefix,'-v',f'{tasks}:/data/tasks:rw','-v',f'{out}:/out:rw',
             '-e',f'BUDGET={budget}','-e','OUT=/out/answers.txt']
        if override:
            cmd+=['--workdir','/app','--entrypoint','python',image,'/app/run_agent.py','--tasks-dir','/data/tasks']
        else:
            cmd+=['-e','TASKS=/data/tasks',image]
        if budget==137:cmd+=['--budget','100000']
        done=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=60)
        (root/'run.log').write_text(done.stdout)
        assert done.returncode==0,(budget,done.returncode,done.stdout[-6000:])
        events=[json.loads(l) for l in done.stdout.splitlines() if l.startswith('{')]
        start=next(e for e in events if e.get('event')=='run_start')
        end=next(e for e in events if e.get('event')=='run_end')
        assert start['budget_tenths']==budget*10,(budget,start)
        assert end['spent_tenths']==end['unknown_reserved_tenths']==0
        assert end['remaining_tenths']==budget*10 and end['submitted']==3 and end['out_complete']
        assert (out/'answers.txt').read_text()=='a\t5\nb\t42\nc\tDONE\n'
        assert (tasks/'a/answer.txt').read_text()=='5\n' and (tasks/'b/answer.txt').read_text()=='42\n'
        assert (tasks/'c/answer.txt').read_text()=='DONE\n'
        assert all(hashlib.sha256(p.read_bytes()).hexdigest()==digest for p,digest in protected.items())
        assert not any(e.get('event')=='llm' for e in events)
        # Ordinary unittest is independent of the harness's validation worker.
        judge=subprocess.run([*prefix,'-v',f'{tasks}/c/workspace:/case:ro','--workdir','/case',
                              '--entrypoint','python',image,'-B','test_mylib.py'],
                             stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=25)
        (root/'judge.log').write_text(judge.stdout)
        assert judge.returncode==0,(budget,judge.stdout)
        results.append({'budget':budget,'entrypoint':'organizer override' if override else 'image default',
                        'budget_source':start['budget_source'],'passed':True,'api_calls':0,
                        'coding_independent_passed':True,'tasks':3})
    (output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);parser.add_argument('--image',required=True)
    parser.add_argument('--cpus',type=int,choices=range(1,5),default=4)
    args=parser.parse_args();smoke(args.image,args.output,args.cpus)
