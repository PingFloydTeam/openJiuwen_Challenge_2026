"""Judge coding via ordinary unittest in one clean Docker container per task.

Does not import the harness worker or contract generators. Only the one task
and its private tests are mounted; no oracle, credentials or other tasks.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

from .experiments import docker_prefix


def audit(suite, image, reference=False):
    metadata=json.loads((suite/'metadata.json').read_text())['cases']
    oracle=json.loads((suite/'oracle.json').read_text())
    results=[]
    for name,entry in sorted(metadata.items()):
        if entry['kind']!='coding' or oracle[name] is None:
            continue
        workspace=suite/'tasks'/name/'workspace'
        protected=all((workspace/p).is_file() and hashlib.sha256((workspace/p).read_bytes()).hexdigest()==h
                      for p,h in entry['public_tests'].items())
        with tempfile.TemporaryDirectory(prefix='pf-independent-') as folder:
            stage=Path(folder)/'task';shutil.copytree(workspace,stage,symlinks=True)
            if any(p.is_symlink() for p in stage.rglob('*')):
                raise ValueError('judge refuses task symlinks')
            judge=suite/'judge'/name
            shutil.copyfile(judge/'test_private.py',stage/'test_pf_private_check.py')
            if reference:
                for dst,src in entry['reference_files'].items():
                    shutil.copyfile(judge/src,stage/dst)
            # Copy only this task to tmpfs so legitimate tests may create files.
            code=("import shutil,os,unittest; shutil.copytree('/task','/tmp/case'); "
                  "os.chdir('/tmp/case'); "
                  "r=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.discover('.')); "
                  "raise SystemExit(not (r.wasSuccessful() and r.testsRun>0))")
            cmd=[*docker_prefix('none'),'-v',f'{stage}:/task:ro','--entrypoint','python',image,'-B','-c',code]
            try:
                proc=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=25)
                ok=protected and proc.returncode==0;output=proc.stdout[-1600:]
            except subprocess.TimeoutExpired:
                ok=False;output='independent judge timeout'
        result={'task':name,'passed':ok,'protected':protected,'reference':reference,'output':output}
        results.append(result);print(json.dumps(result),flush=True)
    print(json.dumps({'independent_passed':sum(e['passed'] for e in results),'total':len(results),'reference':reference}),flush=True)
    return results


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('suite',type=Path);p.add_argument('--image',required=True)
    p.add_argument('--reference',action='store_true');a=p.parse_args()
    audit(a.suite.resolve(),a.image,a.reference)
