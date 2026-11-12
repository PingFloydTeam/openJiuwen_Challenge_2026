"""Explicit macOS experiment adapter, never imported by the contest entry point.

Uses the SAME test adapter for a frozen baseline and candidate. Separate grading
process loads the oracle; agent only receives the tasks subtree. This does not
validate Linux seccomp, image size, or the competition Docker launch.
"""
import argparse
import asyncio
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
WORKER = Path(__file__).with_name('macos_worker.py')


def install_adapter(source):
    import ping_floyd.coding as coding
    def test_run(root, script=''):
        root = Path(root).resolve()
        # Deny network and writes except the bounded captured test output. Deny
        # user/temp data reads except this stage, Python and fixed worker scripts.
        allowed = [root, Path(sys.base_prefix), Path(sys.prefix), WORKER,
                   source / 'ping_floyd' / 'test_worker.py']
        profile = '(version 1)(allow default)(deny network*)(deny file-write*)(deny file-read* (subpath "/Users") (subpath "/Volumes") (subpath "/private/tmp") (subpath "/private/var/folders"))(allow file-read-metadata)'
        for path in allowed:
            profile += '(allow file-read* (subpath '+json.dumps(str(path.resolve()))+'))'
        profile += '(allow file-read* (subpath "/System") (subpath "/usr/lib") (subpath "/private/var/db") (literal "/dev/null") (literal "/dev/urandom"))'
        with tempfile.NamedTemporaryFile(prefix="pf-test-output-") as out:
            profile += '(allow file-write* (literal '+json.dumps(str(Path(out.name).resolve()))+'))'
            try:
                result = subprocess.run(['/usr/bin/sandbox-exec','-p',profile,sys.executable,'-I','-S',str(WORKER),
                                         str(source/'ping_floyd'/'test_worker.py'),script],
                                        cwd=root, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                                        env={'LANG':'C.UTF-8'}, timeout=8)
            except subprocess.TimeoutExpired:
                return {'passed':False,'isolation_error':False,'output':'test timeout'}
            size=out.tell(); out.seek(max(0,size-2500)); tail=out.read().decode(errors='replace')
        ran=any(int(n)>0 for n in re.findall(r'Ran (\d+) tests?',tail))
        return {'passed': result.returncode==0 and ran, 'isolation_error':'sandbox_apply:' in tail, 'output':tail.strip(), 'returncode':result.returncode}
    coding.test_run = test_run


def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['run','grade','audit','smoke'])
    p.add_argument('suite',type=Path)
    p.add_argument('--source',type=Path,default=ROOT)
    p.add_argument('--output',type=Path)
    p.add_argument('--budget',type=int,default=100000)
    p.add_argument('--offline',action='store_true')
    a=p.parse_args(); source=a.source.resolve()
    sys.path.insert(0,str(source))
    install_adapter(source)
    if a.mode=='smoke':
        from ping_floyd.coding import test_run
        print(test_run(a.suite)); return
    if a.mode in {'grade','audit'}:
        sys.path.append(str(ROOT))
        from bench.suite import grade, private_coding_passes
        if a.mode=='grade':
            grade(a.suite)
        else:
            meta=json.loads((a.suite/'metadata.json').read_text())['cases']
            checked={name:private_coding_passes(a.suite,name,entry,reference=True)
                     for name,entry in meta.items() if entry.get('reference_files')}
            print(json.dumps(checked)); assert all(checked.values()), 'reference failure'
        return
    if not a.output:
        p.error('run needs --output')
    root=a.output.resolve(); root.mkdir(parents=True,exist_ok=False)
    shutil.copytree(a.suite/'tasks',root/'tasks')
    spec=importlib.util.spec_from_file_location('contest_entry',source/'run_agent.py')
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    sys.argv=['run_agent.py','--tasks-dir',str(root/'tasks'),'--out',str(root/'answers.txt'),
              '--budget',str(a.budget),'--workers','1','--seconds','3500']
    if a.offline: sys.argv.append('--offline')
    started=time.monotonic()
    with (root/'run.jsonl').open('w') as log, contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        asyncio.run(mod.main(mod.options()))
    (root/'runtime.json').write_text(json.dumps({'elapsed_s':time.monotonic()-started,'source':str(source),
        'platform':'macOS sandbox experiment (not competition Linux)','budget':a.budget})+'\n')
    print(json.dumps({'finished':str(root),'seconds':round(time.monotonic()-started,2)}),flush=True)


if __name__=='__main__': main()
