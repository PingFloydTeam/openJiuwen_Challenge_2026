"""Audit persisted outputs and accounting independently of the agent's claims."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path


def events(path):
    result=[]
    for line in path.read_text().splitlines():
        try: e=json.loads(line)
        except ValueError: continue
        if isinstance(e,dict):result.append(e)
    return result


def audit(root):
    results=[]
    for out in sorted(root.iterdir()):
        if not out.is_dir() or not (out/'summary.json').exists():continue
        summary=json.loads((out/'summary.json').read_text());es=events(out/'run.jsonl')
        grade=events(out/'grade.jsonl')[-1]
        end=next((e for e in reversed(es) if e.get('event')=='run_end'),{})
        llms=[e for e in es if e.get('event')=='llm']
        known=sum(e.get('cost_tenths',0) for e in llms);unknown=sum(e.get('unknown_reserved_tenths',0) for e in llms)
        assert known==end.get('spent_tenths'),out
        assert unknown==end.get('unknown_reserved_tenths'),out
        assert known+unknown <= json.loads((out/'runtime.json').read_text())['budget']*10,out
        assert end['in_flight_tenths']==0 if 'in_flight_tenths' in end else True
        rows=[]
        for answer in sorted((out/'tasks').glob('*/answer.txt')):
            task=json.loads((answer.parent/'task.json').read_text())
            assert 'expected' not in task
            def esc(v):return v.replace('\\','\\\\').replace('\t','\\t').replace('\n','\\n').replace('\r','\\r')
            rows.append((answer.parent.name,esc(task['task_id'])+'\t'+esc(answer.read_text().removesuffix('\n'))+'\n'))
        assert (out/'answers.txt').read_text()==''.join(line for _,line in sorted(rows)),out
        split='development' if out.name.startswith('development-') else 'official' if out.name.startswith('official-') else 'transfer'
        original=root/split
        for file in (original/'tasks').glob('*/workspace/test*.py'):
            assert file.read_bytes()==(out/'tasks'/file.relative_to(original/'tasks')).read_bytes(),file
        source=Path(summary['runtime']['source'])
        hashfile=root/'baseline-sha256.json' if source.name=='baseline' else source.with_suffix('.sha256.json')
        hashes=json.loads(hashfile.read_text())
        assert all(hashlib.sha256((source/n).read_bytes()).hexdigest()==h for n,h in hashes.items())
        # Check all original input manifests and every input-suite hash, where available.
        for manifest in (original/'tasks').glob('*/task.json'):
            assert manifest.read_bytes()==(out/'tasks'/manifest.relative_to(original/'tasks')).read_bytes()
        if (root/(split+'-sha256.json')).exists():
            hs=json.loads((root/(split+'-sha256.json')).read_text())
            assert all(hashlib.sha256((original/n).read_bytes()).hexdigest()==h for n,h in hs.items())
        per_family=defaultdict(Counter)
        meta=json.loads((out/'metadata.json').read_text())['cases']
        for event in events(out/'grade.jsonl'):
            if 'grade' in event and event['grade'] in meta:
                c=per_family[meta[event['grade']]['family']];c[event['outcome']]+=1;c['points']+=event['reward']
        summary.update(agent_summary=end,grade=grade,verified_weighted_tokens=known/10,
                       unknown_reserve=unknown/10,input_tokens=sum(e.get('input_tokens',0) for e in llms),
                       output_tokens=sum(e.get('output_tokens',0) for e in llms),
                       checks={'token_accounting':True,'out_matches_answers':True,'public_tests_unchanged':True,
                               'task_manifests_unchanged':True,'source_snapshot_unchanged':True,'input_suite_unchanged':True},
                       by_family=dict(per_family))
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');results.append(summary)
    (root/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps([{'run':r['label'],'score':r['grade']['score'],'max':r['grade']['max_score'],
                       'solved':r['grade']['correct_answers'],'wrong':r['grade']['wrong_answers'],
                       'false_accepts':r['grade']['false_accepts'],'tokens':r['verified_weighted_tokens'],
                       'reserve':r['unknown_reserve']} for r in results],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);audit(p.parse_args().root)
