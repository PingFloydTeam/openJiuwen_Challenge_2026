"""Small stratified experiment: 32 cases, four kinds, 25% underdetermined."""
import argparse
import json
from pathlib import Path
import random
import shutil
import tempfile

from .intelligence import build


def prepare(destination, seed):
    destination.mkdir(parents=True,exist_ok=False)
    rng=random.Random(seed)
    with tempfile.TemporaryDirectory() as d:
        source=Path(d)/'pool';build(source,False,seed)
        metadata=json.loads((source/'metadata.json').read_text())
        cases=metadata['cases'];oracle=json.loads((source/'oracle.json').read_text())
        selected=[]
        for kind in ('math','search','data_analysis','coding'):
            solvable=sorted(n for n,e in cases.items() if e['kind']==kind and oracle[n] is not None)
            families=sorted({cases[n]['family'] for n in solvable})
            chosen=[rng.choice([n for n in solvable if cases[n]['family']==family]) for family in families]
            chosen+=rng.sample([n for n in solvable if n not in chosen],6-len(chosen))
            chosen+=rng.sample(sorted(n for n,e in cases.items() if e['kind']==kind and oracle[n] is None),2)
            selected+=chosen
        for n in sorted(selected):
            shutil.copytree(source/'tasks'/n,destination/'tasks'/n)
            if (source/'judge'/n).is_dir():shutil.copytree(source/'judge'/n,destination/'judge'/n)
        (destination/'oracle.json').write_text(json.dumps({n:oracle[n] for n in sorted(selected)},indent=2)+'\n')
        (destination/'metadata.json').write_text(json.dumps({'seed':seed,'selection':'six solvable including all families, two underdetermined per kind','cases':{n:cases[n] for n in sorted(selected)}},indent=2)+'\n')
    print(json.dumps({'selected':len(selected),'unsolvable':8,'seed':seed}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('destination',type=Path);p.add_argument('--seed',type=int,default=2026111302)
    a=p.parse_args();prepare(a.destination,a.seed)
