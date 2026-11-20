# SPDX-License-Identifier: Apache-2.0
"""Build an explicit private, single-boundary compute input from committed source."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

from brass_rl.dynamic_league import mainline_rows
from brass_rl.experiments import encoded
from brass_rl.league_study import BASE, PEERS, SOURCE_SHA, declaration
from brass_rl.model_inventory import SECTIONS


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(project, output, account):
    root = Path(__file__).resolve().parents[2]
    dataset, kernel = output/'dataset', output/'kernel'
    dataset.mkdir(parents=True)
    kernel.mkdir()
    revision = subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    with (dataset/'source.bin').open('xb') as stream:
        subprocess.run(['git','archive','--format=tar.gz',revision,'Cargo.lock','Cargo.toml','engine/core',
                        'engine/bindings','engine/tools','players/baselines','training/brass_rl',
                        'training/configs/gae-study.json','training/requirements/linux.txt'],cwd=root,stdout=stream,check=True)
    parent = project/'training/runs/main-bc'
    checkpoint = parent/'updates/000793'
    if digest(checkpoint/'manifest.json') != SOURCE_SHA:
        raise ValueError('source manifest differs from declared BC793')
    catalog_path = project/'players/models/catalog.json'
    catalog = json.loads(catalog_path.read_bytes())
    selected = mainline_rows([r for k in SECTIONS for r in catalog.get(k,[])],BASE)
    required = {r['update']:r for r in selected if r['update']%20==0 or r['policy_id'] in PEERS}
    if set(required)!=set(range(0,BASE+1,20))|{688,738,793}:
        raise ValueError('missing study history or peer')
    files = {f'parent/{p.relative_to(parent).as_posix()}':p for p in checkpoint.iterdir()}
    files.update({'parent/'+name:parent/name for name in ('config.json','experiment.json')})
    for ref in json.loads((checkpoint/'manifest.json').read_bytes())['snapshots']:
        name = 'weights/'+ref['sha256']+'.pt'
        files['parent/'+name] = parent/name
    rows = []
    for n,row in sorted(required.items()):
        prefix = 'players/models/'+row['policy_id'].split(':')[-1] if n%20==0 else 'peers/'+str(n)
        rows.append(dict(row,path=prefix))
        for name,sha in row['files'].items():
            path = project/row['path']/name
            if digest(path)!=sha:
                raise ValueError('modified catalog source '+str(path))
            files[prefix+'/'+name] = path
    in_memory = {'players/models/catalog.json':encoded(dict(models=rows)),
                 'protocol.json':encoded(declaration())}
    with tarfile.open(dataset/'inputs.bin','x:gz',compresslevel=2) as tar:
        tar.dereference = True
        for name,path in sorted(files.items()):
            info = tarfile.TarInfo(name)
            info.size = path.stat().st_size
            with path.open('rb') as stream:
                tar.addfile(info,stream)
        for name,data in in_memory.items():
            item = tarfile.TarInfo(name)
            item.size = len(data)
            tar.addfile(item,io.BytesIO(data))
    manifest = dict(format='brass-league-study-input/1',revision=revision,source_catalog_sha256=digest(catalog_path),
                    files={p.name:digest(p) for p in dataset.glob('*.bin')},
                    logical_inputs={k:digest(p) for k,p in files.items()},
                    protocol=declaration(),scope='one full parent boundary, its league, historical cadence and final peers')
    (dataset/'league-input.json').write_bytes(encoded(manifest))
    prefix = account+'/brass-league-'
    (dataset/'dataset-metadata.json').write_bytes(encoded(dict(
        id=prefix+'input-20261120',title='Brass league study input 20261120',licenses=[dict(name='other')],
        description='Private study input. Original rights retained.')))
    wrapper = subprocess.check_output(['git','show',revision+':training/scripts/kaggle_league.py'],cwd=root,text=True)
    (kernel/'run.py').write_text(wrapper.replace('__INPUT_SHA256__',digest(dataset/'league-input.json')).replace('__RECOVERY_SHA256__','fresh'))
    (kernel/'kernel-metadata.json').write_bytes(encoded(dict(id=prefix+'study-20261120',
        title='Brass league study 20261120',code_file='run.py',language='python',kernel_type='script',
        is_private=True,enable_gpu=False,enable_tpu=False,enable_internet=True,
        dataset_sources=[prefix+'input-20261120'],competition_sources=[],kernel_sources=[],model_sources=[])))
    print(json.dumps(dict(revision=revision,manifest=digest(dataset/'league-input.json'),
                          hashes=manifest['files'],boundaries=1,members=40),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--account',required=True)
    args=parser.parse_args()
    prepare(args.project.resolve(),args.output.resolve(),args.account)
