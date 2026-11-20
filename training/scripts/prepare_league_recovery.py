# SPDX-License-Identifier: Apache-2.0
"""Seal a stopped provider session's output for continuation of its original arms."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile


def prepare(original, prior, output):
    kernel_meta = json.loads((original/'kernel/kernel-metadata.json').read_bytes())
    original_input = (original/'dataset/league-input.json').read_bytes()
    input_sha = hashlib.sha256(original_input).hexdigest()
    with tarfile.open(prior) as tar:
        names = set(tar.getnames())
        def data(name):
            return tar.extractfile('league-return/'+name).read()
        if data('input-provenance.json')!=original_input:
            raise ValueError('prior session does not belong to the original launch')
        if 'league-return/study/results.json' in names:
            raise ValueError('completed scientific results exist; verify and collect them')
        arms = {arm:json.loads(data('study/'+arm+'/experiment.json'))['id'] for arm in ('old','new')
                if 'league-return/study/'+arm+'/experiment.json' in names}
        if 'league-return/study/arm-identities.json' in names and json.loads(data('study/arm-identities.json'))!=arms:
            raise ValueError('partial session has lost a sealed branch')
    archive_sha = hashlib.sha256(prior.read_bytes()).hexdigest()
    dataset, kernel = output/'dataset', output/'kernel'
    dataset.mkdir(parents=True)
    kernel.mkdir()
    desc = json.dumps(dict(input_sha256=input_sha,archive_sha256=archive_sha,arms=arms),sort_keys=True).encode()
    (dataset/'league-recovery.json').write_bytes(desc)
    # Both destination payloads are immutable and distinct from source artifacts.
    try:
        os.link(prior,dataset/'prior-return.bin')
    except OSError:
        shutil.copyfile(prior,dataset/'prior-return.bin')
    slug = kernel_meta['id'].split('/')[0]+'/brass-league-recovery-'+archive_sha[:12]
    (dataset/'dataset-metadata.json').write_text(json.dumps(dict(id=slug,title='Brass league recovery '+archive_sha[:12],
                licenses=[dict(name='other')],description='Private continuation of the sealed BC793 study.')))
    script = (original/'kernel/run.py').read_text()
    old = "RECOVERY_SHA = 'fresh'"
    if script.count(old)!=1 or input_sha not in script:
        raise ValueError('unexpected original scientific wrapper')
    (kernel/'run.py').write_text(script.replace(old,"RECOVERY_SHA = '%s'"%hashlib.sha256(desc).hexdigest()))
    kernel_meta['dataset_sources'].append(slug)
    (kernel/'kernel-metadata.json').write_text(json.dumps(kernel_meta))
    return dict(archive_sha256=archive_sha,arms=arms,original_input=input_sha,recovery_dataset=slug)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('original','prior','output'):
        parser.add_argument(name,type=Path)
    args=parser.parse_args()
    print(json.dumps(prepare(args.original,args.prior,args.output),indent=2))
