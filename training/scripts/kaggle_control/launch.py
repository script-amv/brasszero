# SPDX-License-Identifier: Apache-2.0
"""Kaggle private script. Only hash-pinned source is executed."""
import hashlib
import json
import shutil
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import time

started = time.time()
manifest_path, = Path('/kaggle/input').rglob('run-input.json')
if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != '__DESCRIPTOR_SHA256__':
    raise ValueError('unexpected private Kaggle input')
manifest = json.loads(manifest_path.read_bytes())
with tempfile.TemporaryDirectory(prefix='brass-legacy-') as tmp:
    scratch = Path(tmp)
    source = scratch/'src'
    archive = manifest_path.parent/'source.bin'
    if archive.stat().st_size > 64*1024**2:
        raise ValueError('oversized source input')
    if hashlib.sha256(archive.read_bytes()).hexdigest() != manifest['files']['source.bin']:
        raise ValueError('source pin mismatch')
    seen = set()
    expanded = 0
    budget = min(256*1024**2,shutil.disk_usage(scratch).free-128*1024**2)
    with tarfile.open(archive) as tar:
        for member in tar:
            name = Path(member.name)
            expanded += member.size
            if len(seen) >= 10000 or name in seen or expanded > budget or member.size > 32*1024**2:
                raise ValueError('oversized or duplicate source member')
            seen.add(name)
            if name.is_absolute() or '..' in name.parts or not (member.isdir() or member.isfile()):
                raise ValueError('unexpected source member')
            tar.extract(member, source)
    entry = source/'training/scripts/kaggle_control/job.py'
    subprocess.run([sys.executable,entry,manifest_path,scratch,str(started)],check=True)
