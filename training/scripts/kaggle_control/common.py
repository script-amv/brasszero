# SPDX-License-Identifier: Apache-2.0
"""Trusted, hash-pinned continuation payloads and the run-scoped HTTPS protocol."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tarfile
import urllib.request

# The issuer pins the destination locally. Remote jobs receive that same pin in
# the hash-bound descriptor before they read the individual capability.
def trusted_endpoint():
    configured = os.environ.get('BRASS_CONTROL_ENDPOINT')
    local = Path(__file__).resolve().parents[3] / 'lab/control-endpoint.txt'
    if not configured and local.is_file():
        configured = local.read_text().strip()
    if not configured or not re.fullmatch(r'https://[A-Za-z0-9.-]+(?::[0-9]+)?', configured):
        raise ValueError('configure a trusted HTTPS BRASS_CONTROL_ENDPOINT')
    return configured
WHEEL_NAME = 'torch-2.2.2+cpu-cp310-cp310-linux_x86_64.whl'
WHEEL_SHA = '02c4fac3c964e73f5f49003e0060c697f73b67c10cc23f51c592facb29e1bd53'
BASE = 843
SOURCE = 'bfa24dfaab8a8e855094302648b44d9835615e7d7d9c2d8688d1658286eda011'
POLICY = 'policy-sha256-v1:90dbf8191eb8c3ff0577d66210b713a119ff8f08d9d9c8695d540b4e3095ae75'


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, data):
    path = Path(path)
    with path.with_suffix('.tmp').open('xb') as stream:
        stream.write((json.dumps(data, sort_keys=True)+'\n').encode())
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(path.with_suffix('.tmp'), path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def extract(path, dest, max_bytes=10*1024**3, max_files=100000):
    """Extract only ordinary files/directories and prior internal file hardlinks."""
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.iterdir()):
        raise ValueError('extraction requires an empty destination')
    seen, files = set(), set()
    expanded = 0
    allowable = min(max_bytes, shutil.disk_usage(dest).free-128*1024**2)
    if Path(path).stat().st_size > max_bytes:
        raise ValueError('archive exceeds the permitted input size')
    with tarfile.open(path) as tar:
        for item in tar:
            name = Path(item.name)
            if len(seen) >= max_files or len(item.name) > 1024 or item.size > 128*1024**2:
                raise ValueError('archive member count or size exceeded')
            expanded += item.size
            if expanded > allowable:
                raise ValueError('archive exceeds extraction space limit')
            if name.is_absolute() or not name.parts or '..' in name.parts or name in seen:
                raise ValueError('invalid archive path')
            if item.islnk():
                if Path(item.linkname) not in files:
                    raise ValueError('invalid archive hardlink')
            elif not (item.isdir() or item.isfile()):
                raise ValueError('unsupported archive member')
            tar.extract(item, dest, set_attrs=False)
            seen.add(name)
            if not item.isdir():
                files.add(name)


def seal(root, output):
    temp = output.parent/('.'+output.name+'.partial')
    with tarfile.open(temp, 'x:gz', compresslevel=1) as archive:
        archive.add(root, arcname='state')
    with temp.open('rb') as stream:
        os.fsync(stream.fileno())
    os.replace(temp, output)
    return digest(output)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args):
        return None


def control(cap, action, progress=None):
    endpoint = trusted_endpoint()
    if cap['endpoint'] != endpoint or not re.fullmatch('[a-f0-9]{32}', cap['run']):
        raise ValueError('unrecognized control endpoint or run')
    data = None if action in ('status','poll') else json.dumps(progress or {}).encode()
    req = urllib.request.Request(endpoint+'/runs/'+cap['run']+'/'+action, data=data,
            headers={'Authorization':'Bearer '+cap['token'], 'Content-Type':'application/json',
                     'User-Agent':'BrassTrainingControl/1.0'})
    with urllib.request.build_opener(NoRedirect).open(req, timeout=4) as response:
        result = json.loads(response.read(2048))
    if action != 'progress' and result.get('run') != cap['run']:
        raise ValueError('control response run mismatch')
    return result
