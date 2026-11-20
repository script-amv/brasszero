# SPDX-License-Identifier: Apache-2.0
"""Prove transport payload availability without requiring the bundles at runtime."""
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import tarfile
import tempfile
import zipfile

from .experiments import atomic, encoded, sync_dir
from .retention import file_sha, present_payloads
from .transfer import safe_path


def is_transport(path):
    return path.name.endswith(('.tar.gz', '.tgz', '.tar', '.zip'))


@dataclass
class Limits:
    member: int = 512 * 1024**2
    total: int = 64 * 1024**3
    count: int = 150000


def _stream_bundle(path, visitor, limits):
    """Stream regular members and resolve only earlier, verified in-archive hard links."""
    names, rows = set(), {}
    size = 0
    def check(name, length):
        nonlocal size
        safe_path(name)
        if '\x00' in name or (len(name) >= 2 and name[1] == ':'):
            raise ValueError('unsafe archive member name')
        if name in names or length < 0 or length > limits.member:
            raise ValueError('duplicate or oversized archive member: '+name)
        names.add(name)
        size += length
        if size > limits.total or len(names) > limits.count:
            raise ValueError('archive exceeds audit limits')
    if path.suffix == '.zip':
        with zipfile.ZipFile(path) as archive:
            for item in archive.infolist():
                name = item.filename.rstrip('/')
                check(name, item.file_size)
                if item.is_dir():
                    continue
                mode = stat.S_IFMT(item.external_attr >> 16)
                if mode not in (0, stat.S_IFREG):
                    raise ValueError('unsafe zip member type: '+name)
                with archive.open(item) as src:
                    rows[name] = visitor(name, item.file_size, src)
    else:
        with tarfile.open(path, 'r|*') as archive:
            for item in archive:
                check(item.name, item.size)
                if item.isdir():
                    continue
                if item.islnk():
                    safe_path(item.linkname)
                    if item.linkname not in rows:
                        raise ValueError('unsafe archive hard link: '+item.linkname)
                    rows[item.name] = rows[item.linkname]
                elif item.isfile():
                    with archive.extractfile(item) as src:
                        rows[item.name] = visitor(item.name, item.size, src)
                else:
                    raise ValueError('unsafe archive member type: '+item.name)
    return rows


def _hash_stream(src, size, copy=None):
    digest = hashlib.sha256()
    remaining = size
    while remaining:
        chunk = src.read(min(1024*1024, remaining))
        if not chunk:
            raise ValueError('truncated archive member')
        digest.update(chunk)
        if copy:
            copy.write(chunk)
        remaining -= len(chunk)
    return digest.hexdigest()


def _local_files(project, output):
    seen, local, retired, transports = {}, {}, {}, []
    def inspect(path):
        if path.is_symlink() or not path.is_file() or output in path.parents:
            return
        if is_transport(path):
            if project/'training/runs' in path.parents:
                transports.append(path)
            return
        info = path.stat()
        key = info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns
        if key not in seen:
            seen[key] = file_sha(path)
        local.setdefault(seen[key], path.relative_to(project).as_posix())
        if path.name == 'retention.json' and path.parent.parent.name == 'updates':
            manifest = json.loads((path.parent/'manifest.json').read_bytes())
            if 'optimizer.pt' not in present_payloads(path.parent, manifest):
                retired[manifest['content_hashes']['optimizer.pt']] = path.relative_to(project).as_posix()
    for top in ('training/runs', 'players/models', 'replays/games', 'replays/history'):
        for parent, dirs, files in os.walk(project/top):
            dirs[:] = sorted(d for d in dirs if not d.startswith('.') and Path(parent)/d != output)
            for name in sorted(files):
                inspect(Path(parent)/name)
    if (project/'.git').exists():
        for name in subprocess.check_output(['git', 'ls-files', '-z'], cwd=project).split(b'\x00'):
            if name:
                inspect(project/os.fsdecode(name))
    return local, retired, sorted(set(transports))


def audit(project, output, *, recover=False, limits=None):
    project, output = Path(project).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    limits = limits or Limits()
    local, retired, archives = _local_files(project, output)
    hashes = {p: file_sha(p) for p in archives}
    containers = {h: p.relative_to(project).as_posix() for p, h in hashes.items()}
    report = {'format': 'brass-transport-coverage/1', 'archives': [], 'recovered': []}
    for path in archives:
        relative = path.relative_to(project).as_posix()
        archive_sha = hashes[path]
        package = {}
        def hash_member(name, size, src):
            if name == 'manifest.json':
                if size > 32*1024**2:
                    raise ValueError('oversized package manifest')
                data = src.read(size)
                if len(data) != size:
                    raise ValueError('truncated package manifest')
                header = json.loads(data)
                if header.get('format') == 'brass-experiment-package/1':
                    package.update(header)
                return hashlib.sha256(data).hexdigest(), size
            return _hash_stream(src, size), size
        members = _stream_bundle(path, hash_member, limits)
        logical = dict(members)
        if package:
            logical = {'manifest.json': members['manifest.json']}
            used = {'manifest.json'}
            for name, row in package['files'].items():
                safe_path(name)
                sha = row['sha256']
                member = 'blobs/'+sha
                if members.get(member) != (sha, row['size']):
                    raise ValueError('content-addressed archive member mismatch: '+name)
                logical[name] = members[member]
                used.add(member)
            if used != set(members):
                raise ValueError('undeclared archive members')
        coverage, evidence, missing = Counter(), [], set()
        for name, (sha, size) in sorted(logical.items()):
            destination = local.get(sha)
            if destination is not None:
                status = 'local'
            elif sha in containers and is_transport(PurePosixPath(name)):
                status, destination = 'nested_transport', containers[sha]
            elif sha in retired and PurePosixPath(name).name == 'optimizer.pt':
                status, destination = 'retired_optimizer', retired[sha]
            else:
                obj = output/'objects'/sha
                if obj.is_symlink() or obj.parent.is_symlink():
                    raise ValueError('unsafe recovery object path')
                if obj.exists():
                    if file_sha(obj) != sha:
                        raise ValueError('recovery object hash mismatch: '+sha)
                    status, destination = 'recovered', 'objects/'+sha
                elif recover:
                    status, destination = 'recovered', 'objects/'+sha
                    missing.add(sha)
                else:
                    status, destination = 'missing', None
            coverage[status] += 1
            evidence.append([name, sha, size, status, destination])
            if status == 'recovered':
                report['recovered'].append({'archive': relative, 'member': name, 'sha256': sha, 'bytes': size})
        if missing:
            objects = output/'objects'
            objects.mkdir(exist_ok=True)
            def recover_member(name, size, src):
                sha = members[name][0]
                if sha not in missing:
                    return members[name]
                with tempfile.NamedTemporaryFile(dir=objects, delete=False) as dest:
                    temp = Path(dest.name)
                    try:
                        if _hash_stream(src, size, dest) != sha:
                            raise ValueError('archive changed during recovery')
                        dest.flush()
                        os.fsync(dest.fileno())
                        os.link(temp, objects/sha)
                    finally:
                        temp.unlink(missing_ok=True)
                missing.remove(sha)
                return members[name]
            _stream_bundle(path, recover_member, limits)
            if missing:
                raise ValueError('failed to recover required archive objects')
            sync_dir(objects)
        coverage_name = 'coverage-'+archive_sha+'.jsonl'
        index = b''.join(encoded(row) for row in evidence)
        atomic(output/coverage_name, index)
        report['archives'].append({'path': relative, 'bytes': path.stat().st_size,
            'sha256': archive_sha, 'logical_members': len(logical), 'coverage': dict(coverage),
            'coverage_index': coverage_name, 'coverage_index_sha256': hashlib.sha256(index).hexdigest()})
    atomic(output/'manifest.json', encoded(report))
    return report
