# SPDX-License-Identifier: Apache-2.0
"""Stdlib-only trusted source provenance, separate from portable policy identity."""
import os
from pathlib import Path
import re
import subprocess

SOURCE_ROOT = Path(__file__).resolve().parents[2]


def source_provenance():
    """Use this source checkout, or an explicitly operator-declared archive SHA.

    Never search an archive's parent repositories or use the caller's cwd. An
    archive declaration is provenance metadata, not a binary/source attestation.
    """
    if (SOURCE_ROOT/'.git').exists():
        try:
            revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE_ROOT,
                                               text=True, stderr=subprocess.DEVNULL).strip()
            dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'],
                                            cwd=SOURCE_ROOT, text=True, stderr=subprocess.DEVNULL)
            return {'kind': 'git', 'revision': revision, 'dirty': bool(dirty)}
        except (OSError, subprocess.CalledProcessError):
            return {'kind': 'unavailable', 'revision': None}
    declared = os.environ.get('BRASS_SOURCE_REVISION')
    if declared is not None:
        if not re.fullmatch(r'[0-9a-f]{40}', declared):
            raise ValueError('BRASS_SOURCE_REVISION must be the recorded 40-character lowercase archive commit SHA')
        return {'kind': 'archive-declared', 'revision': declared}
    return {'kind': 'unavailable', 'revision': None}
