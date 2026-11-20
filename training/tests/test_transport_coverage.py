# SPDX-License-Identifier: Apache-2.0
import io
import tarfile
import zipfile

import pytest


def test_archive_recovery_hashes_and_safe_names(tmp_path):
    from brass_rl.transport_coverage import audit
    root = tmp_path/'training/runs'
    root.mkdir(parents=True)
    (root/'known.json').write_bytes(b'known')
    with zipfile.ZipFile(root/'source.zip', 'w') as archive:
        archive.writestr('history/known.json', b'known')
        archive.writestr('history/missing.json', b'only in archive')
    output = tmp_path/'evidence'
    report = audit(tmp_path, output, recover=True)
    assert report['archives'][0]['coverage'] == {'local': 1, 'recovered': 1}
    h = report['recovered'][0]['sha256']
    assert (output/'objects'/h).read_bytes() == b'only in archive'
    with tarfile.open(root/'bad.tar.gz', 'w:gz') as archive:
        member = tarfile.TarInfo('../escape')
        archive.addfile(member, io.BytesIO(b''))
    with pytest.raises(ValueError, match='unsafe'):
        audit(tmp_path, output, recover=True)
