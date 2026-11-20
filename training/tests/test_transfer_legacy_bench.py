# SPDX-License-Identifier: Apache-2.0
"""Legacy benchmark experiments remain addressable during dataset relocation."""
import json
import pytest
from brass_rl.transfer import experiments_under

def test_registers_historical_benchmarks_and_rejects_collisions(tmp_path):
    identity = 'bench-20261119-pair-3-cuda'
    def write(name, experiment):
        directory = tmp_path/name
        directory.mkdir()
        (directory/'experiment.json').write_text(json.dumps({'format':'brass-experiment/1','id':experiment}))
        return directory
    one = write('original', identity)
    assert experiments_under(tmp_path) == {identity:one}
    other = write('collision', identity)
    with pytest.raises(ValueError, match='duplicate'):
        experiments_under(tmp_path)
    (other/'experiment.json').write_text(json.dumps({'format':'brass-experiment/1','id':'bench-20261119-pair-4-cuda'}))
    with pytest.raises(ValueError, match='invalid'):
        experiments_under(tmp_path)
