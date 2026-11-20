# SPDX-License-Identifier: Apache-2.0
"""Fixed school-688 diagnostic. Run once on one isolated Kaggle GPU host."""
from contextlib import ExitStack
from dataclasses import asdict, replace
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import sys
import time
from unittest.mock import patch

import torch
from brass_ai.arena import RefereeGame
from brass_ai.replay_archive import Archive, recording_context
from brass_rl import train as controller
from brass_rl.config import TrainConfig
from brass_rl.replay import atomic_json
from brass_rl.runtime import runtime_check


def schedule():
    for pair in range(4):
        for device in (('cpu', 'cuda') if pair % 2 == 0 else ('cuda', 'cpu')):
            yield pair, device


def stage(function, name, timings, sync, clock=time.perf_counter):
    def call(*args, **kwargs):
        sync()
        start = clock()
        result = function(*args, **kwargs)
        sync()
        timings[name] = (start, clock())
        return result
    return call


def trial(parent, output, manifest, pair, device):
    name = ('warmup' if pair == 0 else 'pair-%d' % pair) + '-' + device
    run = output/name
    shutil.copytree(parent, run)
    os.environ['BRASS_REPLAY_ROOT'] = str(run/'validation')
    parent_info = dict(experiment_id=manifest['parent_experiment'], update=688,
                       checkpoint_sha256=manifest['parent_manifest'], model_id=manifest['parent_model'])
    # Each diagnostic owns a new identity and only the two required parent boundaries.
    atomic_json(run/'experiment.json', dict(format='brass-experiment/1', id='bench-20261119-'+name,
                mode='diagnostic', parent=parent_info, base_update=688,
                historical_gaps=['Only the parent and required development-best boundary were staged.']))
    original = TrainConfig.from_json(run/'config.json')
    assert original.games_per_update == 2 and original.runtime.torch_threads == 1
    config = replace(original, runtime=replace(original.runtime, device=device))
    atomic_json(run/'treatment.json', asdict(config))
    counts = dict(decisions=0, candidates=0)
    check = RefereeGame._check_index
    def count_move(referee, seat, obs, index):
        check(referee, seat, obs, index)
        counts['decisions'] += 1
        counts['candidates'] += obs.num_legal()
    def validation(**metadata):
        return recording_context(**dict(metadata, category='validation'))
    sync = torch.cuda.synchronize if device == 'cuda' else lambda: None
    gc.collect()
    if device == 'cuda':
        # The CPU parent has no CUDA generator state. All CUDA trials start alike.
        torch.cuda.manual_seed(20261119)
        torch.cuda.reset_peak_memory_stats()
    times = {}
    with ExitStack() as hooks:
        for attribute in ('collect_batch', 'update_policy', 'retain_training_replays'):
            fn = getattr(controller, attribute)
            hooks.enter_context(patch.object(controller, attribute, stage(fn, attribute, times, sync)))
        hooks.enter_context(patch.object(RefereeGame, '_check_index', count_move))
        hooks.enter_context(patch.object(controller, 'recording_context', validation))
        result = controller.train(config, run, run/'updates/000688', max_updates=689)
    assert result.preflight['model_id'] == manifest['parent_model']
    assert result.preflight['completed_training_games'] == 1376 and result.preflight['last_completed_update'] == 688
    assert result.completed_updates == 689 and result.next_game_id == 1378
    assert result.model.device.type == device
    assert sum(p.numel() for p in result.model.parameters()) == 390342
    records = Archive().records()
    assert len(records) == 2 and all(r['category']=='validation' and r['status']=='completed' for r in records)
    assert sum(len(r['moves']) for r in records) == counts['decisions']
    metric, = result.state.metrics
    assert metric['ppo']['optimizer_steps'] > 0
    collect_start, collect_end = times['collect_batch']
    ppo_start, ppo_end = times['update_policy']
    row = dict(name=name, pair=pair, device=device, games=2, **counts, ppo=metric['ppo'],
               parent=parent_info, preflight=result.preflight,
               actual_model_device=str(result.model.device), parameters=390342,
               games_manifest=result.state.manifests['train'],
               checkpoint=str(result.latest_checkpoint.relative_to(output)),
               peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
               peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0,
               peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved() if device=='cuda' else 0,
               collection_s=collect_end-collect_start, ppo_s=ppo_end-ppo_start,
               total_update_s=times['retain_training_replays'][1]-collect_start)
    atomic_json(run/'measurement.json', row)
    print(json.dumps({k:row[k] for k in ('name','games','decisions','candidates','collection_s','ppo_s','total_update_s')}), flush=True)
    return row['total_update_s']


def main(parent, output, manifest_path, deadline):
    assert platform.system() == 'Linux' and os.environ['CUDA_VISIBLE_DEVICES']=='0'
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    assert not any(output.glob('*-cpu')) and not any(output.glob('*-cuda'))
    manifest = json.loads(manifest_path.read_bytes())
    for name, expected in manifest['inputs'].items():
        assert hashlib.sha256((parent/name).read_bytes()).hexdigest() == expected
    torch.set_num_interop_threads(1)
    torch.set_num_threads(1)
    host = dict(python=platform.python_version(), torch=str(torch.__version__), cuda=torch.version.cuda,
                device=torch.cuda.get_device_name(0), visible_devices=torch.cuda.device_count(),
                omp_threads=os.environ['OMP_NUM_THREADS'], torch_threads=torch.get_num_threads(),
                interop_threads=torch.get_num_interop_threads(), affinity=sorted(os.sched_getaffinity(0)),
                probes={device:runtime_check(device) for device in ('cpu','cuda')})
    assert torch.__version__.split('+')[0]=='2.2.2' and torch.version.cuda=='12.1'
    assert host['probes']['cpu']['versions']['numpy']=='1.26.4'
    atomic_json(output/'numerical-host.json', host)
    longest = 90.
    for pair,device in schedule():
        logical = sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
        if logical + 256*2**20 > 8*2**30 or time.monotonic()+1.5*longest+90 >= deadline:
            print('Stopped at sample boundary for wall-time/output reserve.', flush=True)
            break
        longest = max(longest, trial(parent, output, manifest, pair, device))


if __name__ == '__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), float(sys.argv[4]))
