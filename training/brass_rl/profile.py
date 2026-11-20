# SPDX-License-Identifier: Apache-2.0
"""Bounded, single-writer instrumentation of the real collection/PPO/journal path.

Runs fresh disposable training; never accepts a published checkpoint as output.
Hooks are process-wide for the duration: call from an otherwise idle process.
"""
from contextlib import ExitStack
from dataclasses import asdict, dataclass, replace
import platform
from pathlib import Path
import resource
import subprocess
import tempfile
import time
from typing import Dict
from unittest.mock import patch

import numpy as np
import torch

from brass_ai.arena import RefereeGame
from . import agent, train as controller
from .artifacts import capture_rng_states, restore_rng_states
from .config import TrainConfig
from .model import BrassPolicy
from .rollout import CollectionMemoryError
from .seeding import SeedStreams
from .source import source_provenance


@dataclass(frozen=True)
class ProfileReport:
    games: int
    decisions: int
    training_decisions: int
    candidates_total: int
    candidates_median: float
    candidates_p95: float
    candidates_max: int
    largest_position: dict
    games_per_second: float
    decisions_per_second: float
    wall_seconds: float
    stage_seconds: Dict[str, float]
    peak_rss_mib: float
    peak_cuda_allocated_mib: float
    parameter_count: int
    optimizer_steps: int
    disk_bytes: Dict[str, int]
    versions: dict
    hardware: dict
    config: dict
    source_revision: str
    notes: tuple


def _version(command):
    root = Path(__file__).resolve().parents[2]
    if command == ['git', 'rev-parse', 'HEAD']:
        return source_provenance()['revision'] or 'unavailable (source archive; record archive revision separately)'
    try:
        return subprocess.check_output(command, cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unavailable'


def profile_pipeline(config: TrainConfig, games: int) -> ProfileReport:
    """Measure one real update over exactly games complete games (including PPO).

    Collection timings partition encoding, neural forward/memory, and other work.
    PPO is inclusive of its re-encoding, backward and Adam. Controller/IO is the
    residual outside collection/update. CUDA timers synchronize at both edges.
    RSS is process lifetime high-water (native allocations included), not a delta.
    Disk counts logical file bytes before disposable cleanup, including update 0.
    """
    if not isinstance(config, TrainConfig) or type(games) is not int or games < 1:
        raise ValueError('profile requires TrainConfig and games >= 1')
    if config.eval.interval_updates:
        raise ValueError('profile requires eval.interval_updates=0; measure evaluation separately')
    if config.runtime.max_wall_time_seconds is not None:
        raise ValueError('profile requires no training wall cutoff to finish the measured update')
    runtime = config.runtime
    if runtime.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('cuda device is unavailable')
    config = replace(config, games_per_update=games, updates=1)
    previous_rng = capture_rng_states()
    previous_threads = torch.get_num_threads()
    # Interop can only be set before work starts. Never silently claim one thread.
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.set_num_threads(runtime.torch_threads)
    if runtime.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    positions = {}
    timings = dict(encoding=0., inference=0., collection=0., ppo_update=0.)
    master = config.master_seed
    # train generates its own high-entropy master if not supplied; select once so
    # native deal IDs can be mapped without exposing them in the public report.
    if master is None:
        import secrets
        master = secrets.randbits(64)
        config = replace(config, master_seed=master)
    streams = SeedStreams(master)
    game_ids = {streams.seed('train.deal',i):i for i in range(games)}
    pending_original = RefereeGame.pending

    def pending(referee):
        result = pending_original(referee)
        key = (game_ids[referee._seed], referee._ply)
        positions[key] = result[1].num_legal()  # apply_index rechecks; count once.
        return result

    def sync():
        if runtime.device == 'cuda':
            torch.cuda.synchronize()
        elif runtime.device == 'mps':
            torch.mps.synchronize()

    def timed(function, name):
        def call(*args, **kwargs):
            sync()
            started = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                sync()
                timings[name] += time.perf_counter()-started
        return call

    collect = controller.collect_batch
    def collection(*args, **kwargs):
        # Instrument only collection. PPO's own forwards/encodes belong to update.
        with ExitStack() as hooks:
            for module, name in ((agent,'encode_observation'),(agent,'_encode_transition')):
                hooks.enter_context(patch.object(module,name,timed(getattr(module,name),'encoding')))
            for name in ('forward','update_memory'):
                hooks.enter_context(patch.object(BrassPolicy,name,timed(getattr(BrassPolicy,name),'inference')))
            return timed(collect,'collection')(*args, **kwargs)

    started = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix='brass-profile-') as temporary:
            path = Path(temporary)/'run'
            with patch.object(RefereeGame,'pending',pending), \
                 patch.object(controller,'collect_batch',collection), \
                 patch.object(controller,'update_policy',timed(controller.update_policy,'ppo_update')):
                try:
                    result = controller.train(config,path)
                except CollectionMemoryError as exc:
                    if positions:
                        (game,ply), count = max(positions.items(),key=lambda item:item[1])
                        context = 'game=%d, ply=%d, candidates=%d' % (game,ply,count)
                    else:
                        context = 'none observed (failed before first position)'
                    raise CollectionMemoryError('%s; largest position: %s' % (exc,context)) from exc
            sync()
            wall = time.perf_counter()-started
            disk = {name:sum(p.stat().st_size for p in (path/name).rglob('*') if p.is_file())
                    for name in ('updates','policies','weights','replays')}
            disk['total'] = sum(p.stat().st_size for p in path.rglob('*') if p.is_file())
            parameters = sum(p.numel() for p in result.model.parameters())
            metrics = result.state.metrics[0]['ppo']
    finally:
        restore_rng_states(previous_rng)
        torch.set_num_threads(previous_threads)
    counts = list(positions.values())
    (game,ply), largest = max(positions.items(),key=lambda item:item[1])
    stages = {'collection_encoding':timings['encoding'], 'collection_inference':timings['inference'],
              'collection_other':timings['collection']-timings['encoding']-timings['inference'],
              'ppo_update':timings['ppo_update'],
              'controller_io':wall-timings['collection']-timings['ppo_update']}
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mib = rss/(1024**2 if platform.system() == 'Darwin' else 1024)
    return ProfileReport(games,len(counts),metrics['value_decisions'],sum(counts),float(np.median(counts)),
        float(np.percentile(counts,95)),largest,{'game_id':game,'ply':ply,'candidates':largest},
        games/wall,len(counts)/wall,wall,stages,rss_mib,
        torch.cuda.max_memory_allocated()/1024**2 if runtime.device == 'cuda' else 0.,
        parameters,metrics['optimizer_steps'],disk,
        {'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,
         'rustc':_version(['rustc','--version'])},
        {'platform':platform.platform(),'processor':platform.processor(),'machine':platform.machine(),
         'device':runtime.device,'torch_threads':runtime.torch_threads,
         'interop_threads':torch.get_num_interop_threads(),
         'cuda_device':torch.cuda.get_device_name() if runtime.device == 'cuda' else None},
        # Deliberately omit the private master/deal setup from shareable reports.
        {k:v for k,v in asdict(config).items() if k != 'master_seed'},
        _version(['git','rev-parse','HEAD']),
        ('one instrumented update; no throughput threshold; no raw-engine speed claim',
         'RSS is process lifetime high-water; memory_budget_mib is collector estimate only',
         'disk logical bytes include initial and trained update/export; historical files are not GC bounded',
         'interop thread setting is process-wide; hooks require an idle single-writer process'))
