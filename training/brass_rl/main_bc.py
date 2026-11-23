# SPDX-License-Identifier: Apache-2.0
"""Normal supported BC training and the explicit historical BC788 continuation."""
import hashlib
from pathlib import Path

from . import experiments
from .config import TrainConfig

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path('training/runs/combination-study-2026-11-20/BC/updates/000788')
DESTINATION = Path('training/runs/main-bc')
CURRENT_DESTINATION = Path('training/runs/bc-current')
PROFILE = Path('training/configs/main-bc.json')
SOURCE_HASH = 'e4712626768a553d67eb9917949b5de1c039b951e4df77ef2bb3c0058fed2cb0'
SOURCE_ID = 'policy-sha256-v1:3466c7765c2d9fb8abb6ae249469aec8b1cefc1c3593279fa47f0ad14f74d4a1'


def source_parent():
    return {'experiment_id': 'exp-32ad838df2584406bada24369079e35f',
            'update': 788, 'model_id': SOURCE_ID, 'checkpoint_sha256': SOURCE_HASH}


def prepare_main_run(root):
    run = root/DESTINATION
    if not run.exists():
        source = root/SOURCE
        if hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest() != SOURCE_HASH:
            raise ValueError('BC 788 source manifest differs from the selected training boundary')
        experiments.branch_experiment(source, run)
    if experiments.read_experiment(run).get('parent') != source_parent():
        raise ValueError('main BC run has a different parent checkpoint')
    base = run/'updates/000788/manifest.json'
    if hashlib.sha256(base.read_bytes()).hexdigest() != SOURCE_HASH:
        raise ValueError('main BC run does not contain its original full-state boundary')
    return run


def continue_bc(ceiling, stop, *, dynamic=False):
    import torch
    from .train import train

    if ceiling < 788:
        raise ValueError('train-bc requires an absolute update ceiling of at least 788')
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    config = TrainConfig.from_json(ROOT/PROFILE)
    run = prepare_main_run(ROOT)
    if TrainConfig.from_json(run/'config.json') != config:
        raise ValueError('main BC run configuration differs from its inherited BC profile')
    members = None
    if dynamic:
        from functools import partial
        from .dynamic_league import catalog_members
        # Resolve the actual source boundary under train's writer lock.
        members = partial(catalog_members, ROOT)
    return train(config, run, run/'latest', max_updates=ceiling, stop_requested=stop, dynamic_members=members)


def train_bc(run, ceiling, stop, *, source=None):
    from .bc_current import fork_bc
    from .config import BC_LEARNER
    from .train import train
    import torch

    run = Path(run)
    if ceiling < 0 or (source is not None and Path(source).name.isdecimal() and ceiling < int(Path(source).name)):
        raise ValueError('absolute update ceiling precedes the requested boundary')
    if source is not None:
        torch.set_num_threads(1)
        fork_bc(source, run)
    cfg = TrainConfig.from_json(run/'config.json')
    if cfg.learner != BC_LEARNER:
        raise ValueError('train-bc requires an activated supported BC experiment')
    return train(cfg, run, run/'latest', max_updates=ceiling, stop_requested=stop)
