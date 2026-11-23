"""Lazy optional-runtime command entry point; help works without torch installed."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import signal
import time
from typing import List, Optional

from .config import TrainConfig


class _SearchStop:
    def __init__(self):
        self.requested = False

    def request(self):
        self.requested = True

    def __call__(self):
        return self.requested


@contextmanager
def _graceful_signals(stop):
    previous = {}
    try:
        for number in (signal.SIGINT, signal.SIGTERM):
            previous[number] = signal.signal(number, lambda signum, frame: stop.request())
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def main(argv: Optional[List[str]] = None):
    invocation_started = time.monotonic()
    parser = argparse.ArgumentParser(prog='brass_rl')
    sub = parser.add_subparsers(dest='cmd', required=True)
    library = sub.add_parser('library', help='reconcile or adopt exact local library occurrences')
    ops = library.add_subparsers(dest='operation', required=True)
    repair = ops.add_parser('reconcile')
    repair.add_argument('--run', dest='runs', action='append', type=Path)
    accept = ops.add_parser('adopt')
    accept.add_argument('--checkpoint', type=Path, required=True)
    accept.add_argument('--expected-tip', required=True, help='exact predecessor policy ID, or none for an empty chain')
    cloud = ops.add_parser('ingest-kaggle')
    cloud.add_argument('--verified', type=Path, required=True)
    for op in (repair, accept, cloud):
        op.add_argument('--project-root', required=True, type=Path)
    retention = sub.add_parser('retention', help='audit checkpoint retention; default is read-only')
    retention.add_argument('--runs-root', type=Path, default=Path('training/runs'))
    retention.add_argument('--apply', action='store_true', help='pin historical endpoints and retire eligible Adam files')
    coverage = sub.add_parser('transport-audit', help='verify transport contents against independent local bytes')
    coverage.add_argument('--project-root', type=Path, default=Path.cwd())
    coverage.add_argument('--output', type=Path, required=True)
    coverage.add_argument('--recover', action='store_true', help='copy missing bytes to hash-named recovery objects')
    runtime = sub.add_parser('runtime', help='check optional numerical runtime')
    runtime.add_argument('--device', choices=('cpu', 'cuda', 'mps'), default='cpu')
    bc = sub.add_parser('train-bc', help='activate or resume eight-game BC with independent critic and loss-priority')
    route = bc.add_mutually_exclusive_group(required=True)
    route.add_argument('--source', type=Path, help='full numeric BC source checkpoint for a new experiment')
    route.add_argument('--resume', choices=['latest'], help='resume the latest complete supported BC boundary')
    route.add_argument('--legacy', action='store_true', help='continue the historical main-bc line from BC788')
    bc.add_argument('--out', type=Path, help='supported experiment directory (required for --source or --resume)')
    bc.add_argument('--max-updates', type=int, required=True, help='absolute completed-update ceiling')
    bc.add_argument('--max-wall-time-seconds', type=float, help='training wall budget for a new --source activation')
    bc.add_argument('--experimental', action='store_true', help='record experimental intent for the new BC child')
    bc.add_argument('--dynamic-league', action='store_true', help='activate the ranked main-history league')
    training = sub.add_parser('train', help='train/resume complete PPO update boundaries')
    training.add_argument('--config', type=Path, required=True)
    training.add_argument('--out', type=Path, required=True)
    training.add_argument('--resume', metavar='latest|PATH')
    training.add_argument('--max-updates', type=int, default=None, help='absolute completed-update ceiling')
    branch = sub.add_parser('branch', help='new full-state experiment from explicit historical checkpoint')
    branch.add_argument('--checkpoint', type=Path, required=True)
    branch.add_argument('--out', type=Path, required=True)
    adopt = sub.add_parser('adopt', help='add provenance to a trusted historical run without rewriting its payloads')
    adopt.add_argument('--run', type=Path, required=True)
    adopt.add_argument('--parent-checkpoint', type=Path)
    package = sub.add_parser('pack', help='package every update/model/replay and all ancestor experiments')
    package.add_argument('--run', type=Path, required=True)
    package.add_argument('--out', type=Path, required=True)
    package.add_argument('--replay-root', type=Path)
    check = sub.add_parser('verify', help='verify experiment package integrity and completeness without loading pickle')
    check.add_argument('package', type=Path)
    importing = sub.add_parser('import', help='verify and add an experiment package to this project')
    importing.add_argument('package', type=Path)
    importing.add_argument('--project-root', type=Path, default=Path.cwd())
    evaluation = sub.add_parser('eval', help='evaluate a preselected policy on an immutable unseen suite')
    evaluation.add_argument('--checkpoint', type=Path, required=True)
    evaluation.add_argument('--suite', type=Path, required=True)
    evaluation.add_argument('--out', type=Path, required=True)
    profiling = sub.add_parser('profile', help='measure fresh disposable collection/PPO without modifying checkpoints')
    profiling.add_argument('--config', type=Path, required=True)
    profiling.add_argument('--games', type=int, required=True)
    profiling.add_argument('--out', type=Path, required=True)
    arena = sub.add_parser('arena', help='register checkpoints then run the existing match/play CLI')
    arena.add_argument('--agent', action='append', required=True, metavar='NAME=CHECKPOINT')
    arena.add_argument('--device', choices=('cpu', 'cuda', 'mps'), default='cpu')
    arena.add_argument('--mode', choices=('sample', 'argmax'), default='sample')
    arena.add_argument('arguments', nargs=argparse.REMAINDER, help='-- <existing brass_ai arguments>')
    search_eval = sub.add_parser('search-eval', help='bounded immutable paired frozen-search evaluation')
    search_eval.add_argument('--checkpoint', type=Path)
    search_eval.add_argument('--config', type=Path)
    search_eval.add_argument('--out', type=Path, required=True)
    search_eval.add_argument('--master-seed', type=int)
    search_eval.add_argument('--report-only', action='store_true', help='recover/inspect existing run without gameplay allowance')
    search_arena = sub.add_parser('search-arena', help='register frozen search then run existing terminal CLI')
    search_arena.add_argument('--agent', action='append', required=True, metavar='NAME=CHECKPOINT')
    search_arena.add_argument('--config', type=Path, required=True)
    search_arena.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        if args.cmd == 'library':
            from .library import adopt_mainline, ingest_kaggle_return, reconcile_library
            if args.operation == 'reconcile':
                result = reconcile_library(args.project_root, runs=args.runs)
            elif args.operation == 'adopt':
                expected = None if args.expected_tip == 'none' else args.expected_tip
                result = adopt_mainline(args.project_root, args.checkpoint, expected_tip=expected)
            else:
                result = ingest_kaggle_return(args.verified, args.project_root)
            print(json.dumps(result, indent=2, allow_nan=False))
        elif args.cmd == 'transport-audit':
            from .transport_coverage import audit
            result = audit(args.project_root, args.output, recover=args.recover)
            print(json.dumps({'manifest': str(args.output/'manifest.json'),
                              'archives': len(result['archives']),
                              'missing': sum(row['coverage'].get('missing', 0) for row in result['archives'])}))
        elif args.cmd == 'retention':
            from contextlib import ExitStack
            from .experiments import writer_lock
            from .retention import apply_plan, inspect_tree, plan_runs, summary
            runs = inspect_tree(args.runs_root)
            if args.apply and not runs:
                raise ValueError('no checkpoint runs found under '+str(args.runs_root))
            with ExitStack() as locks:
                if args.apply:
                    for run in sorted(runs):
                        locks.enter_context(writer_lock(run))
                plan = plan_runs(runs)
                result = summary(plan)
                if args.apply:
                    apply_plan(plan, pin_endpoints=True)
                print(json.dumps(dict(result, applied=args.apply), indent=2))
        elif args.cmd == 'search-eval':
            from brass_search import SearchConfig
            if not args.report_only and (args.checkpoint is None or args.config is None):
                raise ValueError('search-eval requires --checkpoint and --config unless --report-only')
            config = None if args.report_only else SearchConfig.from_json(args.config)
            if config is not None and config.wall_time_seconds is not None:
                raise ValueError('fixed-work evaluation forbids per-move wall cutoff')
            import secrets
            from .seeding import SeedStreams
            existing = args.out/'suite.json'
            master = args.master_seed
            if master is None:
                master = json.loads(existing.read_text())['master_seed'] if existing.exists() else secrets.randbits(64)
            SeedStreams(master)
            from .search_budget import RunInvocation
            stop = _SearchStop()
            with _graceful_signals(stop):
                with RunInvocation(args.out, report_only=args.report_only, started=invocation_started) as invocation:
                    from .search_evaluate import evaluate_search
                    report = evaluate_search(args.checkpoint, config, args.out, master_seed=master,
                                             stop_requested=stop, report_only=args.report_only, announce=True,
                                             _invocation_started=invocation_started, _invocation=invocation)
                    invocation.report = report
                    print(json.dumps(report, indent=2, allow_nan=False))
        elif args.cmd in ('pack', 'verify', 'import'):
            from .transfer import pack, verify, import_package
            if args.cmd == 'pack':
                result = pack(args.run, args.out, replay_root=args.replay_root)
            elif args.cmd == 'verify':
                result = verify(args.package)
            else:
                result = import_package(args.package, args.project_root)
            print(json.dumps(result, indent=2, allow_nan=False))
        elif args.cmd == 'branch':
            from .experiments import branch_experiment
            print(json.dumps(branch_experiment(args.checkpoint, args.out), indent=2))
        elif args.cmd == 'adopt':
            from .experiments import adopt_experiment
            result = adopt_experiment(args.run, args.parent_checkpoint)
            print(json.dumps(result, indent=2))
        elif args.cmd == 'runtime':
            from .runtime import runtime_check
            print(json.dumps(runtime_check(args.device), indent=2, allow_nan=False))
        elif args.cmd in ('train', 'train-bc'):
            from .train import StopFlag, train
            stop = StopFlag()
            with _graceful_signals(stop):
                if args.cmd == 'train-bc':
                    if args.legacy:
                        if args.experimental:
                            parser.error('--experimental requires activation with --source')
                        if args.max_wall_time_seconds is not None:
                            parser.error('--max-wall-time-seconds requires activation with --source')
                        if args.out is not None:
                            parser.error('--legacy uses the historical main-bc run directory')
                        from .main_bc import continue_bc
                        report = continue_bc(args.max_updates, stop, dynamic=args.dynamic_league)
                    else:
                        if args.out is None or args.dynamic_league:
                            parser.error('supported BC requires --out and uses the loss-priority league')
                        from .main_bc import train_bc
                        report = train_bc(args.out, args.max_updates, stop, source=args.source,
                                          max_wall_time_seconds=args.max_wall_time_seconds, experimental=args.experimental)
                else:
                    config = TrainConfig.from_json(args.config)
                    resume = args.out/'latest' if args.resume == 'latest' else Path(args.resume) if args.resume else None
                    report = train(config, args.out, resume=resume, max_updates=args.max_updates, stop_requested=stop)
            print(json.dumps({'latest_checkpoint': str(report.latest_checkpoint),
                              'latest_policy': str(report.latest_policy),
                              'completed_updates': report.completed_updates, 'next_game_id': report.next_game_id,
                              'stop_reason': report.stop_reason, 'rng_restore_notes': report.rng_restore_notes},
                             indent=2, allow_nan=False))
        elif args.cmd == 'profile':
            from dataclasses import asdict
            # Exclusive creation precedes any expensive run, preventing clobber of
            # existing artifacts, directories, and symlink targets.
            for parent in (args.out.resolve(), *args.out.resolve().parents):
                if (parent/'manifest.json').is_file() and (parent/'policy.pt').is_file():
                    raise ValueError('published checkpoint directory exists; choose an output outside it')
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open('x', encoding='utf-8') as output:
                from .profile import profile_pipeline
                report = profile_pipeline(TrainConfig.from_json(args.config), args.games)
                json.dump(asdict(report), output, indent=2, allow_nan=False)
                output.write('\n')
            print(json.dumps({'profile': str(args.out)}, indent=2))
        elif args.cmd == 'eval':
            from dataclasses import asdict
            from .evaluate import EvalSuite, evaluate
            from .config import _from_mapping
            import torch
            torch.set_num_threads(1)
            if torch.get_num_interop_threads() != 1:
                torch.set_num_interop_threads(1)
            suite = _from_mapping(EvalSuite, json.loads(args.suite.read_text()), 'evaluation suite')
            report = evaluate(args.checkpoint, suite, args.out)
            print(json.dumps({'candidate_id': report.candidate_id, 'suite_id': report.suite_id,
                              'summaries': {name: asdict(summary) for name, summary in report.summaries.items()},
                              'strength_gate': report.strength_gate}, indent=2, allow_nan=False))
        else:
            definitions = []
            names = set()
            from brass_ai.agents import is_known
            for definition in args.agent:
                name, separator, path = definition.partition('=')
                if not separator or not name or not path:
                    raise ValueError('--agent requires NAME=CHECKPOINT')
                if name in names or is_known(name):
                    raise ValueError('checkpoint agent name is already registered: '+name)
                names.add(name)
                definitions.append((name, Path(path)))
            arguments = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
            if not arguments:
                raise ValueError('arena requires -- <existing brass_ai arguments>')
            from brass_ai.__main__ import main as arena_main
            if args.cmd == 'search-arena':
                from brass_search import SearchConfig
                config = SearchConfig.from_json(args.config)
                from brass_search import register_search_checkpoint
                from brass_ai.replay_archive import recording_context
                from brass_search.identity import player_manifest
                from .policy_io import inspect_policy
                import torch
                torch.set_num_threads(1)
                manifests = {}
                for name, path in definitions:
                    manifests[name] = player_manifest(inspect_policy(path), config)
                    register_search_checkpoint(name, path, config)
                provenance = {'category': 'terminal', 'search_players': manifests}
                if arguments[0] == 'play':
                    import uuid
                    provenance['session'] = uuid.uuid4().hex
                with recording_context(**provenance):
                    arena_main(arguments)
            else:
                from .agent import register_checkpoint
                for name, path in definitions:
                    register_checkpoint(name, path, device=args.device, mode=args.mode)
                arena_main(arguments)
    except ImportError as exc:
        parser.error('Install the optional RL dependencies in a separate training environment: %s' % exc)
    except (OSError, ValueError, RuntimeError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
