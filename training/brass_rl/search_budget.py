"""Single-writer, crash-conservative cumulative controller allowance."""
import fcntl
import json
import math
from pathlib import Path
import time

from .replay import atomic_json


class RunLock:
    def __init__(self, out):
        self.out = Path(out)
        self.file = None

    def __enter__(self):
        self.out.mkdir(parents=True, exist_ok=True)
        self.file = (self.out/'.writer.lock').open('a+b')
        try:
            fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.file.close()
            raise RuntimeError('another search evaluation writer is active') from exc
        return self

    def __exit__(self, *args):
        self.file.close()


class WallBudget:
    FORMAT = 'brass-search-budget/1'

    def __init__(self, out, limit):
        if type(limit) not in (int, float) or not math.isfinite(limit) or limit <= 0:
            raise ValueError('wall_seconds must be finite and positive')
        self.path = Path(out)/'budget.json'
        self.limit = float(limit)
        self.started = None
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            if not isinstance(self.state, dict) or self.state.get('format') != self.FORMAT:
                raise ValueError('unrecognized cumulative allowance')
            if self.state.get('limit_seconds') != self.limit:
                raise ValueError('immutable wall allowance mismatch')
        else:
            self.state = dict(format=self.FORMAT, limit_seconds=self.limit,
                              remaining_seconds=self.limit, measured_seconds=0., reservation=None)
            atomic_json(self.path, self.state)
        self.remaining = self.state.get('remaining_seconds')
        if type(self.remaining) not in (int, float) or not 0 <= self.remaining <= self.limit:
            raise ValueError('invalid remaining allowance')

    def can_start_suite(self):
        measured = self.state.get('measured_seconds')
        return ('reservation' in self.state and self.state['reservation'] is None
                and self.remaining > 0 and type(measured) in (int, float) and measured >= 0
                and math.isclose(self.remaining + measured, self.limit))

    def reserve(self, now=None):
        if self.started is not None:
            raise RuntimeError('allowance already reserved')
        self.started = time.monotonic() if now is None else now
        self.reserved = self.remaining
        self.state.update(remaining_seconds=0., reservation={'seconds': self.reserved})
        atomic_json(self.path, self.state)  # durable before model load or actual play

    def elapsed(self):
        return time.monotonic()-self.started

    def exhausted(self):
        return self.elapsed() >= self.reserved

    def snapshot(self, now=None):
        elapsed = max(0., (time.monotonic() if now is None else now)-self.started)
        return dict(self.state, remaining_seconds=max(0., self.reserved-elapsed), reservation=None,
                    measured_seconds=self.state['measured_seconds']+elapsed)

    def finish(self, now=None):
        self.state = self.snapshot(now)
        self.remaining = self.state['remaining_seconds']
        atomic_json(self.path, self.state)
        self.started = None


class RunInvocation:
    """Lightweight CLI/API boundary: reserve before importing optional inference.

    The owner keeps this context through final JSON/report bookkeeping. Only a
    clean exit refunds; an unaccounted process exit retains the full reservation.
    """
    def __init__(self, out, limit=1800, *, report_only=False, started=None):
        self.out = Path(out)
        self.limit = limit
        self.report_only = report_only
        self.started = time.monotonic() if started is None else started
        self.budget = None
        self.report = None
        self.lock = None

    def __enter__(self):
        resolved = self.out.resolve()
        if any((p/'manifest.json').is_file() and (p/'policy.pt').is_file()
               for p in (resolved, *resolved.parents)):
            raise ValueError('output is inside a published checkpoint; choose an independent run directory')
        if self.report_only and not (self.out/'suite.json').exists():
            raise ValueError('report-only requires an existing suite')
        self.lock = RunLock(self.out)
        self.lock.__enter__()
        try:
            if not self.report_only:
                exists = (self.out/'suite.json').exists()
                if exists and not (self.out/'budget.json').exists():
                    raise ValueError('missing cumulative allowance; fail closed without recreating budget')
                if not exists:
                    for path in self.out.iterdir():
                        if path.name == '.writer.lock':
                            continue
                        if path.name != 'budget.json' or path.is_symlink() or not path.is_file():
                            raise ValueError('new suite output contains unrelated files')
                self.budget = WallBudget(self.out, self.limit)
                if not exists and not self.budget.can_start_suite():
                    raise ValueError('new suite requires a clean cumulative allowance with time remaining')
                if self.budget.remaining > 0:
                    self.budget.reserve(now=self.started)
            return self
        except BaseException:
            self.lock.__exit__(None, None, None)
            raise

    def __exit__(self, *args):
        try:
            if self.budget is not None and self.budget.started is not None:
                self.budget.finish()
                if self.report is not None:
                    self.report['budget'] = dict(self.budget.state)
        finally:
            self.lock.__exit__(*args)
