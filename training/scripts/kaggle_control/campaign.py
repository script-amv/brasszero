# SPDX-License-Identifier: Apache-2.0
"""Mac-only coordination across a recovered bootstrap attempt, its retry and one long segment."""
from contextlib import contextmanager
import fcntl

from common import digest, read, write


@contextmanager
def lock(directory):
    with (directory/'.owner.lock').open('a+b') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        yield


def member(directory, job):
    return dict(run=job['run'],directory=directory.name)


def campaign(directory, job):
    path = directory.parent/'campaign-handoff.json'
    if not path.exists():
        return None
    state = read(path)
    if member(directory,job) not in state['members']:
        return None
    if job['deadline'] != state['deadline'] or job['initial']['experiment'] != state['experiment']:
        raise ValueError('campaign member metadata changed')
    return state


def effective(base, state):
    active = state['members'][-1]
    path = base/active['directory']
    job = read(path/'job.json')
    if (path.parent != base or job['run'] != active['run'] or job['deadline'] != state['deadline']
            or job['initial']['experiment'] != state['experiment']):
        raise ValueError('campaign active member identity mismatch')
    return path


def halted(base, state):
    return bool(state.get('halted_by') or any((base/m['directory']/'operator-halt.json').exists()
                                              for m in state['members']))


def adopt_retry(original, retry):
    original, retry = original.resolve(),retry.resolve()
    base = original.parent
    if retry.parent != base or original == retry:
        raise ValueError('retry and original must be distinct siblings')
    with lock(base):
        first, second = read(original/'job.json'),read(retry/'job.json')
        returned = read(original/'verified/verified.json')
        budget = read(base/'campaign.json')
        if (first['stage'] != 'proof' or second['stage'] != 'proof' or first['run'] == second['run'] or
                first['deadline'] != second['deadline'] or first['deadline'] != budget['deadline'] or
                first['initial']['experiment'] != budget['experiment'] or second['initial'] != returned or
                returned['control_id'] != first['run'] or returned['kernel'] != first['kernel'] or
                returned['experiment'] != budget['experiment'] or
                digest(original/'verified/continuation.tar.gz') != returned['sha256'] or
                digest(retry/'dataset/state.bin') != returned['sha256'] or returned['outcome'] != 'recovered' or
                returned['reason'] != 'setup_error' or returned['update'] != 843 or returned['replay_count']):
            raise ValueError('retry is not bound to the recovered zero-game campaign prefix')
        first_two = [member(original,first),member(retry,second)]
        path = base/'campaign-handoff.json'
        if path.exists():
            state = read(path)
            if state['members'][:2] != first_two or state['deadline'] != budget['deadline'] or state['experiment'] != budget['experiment']:
                raise ValueError('different durable campaign binding')
        else:
            state = dict(members=first_two,deadline=budget['deadline'],experiment=budget['experiment'])
        if halted(base,state):
            state.setdefault('halted_by','preexisting-operator-halt')
        write(path,state)
        return state


def route_stop(directory, job):
    base = directory.parent
    state = campaign(directory,job)
    if state is None:
        return None
    state.setdefault('halted_by',job['run'])
    write(base/'campaign-handoff.json',state)
    return effective(base,state)


def reserve_long(previous, directory, job):
    base = previous.parent
    old = read(previous/'job.json') if (previous/'job.json').exists() else {}
    state = campaign(previous,old) if old else None
    if state is None:
        return False
    if (halted(base,state) or effective(base,state) != previous or len(state['members']) != 2 or
            directory.parent != base or job['proof_run'] != old['run'] or
            job['initial']['experiment'] != state['experiment'] or job['deadline'] != state['deadline'] or
            job['revision'] != old['revision'] or job['initial']['control_id'] != old['run']):
        raise ValueError('campaign halted, already claimed or successor differs from verified proof')
    state['members'].append(member(directory,job))
    write(base/'campaign-handoff.json',state)
    return True


def allow_long(previous):
    with lock(previous.parent):
        state = campaign(previous,read(previous/'job.json'))
        if state and (halted(previous.parent,state) or effective(previous.parent,state) != previous):
            raise ValueError('campaign has already ended or advanced')
