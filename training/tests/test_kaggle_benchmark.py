# SPDX-License-Identifier: Apache-2.0
from scripts import kaggle_benchmark as benchmark


def test_schedule_is_bounded_and_order_balanced():
    schedule = list(benchmark.schedule())
    assert schedule == [(0, 'cpu'), (0, 'cuda'), (1, 'cuda'), (1, 'cpu'),
                        (2, 'cpu'), (2, 'cuda'), (3, 'cuda'), (3, 'cpu')]
    assert len(schedule) * 2 == 16


def test_stage_waits_for_device_at_both_edges():
    events = []
    def sync():
        events.append('sync')
    def clock():
        events.append('clock')
        return len(events)
    timings = {}
    def work(value):
        events.append('work')
        return 2 * value
    result = benchmark.stage(work, 'ppo', timings, sync, clock)(3)
    assert result == 6
    assert events == ['sync', 'clock', 'work', 'sync', 'clock']
    assert timings == {'ppo': (2, 5)}
