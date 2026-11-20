# SPDX-License-Identifier: Apache-2.0
"""Endpoint-only performance on eight paired, four-seat deal blocks."""
import numpy as np

CANDIDATES = ('parent', 'control', 'A', 'B', 'C')
LINEUPS = ('greedy', 'frozen688')


def _blocks(rows, lineup):
    games = sorted((r for r in rows if r['lineup'] == lineup),
                   key=lambda r: (r['deal_id'], r['candidate_seat']))
    if [(r['deal_id'], r['candidate_seat']) for r in games] != [(d, s) for d in range(8) for s in range(4)]:
        raise ValueError('each lineup requires eight deals and four distinct focal seats per deal')
    values, seeds = [], []
    for r in games:
        s = r['candidate_seat']
        if r['rotation'] != (-s) % 4:
            raise ValueError('rotation disagrees with candidate seat')
        rank = r['ranking'][s]
        values.append((r['vp'][s], rank + 1, int(rank == 0 and r['ranking'].count(0) == 1)))
        seeds.append(r['deal_seed'])
    if len(set(seeds)) != 8 or any(len(set(seeds[i:i+4])) != 1 for i in range(0, 32, 4)):
        raise ValueError('eight distinct native deal seeds are required')
    return np.array(values).reshape(8, 4, 3), seeds


def _metrics(data):
    vp, rank, win = (data[..., i] for i in range(3))
    return {'mean_vp': vp.mean(-1), 'median_vp': np.median(vp, axis=-1),
            'lower_quartile_vp': np.percentile(vp, 25, axis=-1), 'max_vp': vp.max(-1),
            'mean_rank': rank.mean(-1), 'strict_win_rate': win.mean(-1)}


def summarize(rows, *, candidates=CANDIDATES):
    """Percentile intervals resample matching deal indices with all four seats."""
    if set(rows) != set(candidates) or any(len(v) != 64 for v in rows.values()):
        raise ValueError('each declared frozen policy must have 64 games')
    policies, paired, metric_sets = {}, {}, {}
    draws = np.random.default_rng(20261119).integers(0, 8, size=(4096, 8))
    common = None
    for name in candidates:
        policies[name] = {}
        for lineup in LINEUPS:
            blocks, seeds = _blocks(rows[name], lineup)
            if common is None:
                common = seeds
            if seeds != common:
                raise ValueError('mismatched deal blocks across candidates or lineups')
            pooled = _metrics(blocks.reshape(32, 3))
            metric_sets[name, lineup] = (pooled, _metrics(blocks), _metrics(blocks[draws].reshape(-1, 32, 3)))
            stats = {k: float(v) for k, v in pooled.items() if k != 'strict_win_rate'}
            policies[name][lineup] = dict(stats, games=32, deals=8, strict_wins=int(blocks[:, :, 2].sum()))
    for name in candidates:
        paired[name] = {}
        for reference in ('control', 'parent'):
            if reference == name:
                continue
            paired[name][reference] = {}
            for lineup in LINEUPS:
                point, blocks, samples = metric_sets[name, lineup]
                base, ref_blocks, ref_samples = metric_sets[reference, lineup]
                comparison = {}
                for k in point:
                    delta = blocks[k] - ref_blocks[k]
                    comparison[k] = {'deal_differences': delta.tolist(), 'mean_difference': float(delta.mean()),
                                     'pooled_difference': float(point[k] - base[k]),
                                     'bootstrap_95': np.percentile(samples[k] - ref_samples[k], [2.5, 97.5]).tolist()}
                paired[name][reference][lineup] = comparison
    return {'policies': policies, 'paired': paired,
            'method': 'VP quartiles use linear interpolation. Paired percentile intervals resample all four '
                      'seat rotations jointly within 8 deals, 4096 fixed-seed draws. Intervals describe '
                      'pooled differences. Per-deal differences and their mean are also supplied. '
                      'Win differences are strict-win rate fractions. One training replicate, eight evaluation '
                      'blocks; exploratory uncertainty omits training variation and multiplicity.'}


def markdown_report(result, *, candidates=CANDIDATES, title='Four independent continuations from 688'):
    training = result.get('training_games', (len(candidates)-1)*100)
    evaluation = result.get('evaluation_games', len(candidates)*64)
    lines = ['# '+title, '',
             f'{training} training games and {evaluation} evaluation games completed. Each policy played eight fresh deals '
             'in all four candidate seats against each lineup.', '',
             '| Policy | Lineup | Mean VP | Median VP | Lower quartile VP | Maximum VP | Strict wins / 32 | Mean rank |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for arm in candidates:
        for lineup in LINEUPS:
            row = result['policies'][arm][lineup]
            lines.append('| %s | %s | %.3f | %.2f | %.2f | %.0f | %d | %.3f |' %
                         (arm, lineup, row['mean_vp'], row['median_vp'], row['lower_quartile_vp'],
                          row['max_vp'], row['strict_wins'], row['mean_rank']))
    lines.extend(['', result['method'], '', '## Paired differences by lineup', ''])
    for arm, references in result['paired'].items():
        for reference, lineups in references.items():
            lines.extend(['### '+arm+' versus '+reference, '',
                          '| Lineup | Metric | Pooled difference | Paired 95% interval | Deal differences (0–7) |',
                          '|---|---|---:|---|---|'])
            for lineup, metrics in lineups.items():
                for metric, row in metrics.items():
                    low, high = row['bootstrap_95']
                    values = ', '.join('%g' % d for d in row['deal_differences'])
                    lines.append('| %s | %s | %.3f | [%.3f, %.3f] | %s |' %
                                 (lineup, metric, row['pooled_difference'], low, high, values))
            lines.append('')
    lines.extend(['## Frozen models and resources', ''])
    for arm in candidates:
        lines.append('- %s `%s`' % (arm, result['endpoints'][arm]['id']))
    lines.extend(['', 'Active controller time %.2f h. Elapsed wall time including pauses %.2f h. '
                  'Study and shared growth {:,} logical bytes.'.format(result['disk']['combined_logical_bytes']) %
                  (result['active_seconds']/3600, result['elapsed_seconds']/3600), ''])
    return '\n'.join(lines)
