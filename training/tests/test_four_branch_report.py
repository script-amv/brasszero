# SPDX-License-Identifier: Apache-2.0
import pytest


def fixture_rows(offset):
    rows = []
    for name in ('greedy', 'frozen688'):
        for deal in range(8):
            for seat in range(4):
                vp = [0] * 4
                vp[seat] = offset * deal + seat
                ranks = [0, 1, 2, 3] if deal else [0, 0, 2, 3]
                rows.append(dict(lineup=name, deal_id=deal, candidate_seat=seat, rotation=(-seat) % 4,
                                 vp=vp, ranking=ranks, deal_seed=2**63+deal))
    return rows


def test_report_counts_ties_rotations_and_paired_deal_differences():
    from brass_rl.four_branch_report import summarize
    rows = {arm: fixture_rows(i) for i, arm in enumerate(('parent', 'control', 'A', 'B', 'C'))}
    result = summarize(rows)
    parent = result['policies']['parent']['greedy']
    assert parent == dict(games=32, deals=8, mean_vp=1.5, median_vp=1.5, lower_quartile_vp=.75,
                          max_vp=3, strict_wins=7, mean_rank=79/32)
    paired = result['paired']['A']['control']['frozen688']['mean_vp']
    assert paired['deal_differences'] == list(range(8))
    assert paired['mean_difference'] == 3.5
    assert 0 < paired['bootstrap_95'][0] < 3.5 < paired['bootstrap_95'][1] < 7
    assert result['paired']['A']['parent']['greedy']['mean_vp']['mean_difference'] == 7
    rows['C'][0] = rows['C'][1]
    with pytest.raises(ValueError):
        summarize(rows)


def test_markdown_report_exposes_samples_units_and_both_references():
    from brass_rl.four_branch_report import markdown_report, summarize
    rows = {arm: fixture_rows(i) for i, arm in enumerate(('parent', 'control', 'A', 'B', 'C'))}
    result = summarize(rows)
    result.update(active_seconds=7200, elapsed_seconds=7300, disk={'combined_logical_bytes': 1024},
                  endpoints={arm: {'id': arm+'-policy'} for arm in rows})
    report = markdown_report(result)
    assert '400 training games' in report and '320 evaluation games' in report
    assert 'A versus control' in report and 'A versus parent' in report and 'A-policy' in report
    assert '2.00 h' in report and '1,024' in report
