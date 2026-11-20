# SPDX-License-Identifier: Apache-2.0
import json
import brass
from brass_ai.replay_archive import Archive, Recorder, game_file


def test_durable_completion_rebuild_and_retry(tmp_path):
    archive = Archive(tmp_path)
    seed = 2**64 - 1
    seats = [{'kind': 'bot', 'name': 'Random', 'bot': 'random:test'}]*4
    recorder = Recorder(archive, seed=seed, seats=seats, provenance={'category': 'test', 'game': 1})
    game = brass.Game(seed)
    actions = []
    while not game.is_over():
        actor = game.current_player()
        move = json.loads(game.legal_moves()[0])
        recorder.append(actor, move)
        game.apply(json.dumps(move))
        actions.append({'player': actor, 'move': move})
    recorder.finish(game)
    record = archive.read(recorder.game_id)
    assert record['status'] == 'completed'
    assert record['result']['vp'] == list(game.vp())
    assert record['result']['ranking'] == list(game.ranking())
    assert game_file(record)['seed'] == seed
    assert record['moves'] == actions
    assert len(archive.rebuild()) == 1
    retry = Recorder(archive, seed=seed, seats=seats, provenance={'category': 'test', 'game': 1})
    for action in actions:
        retry.append(action['player'], action['move'])
    retry.finish(game)
    assert len(archive.rebuild()) == 1


def test_interrupted_prefix_survives_partial_tail(tmp_path):
    archive = Archive(tmp_path)
    recorder = Recorder(archive, seed=42, seats=[{'kind':'human','name':'Human','bot':None}]*4,
                        provenance={'category':'gui'})
    game = brass.Game(42)
    recorder.append(game.current_player(), json.loads(game.legal_moves()[0]))
    with recorder.path.open('ab') as f:
        f.write(b'{"partial":')
    record = archive.read(recorder.game_id)
    assert len(record['moves']) == 1
    assert record['status'] in ('in-progress', 'interrupted')
    assert len(archive.rebuild()) == 1


def test_legacy_copies_import_once_and_preserve_bytes(tmp_path):
    from brass_ai.replay_archive import import_legacy
    source = tmp_path/'old'
    source.mkdir()
    game = brass.Game(999)
    moves = []
    while not game.is_over():
        moves.append({'player':game.current_player(), 'move':json.loads(game.legal_moves()[0])})
        game.apply(json.dumps(moves[-1]['move']))
    payload = {'format':'brass-gui-game/1', 'seed':999,
               'seats':[{'kind':'human','name':'old','bot':None}]*4, 'moves':moves}
    raw = json.dumps(payload).encode()
    for name in ('a.json', 'copy.json'):
        (source/name).write_bytes(raw)
    (source/'outcome.json').write_text(json.dumps({'vp':list(game.vp())}))
    archive = Archive(tmp_path/'archive')
    first = import_legacy(archive, source)
    assert first['imported'] == 1 and first['duplicates'] == 1
    assert import_legacy(archive, source)['imported'] == 0
    assert len(archive.records()[0]['provenance']['sources']) == 2
    assert (source/'a.json').read_bytes() == raw


def test_writer_conflict_and_storage_failure_stop_before_mutation(tmp_path, monkeypatch):
    import pytest
    import brass_ai.replay_archive as module
    args = dict(seed=3, seats=[{'kind':'human','name':'h'}]*4, provenance={'category':'test'})
    recorder = Recorder(Archive(tmp_path), **args)
    with pytest.raises(module.ReplayStorageError, match='Another controller'):
        Recorder(Archive(tmp_path), **args)
    def fail(fd):
        raise OSError('disk full')
    monkeypatch.setattr(module.os, 'fsync', fail)
    with pytest.raises(module.ReplayStorageError, match='game stopped'):
        recorder.append(0, 'pass')


def test_interrupt_is_idempotent_and_cannot_modify_retry(tmp_path):
    import pytest
    from brass_ai.replay_archive import ReplayStorageError
    archive = Archive(tmp_path)
    args = dict(seed=3, seats=[{'kind':'human','name':'h'}]*4, provenance={'category':'test'})
    first = Recorder(archive, **args)
    first.interrupt()
    retry = Recorder(archive, **args)
    before = retry.path.read_bytes()
    first.interrupt()
    assert retry.path.read_bytes() == before
    with pytest.raises(ReplayStorageError, match='closed'):
        first.append(0, 'pass')
    retry.interrupt()


def test_header_storage_failure_releases_lock(tmp_path, monkeypatch):
    import pytest
    import brass_ai.replay_archive as module
    archive = Archive(tmp_path)
    args = dict(seed=3, seats=[{'kind':'human','name':'h'}]*4, provenance={'category':'test'})
    original = module.atomic_bytes
    def fail(*args):
        raise OSError('disk full')
    monkeypatch.setattr(module, 'atomic_bytes', fail)
    with pytest.raises(module.ReplayStorageError, match='storage failed'):
        Recorder(archive, **args)
    monkeypatch.setattr(module, 'atomic_bytes', original)
    retry = Recorder(archive, **args)
    retry.interrupt()


def test_actual_arena_evaluation_archives_rotations_and_deduplicates_resume():
    from brass_ai.arena import run_match
    results = run_match(['random']*4, 1, 17326)
    archive = Archive()
    records = archive.records()
    assert len(records) == 4
    for record in records:
        result = results[record['provenance']['game']]
        assert record['result']['vp'] == list(result.vp)
        assert record['result']['ranking'] == list(result.ranking)
        assert all(s['bot'].startswith('random:native-v1:') for s in record['seats'])
    assert run_match(['random']*4, 1, 17326) == results
    assert len(archive.rebuild()) == 4


def test_legacy_default_lineup_and_nulls_deduplicate(tmp_path):
    from brass_ai.replay_archive import import_legacy
    source = tmp_path/'old'; source.mkdir()
    seats = [{'kind':'human', 'name':'Human', 'bot':None}]*4
    payload = {'format':'brass-gui-game/1','seed':3,'seats':seats,'moves':[]}
    (source/'original.json').write_text(json.dumps(payload))
    copy = dict(payload, start_state=None, lineups=[{'ply':0,'seats':seats}])
    (source/'export.json').write_text(json.dumps(copy))
    archive = Archive(tmp_path/'archive')
    assert import_legacy(archive, source)['imported'] == 1
    assert import_legacy(archive, source)['imported'] == 0
    assert len(archive.records()) == 1


def test_generated_exemplar_links_to_recorded_origin(tmp_path):
    from brass_ai.replay_archive import import_legacy, labeled_seats
    source = tmp_path/'run'/'replays'; source.mkdir(parents=True)
    archive = Archive(tmp_path/'archive')
    seats = labeled_seats(['random']*4)
    recorder = Recorder(archive, seed=17, seats=seats, provenance={
        'category':'training','game':0,'experiment':'test','experiment_path':str(source.parent)})
    game = brass.Game(17)
    moves=[]
    while not game.is_over():
        move={'player':game.current_player(),'move':json.loads(game.legal_moves()[0])}
        recorder.append(move['player'],move['move']); game.apply(json.dumps(move['move'])); moves.append(move)
    recorder.finish(game)
    payload={'format':'brass-gui-game/1','seed':17,'seats':[{'kind':'human','name':'random','bot':None}]*4,'moves':moves}
    (source/'0.json').write_text(json.dumps(payload))
    assert import_legacy(archive, source)['imported'] == 0
    assert len(archive.records()) == 1
    assert archive.read(recorder.game_id)['seats'] == seats


def test_reader_handles_atomic_journal_to_gzip_handoff(tmp_path, monkeypatch):
    import gzip
    from pathlib import Path
    from brass_ai.replay_archive import atomic_bytes, encoded
    archive = Archive(tmp_path)
    recorder = Recorder(archive, seed=3, seats=[{'kind':'human','name':'h'}]*4, provenance={'category':'test'})
    original = Path.read_bytes
    def race(path):
        if path == recorder.path:
            atomic_bytes(archive.path(recorder.game_id), gzip.compress(encoded(recorder.record)))
            path.unlink()
        return original(path)
    monkeypatch.setattr(Path, 'read_bytes', race)
    assert archive.read(recorder.game_id)['game_id'] == recorder.game_id


def test_import_preserves_modern_gui_origin_metadata(tmp_path):
    from brass_ai.replay_archive import import_legacy, game_file
    archive = Archive(tmp_path/'archive')
    recorder = Recorder(archive, seed=4, seats=[{'kind':'human','name':'h'}]*4,
                        provenance={'category':'gui','session':'original','parent':'parent','cursor':0})
    recorder.interrupt()
    source=tmp_path/'source';source.mkdir()
    (source/'game.json').write_text(json.dumps(game_file(archive.read(recorder.game_id))))
    assert import_legacy(archive, source)['imported'] == 0
    record=archive.read(recorder.game_id)
    assert record['provenance']['session'] == 'original'
    assert record['provenance']['parent'] == 'parent'


def test_process_exit_retains_durable_chosen_prefix(tmp_path):
    import os
    import subprocess
    import sys
    code = '''import os,json,brass
from brass_ai.replay_archive import Recorder
r=Recorder(seed=9,seats=[{'kind':'human','name':'h'}]*4,provenance={'category':'test'})
g=brass.Game(9);r.append(g.current_player(),json.loads(g.legal_moves()[0]))
print(r.game_id,flush=True)
os._exit(7)
'''
    env=dict(os.environ, BRASS_REPLAY_ROOT=str(tmp_path))
    process=subprocess.run([sys.executable,'-c',code],env=env,capture_output=True,text=True)
    assert process.returncode == 7
    record=Archive(tmp_path).read(process.stdout.strip())
    assert record['status'] == 'interrupted'
    assert len(record['moves']) == 1
    game=brass.Game(9)
    game.apply(json.dumps(record['moves'][0]['move']))


def test_llm_status_journal_preserves_identity_and_public_move_stream(tmp_path):
    archive = Archive(tmp_path)
    seats = [{'kind':'bot','name':'Remote','bot':'llm:sha256-v1:abc','bot_family':'llm',
              'llm':{'profile': {'model':'vendor/alias'}}}] + [{'kind':'human','name':'Human'}]*3
    recorder = Recorder(archive, seed=7, seats=seats, provenance={'category':'test'})
    try:
        recorder.set_llm_status(0, {'calls':1, 'reported_model':'model-1', 'usage':{'total_tokens':5}})
        move = json.loads(brass.Game(7).legal_moves()[0])
        recorder.append(0, move)
        recorder.set_llm_status(0, {'calls':2, 'reported_model':'model-2', 'usage':{'total_tokens':10}})
    finally:
        recorder.interrupt()
    # A still-running pre-LLM reader skips status frames and treats every other
    # row as a move. It must observe exactly the same legal action prefix.
    journal = [json.loads(line) for line in recorder.path.read_text().splitlines()][1:]
    assert [row for row in journal if 'status' not in row] == [{'player':0, 'move':move}]
    record = archive.read(recorder.game_id)
    assert record['moves'] == [{'player':0, 'move':move}]
    assert record['llm_status']['0']['calls'] == 2 and record['llm_status']['0']['reported_model'] == 'model-2'
    assert record['seats'] == seats and record['seat_identities'][0]['kind'] == 'llm'
    assert game_file(record)['seats'][0]['llm']['profile']['model'] == 'vendor/alias'
