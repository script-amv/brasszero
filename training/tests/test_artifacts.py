# SPDX-License-Identifier: Apache-2.0
"""Persistence contracts: committed updates, strict policy envelopes, private streams."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import sys
import subprocess

import numpy as np
import pytest
import torch

from brass_rl.config import ModelConfig, TrainConfig
from brass_rl.model import BrassPolicy
from brass_rl.schema import ComponentData


def api():
    from brass_rl import artifacts
    return artifacts


@pytest.fixture
def training(tmp_path):
    torch.set_num_threads(1)
    c = ComponentData.load()
    config = TrainConfig(model=ModelConfig(entity_dim=4, context_dim=8, recurrent_dim=4))
    model = BrassPolicy(config.model, c.schema, c)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.ppo.learning_rate)
    sum(p.square().sum() for p in model.parameters()).backward()
    optimizer.step()
    from brass_rl.league import Snapshot
    path = tmp_path / "initial.pt"
    torch.save(model.state_dict(), path)
    snapshot = Snapshot("initial", path, 0, "initial")
    state = api().RunState(config=config, master_seed=123, controller_rng_states={"shuffle": random.Random(12).getstate()},
        next_game_id=8, completed_updates=1, league_snapshots=(snapshot,),
        metrics=({"update": 1, "loss": .5},), manifests={"train": [0, 1], "dev": [2]})
    return model, optimizer, state


def draws():
    return random.random(), float(np.random.random()), torch.rand(3)


def assert_draws(a, b):
    assert a[:2] == b[:2]
    assert torch.equal(a[2], b[2])


def test_complete_update_round_trip(tmp_path, training):
    a = api()
    model, optimizer, state = training
    before = a.capture_rng_states()
    checkpoint = a.save_update(tmp_path / "run", model, optimizer, state)
    expected = draws()
    a.restore_rng_states(before)
    assert_draws(draws(), expected)  # save did not consume RNG
    draws()
    loaded = a.load_update(checkpoint, device="cpu")
    assert_draws(draws(), expected)  # restored AFTER model construction
    assert loaded.state.next_game_id == 8
    assert loaded.state.league_snapshot_ids == ("initial",)
    assert loaded.state.manifests == state.manifests
    assert loaded.state.controller_rng_states == state.controller_rng_states
    assert loaded.state.master_seed == 123
    assert all(torch.equal(v, loaded.model.state_dict()[k]) for k,v in model.state_dict().items())
    assert loaded.optimizer_state["param_groups"] == optimizer.state_dict()["param_groups"]
    for key, values in optimizer.state_dict()["state"].items():
        for name, value in values.items():
            assert torch.equal(value, loaded.optimizer_state["state"][key][name])
    snap = loaded.state.league_snapshots[0]
    assert snap.weights_path.parent == tmp_path / "run" / "weights"
    assert hashlib.sha256(snap.weights_path.read_bytes()).hexdigest() == snap.weights_path.stem
    assert a.save_update(tmp_path / "run", model, optimizer, state) == checkpoint
    assert len(list((tmp_path / "run" / "updates").iterdir())) == 1


@pytest.mark.parametrize("failure", ["write", "rename", "pointer"])
def test_save_faults_and_commit_point(tmp_path, training, monkeypatch, failure):
    a = api()
    model, optimizer, state = training
    run = tmp_path / "run"
    a.save_update(run, model, optimizer, state)
    original_write, original_replace = a._write_bytes, a.os.replace
    def write(path, data):
        if failure == "write" and path.name == "controller.pt":
            original_write(path, data[:20])
            raise OSError("injected incomplete file")
        return original_write(path, data)
    def rename(src, dst):
        dst = Path(dst)
        if (failure == "rename" and dst.name == "000002") or (failure == "pointer" and dst.name == "latest"):
            raise OSError("injected publication failure")
        return original_replace(src, dst)
    monkeypatch.setattr(a, "_write_bytes", write)
    monkeypatch.setattr(a.os, "replace", rename)
    with pytest.raises(OSError):
        a.save_update(run, model, optimizer, replace(state, completed_updates=2, next_game_id=16))
    loaded = a.load_update(run / "latest", "cpu")
    assert loaded.state.completed_updates == (2 if failure == "pointer" else 1)
    assert loaded.state.next_game_id == (16 if failure == "pointer" else 8)
    assert loaded.state.league_snapshot_ids == ("initial",)


def test_portable_policy_identity_rng_and_probabilities(tmp_path, training):
    a = api()
    from brass_rl.agent import load_agent, register_checkpoint
    from brass_ai.agents import make_agent
    from brass_rl.features import encode_observation
    import brass
    model, optimizer, state = training
    checkpoint = a.save_update(tmp_path / "run", model, optimizer, state)
    before = a.capture_rng_states()
    exported = a.export_policy(checkpoint, tmp_path / "portable")
    metadata = a.inspect_policy(exported)
    agent = load_agent(exported, 55)
    after = draws()
    a.restore_rng_states(before)
    assert_draws(draws(), after)
    assert set(p.name for p in exported.iterdir()) == {"policy.pt", "metadata.json", "manifest.json"}
    weights = torch.load(exported / "policy.pt", map_location="cpu", weights_only=True)
    assert weights and all(isinstance(v, torch.Tensor) for v in weights.values())
    assert metadata.requires_history is True
    assert metadata.portable_path == exported.resolve()
    with pytest.raises(TypeError):
        metadata.content_hashes["policy.pt"] = "altered"
    # Source checkpoint inference uses no controller deserialization.
    (checkpoint / "controller.pt").write_bytes(b"not a pickle")
    assert load_agent(checkpoint, 55).model.model_id == model.model_id
    with pytest.raises(a.ArtifactError):
        a.inspect_policy(checkpoint)
    game = brass.Game(22)
    obs = game.observation(game.current_player())
    encoded = encode_observation(obs, ComponentData.load())
    with torch.no_grad():
        source = model([encoded], model.initial_memory(1, "cpu"))
        result = agent.model([encoded], agent.model.initial_memory(1, "cpu"))
    assert torch.equal(source.logits.softmax(0), result.logits.softmax(0))
    register_checkpoint("fixture", exported)
    first, second = make_agent("fixture", 7), make_agent("fixture", 7)
    assert first is not second and first._generator is not second._generator
    public = brass.Game(22).observation(0).public_json()
    first.reset(0, public); second.reset(1, public)
    assert first._memory.data_ptr() != second._memory.data_ptr()
    # Identical public payload at another private master/path has identical ID.
    other = a.save_update(tmp_path / "other", model, optimizer, replace(state, master_seed=999))
    other_export = a.export_policy(other, tmp_path / "other-portable")
    assert metadata.id == a.inspect_policy(other_export).id
    text = (exported / "metadata.json").read_text()
    assert "master" not in text and str(tmp_path) not in text


@pytest.mark.parametrize("corruption", ["missing", "hash", "schema", "board", "engine", "extra", "nonfinite"])
def test_reject_invalid_policy_before_act(tmp_path, training, corruption):
    a = api()
    from brass_rl.agent import load_agent
    model, optimizer, state = training
    exported = a.export_policy(a.save_update(tmp_path / "run", model, optimizer, state), tmp_path / "portable")
    if corruption == "missing":
        (exported / "policy.pt").unlink()
    elif corruption == "hash":
        with (exported / "policy.pt").open("ab") as f: f.write(b"x")
    elif corruption == "extra":
        (exported / "controller.pt").write_bytes(b"private")
    else:
        metadata = json.loads((exported / "metadata.json").read_text())
        if corruption == "nonfinite":
            weights = torch.load(exported / "policy.pt", weights_only=True)
            weights["location_ids.weight"][0,0] = float("nan")
            torch.save(weights, exported / "policy.pt")
        else:
            metadata[{"schema": "schema_id", "board": "component_hash", "engine": "engine_fingerprint"}[corruption]] = "wrong"
            (exported / "metadata.json").write_text(json.dumps(metadata))
        from brass_rl.policy_io import policy_identity
        metadata["id"] = policy_identity(metadata, hashlib.sha256((exported / "policy.pt").read_bytes()).hexdigest())
        (exported / "metadata.json").write_text(json.dumps(metadata))
        manifest = json.loads((exported / "manifest.json").read_text())
        for name in ("metadata.json", "policy.pt"):
            manifest["content_hashes"][name] = hashlib.sha256((exported / name).read_bytes()).hexdigest()
        (exported / "manifest.json").write_text(json.dumps(manifest))
    for reader in (a.inspect_policy, lambda p: load_agent(p, 4)):
        with pytest.raises(a.ArtifactError): reader(exported)


def test_resume_rejects_training_config_mismatch(tmp_path, training):
    a = api()
    model, optimizer, state = training
    path = a.save_update(tmp_path / "run", model, optimizer, state)
    for config in (replace(state.config, model=replace(state.config.model, use_history=False)),
                   replace(state.config, ppo=replace(state.config.ppo, entropy_coef=.02)),
                   replace(state.config, league=replace(state.config.league, max_snapshots=2))):
        with pytest.raises(a.ArtifactError): a.load_update(path, "cpu", config=config)
    override = replace(state.config, runtime=replace(state.config.runtime, device="mps", max_wall_time_seconds=5))
    assert a.load_update(path, "cpu", config=override).state.config == override


def rewrite_manifest(path):
    manifest = json.loads((path / "manifest.json").read_text())
    for name in manifest["content_hashes"]:
        manifest["content_hashes"][name] = hashlib.sha256((path / name).read_bytes()).hexdigest()
    (path / "manifest.json").write_text(json.dumps(manifest))


def test_inference_rejects_partial_training_directory(tmp_path, training):
    a = api()
    from brass_rl.agent import load_agent
    model, optimizer, state = training
    path = a.save_update(tmp_path / "run", model, optimizer, state)
    (path / "optimizer.pt").unlink()
    with pytest.raises(a.ArtifactError): load_agent(path, 1)


def test_bad_tensor_archive_has_specific_error_and_preserves_rng(tmp_path, training):
    a = api()
    model, optimizer, state = training
    path = a.export_policy(a.save_update(tmp_path / "run", model, optimizer, state), tmp_path / "export")
    (path / "policy.pt").write_bytes(b"not a torch archive")
    metadata = json.loads((path / "metadata.json").read_text())
    from brass_rl.policy_io import policy_identity
    metadata["id"] = policy_identity(metadata, hashlib.sha256((path / "policy.pt").read_bytes()).hexdigest())
    (path / "metadata.json").write_text(json.dumps(metadata))
    rewrite_manifest(path)
    before = a.capture_rng_states()
    with pytest.raises(a.ArtifactError): a.inspect_policy(path)
    actual = draws()
    a.restore_rng_states(before)
    assert_draws(actual, draws())


def test_invalid_best_reference_cannot_commit(tmp_path, training):
    a = api()
    model, optimizer, state = training
    run = tmp_path / "run"
    a.save_update(run, model, optimizer, state)
    with pytest.raises(a.ArtifactError):
        a.save_update(run, model, optimizer, replace(state, completed_updates=2, development_best_id="missing"))
    assert a.load_update(run / "latest").state.completed_updates == 1


def test_optimizer_nonfinite_state_rejected_before_commit(tmp_path, training):
    a = api()
    model, optimizer, state = training
    next(iter(optimizer.state.values()))["exp_avg"].fill_(float("nan"))
    with pytest.raises(a.ArtifactError): a.save_update(tmp_path / "run", model, optimizer, state)
    assert not (tmp_path / "run/updates/000001").exists()


def test_best_recovers_stale_pointer_and_pool_remains_immutable(tmp_path, training):
    a = api()
    model, optimizer, state = training
    run = tmp_path / "run"
    first = a.save_update(run, model, optimizer, state)
    first_id = json.loads((first / "metadata.json").read_text())["id"]
    second = a.save_update(run, model, optimizer, replace(state, completed_updates=2, development_best_id=first_id))
    (run / "best").write_text("invalid cache")
    assert a.load_update(run / "best").path == first
    assert a.load_update(run / "latest").path == second
    assert len(list((run / "weights").glob("*.pt"))) == 1
    pool_path = a.load_update(second).state.league_snapshots[0].weights_path
    pool_path.write_bytes(b"modified")
    with pytest.raises(a.ArtifactError): a.load_update(run / "latest")


def test_committed_corruption_does_not_fall_back(tmp_path, training):
    a = api()
    model, optimizer, state = training
    run = tmp_path / "run"
    a.save_update(run, model, optimizer, state)
    second = a.save_update(run, model, optimizer, replace(state, completed_updates=2))
    (second / "controller.pt").unlink()
    with pytest.raises(a.ArtifactError): a.load_update(run / "latest")


def test_inference_imports_no_training_state_classes(tmp_path, training):
    a = api()
    model, optimizer, state = training
    path = a.export_policy(a.save_update(tmp_path / "run", model, optimizer, state), tmp_path / "export")
    root = Path(__file__).resolve().parents[2]
    source = """import sys
sys.path[:0] = %r
from brass_rl.agent import load_agent
from brass_rl.policy_io import inspect_policy
inspect_policy(%r)
load_agent(%r, 1)
assert 'brass_rl.artifacts' not in sys.modules
assert 'brass_rl.rollout' not in sys.modules
assert 'brass_rl.league' not in sys.modules
assert 'brass_rl.ppo' not in sys.modules
""" % ([str(root / "engine/tools"), str(root / "training")], str(path), str(path))
    completed = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_resume_save_captures_current_global_rng(tmp_path, training):
    a = api()
    model, optimizer, state = training
    first = a.save_update(tmp_path / "run", model, optimizer, state)
    loaded = a.load_update(first)
    resumed_optimizer = torch.optim.Adam(loaded.model.parameters())
    resumed_optimizer.load_state_dict(loaded.optimizer_state)
    draws()  # work after the restored boundary advances all global streams
    before = a.capture_rng_states()
    second = a.save_update(tmp_path / "run", loaded.model, resumed_optimizer,
                           replace(loaded.state, completed_updates=2, next_game_id=16))
    expected = draws()
    a.load_update(second)
    assert_draws(draws(), expected)
    a.restore_rng_states(before)


def test_league_restore_survives_evicted_file_pruning(tmp_path, training):
    a = api()
    from brass_rl.league import League, Snapshot
    from brass_rl.seeding import SeedStreams
    model, optimizer, state = training
    config = replace(state.config, league=replace(state.config.league, max_snapshots=3))
    league = League(SeedStreams(123), state.league_snapshots[0], config.league)
    for update in range(1, 8):
        path = tmp_path / ("snapshot-%d.pt" % update)
        torch.save(model.state_dict(), path)
        league.add(Snapshot(str(update), path, update, "training"))
    holdout = Snapshot("holdout", state.league_snapshots[0].weights_path, 0, "holdout")
    league.add(holdout)
    snapshots = league.snapshots
    active = {s.id for s in snapshots}
    evicted = next(str(i) for i in range(1, 8) if str(i) not in active)
    old_path = tmp_path / ("snapshot-%s.pt" % evicted)
    old_bytes = old_path.read_bytes()
    old_path.unlink()
    checkpoint = a.save_update(tmp_path / "run", model, optimizer, replace(state, config=config,
        completed_updates=7, league_snapshots=snapshots, holdouts=league.holdouts, league_history=league.history))
    loaded = a.load_update(checkpoint)
    restored = League.restore(SeedStreams(123), loaded.state.league_snapshots, config.league,
                              holdouts=loaded.state.holdouts, history=loaded.state.league_history)
    assert tuple(s.id for s in restored.snapshots) == tuple(s.id for s in snapshots)
    assert tuple(s.id for s in restored.holdouts) == ("holdout",)
    assert restored.history == league.history
    assert [restored.sample(i) for i in range(40)] == [league.sample(i) for i in range(40)]
    for snapshot in restored.snapshots + restored.holdouts:
        restored.verify(snapshot)
    new_path = tmp_path / "reused.pt"
    new_path.write_bytes(old_bytes + b"different")
    with pytest.raises(ValueError):
        restored.add(Snapshot(evicted, new_path, int(evicted), "training"))
    # Same ID/content tombstone is harmless and never re-enters the active pool.
    new_path.write_bytes(old_bytes)
    restored.add(Snapshot(evicted, new_path, int(evicted), "training"))
    assert tuple(s.id for s in restored.snapshots) == tuple(s.id for s in snapshots)


def test_retry_after_pointer_failure_is_idempotent(tmp_path, training, monkeypatch):
    a = api()
    model, optimizer, state = training
    run = tmp_path / "run"
    a.save_update(run, model, optimizer, state)
    next_state = replace(state, completed_updates=2)
    original = a.os.replace
    def fail_pointer(src, dst):
        if Path(dst).name == "latest": raise OSError("pointer failed")
        return original(src, dst)
    monkeypatch.setattr(a.os, "replace", fail_pointer)
    with pytest.raises(OSError): a.save_update(run, model, optimizer, next_state)
    monkeypatch.setattr(a.os, "replace", original)
    committed = (run / "updates/000002/controller.pt").read_bytes()
    draws()
    assert a.save_update(run, model, optimizer, next_state).name == "000002"
    assert (run / "updates/000002/controller.pt").read_bytes() == committed
    assert len(list((run / "updates").iterdir())) == 2
    with pytest.raises(a.ArtifactError):
        a.save_update(run, model, optimizer, replace(next_state, next_game_id=999))


def test_changed_snapshot_conflicting_with_history_cannot_commit(tmp_path, training):
    a = api()
    from brass_rl.league import League
    from brass_rl.seeding import SeedStreams
    model, optimizer, state = training
    league = League(SeedStreams(123), state.league_snapshots[0], state.config.league)
    history = league.history
    path = state.league_snapshots[0].weights_path
    weights = torch.load(path, weights_only=True)
    weights["location_ids.weight"][0,0] += 1
    torch.save(weights, path)
    with pytest.raises(a.ArtifactError):
        a.save_update(tmp_path / "run", model, optimizer, replace(state, league_history=history))
    assert not (tmp_path / "run/updates/000001").exists()


def test_renamed_directory_recovers_even_when_parent_sync_fails(tmp_path, training, monkeypatch):
    a = api()
    model, optimizer, state = training
    run = tmp_path / "run"
    a.save_update(run, model, optimizer, state)
    original = a._sync_dir
    def failed_sync(path):
        if Path(path).name == "updates": raise OSError("parent sync failure after commit")
        return original(path)
    monkeypatch.setattr(a, "_sync_dir", failed_sync)
    with pytest.raises(OSError):
        a.save_update(run, model, optimizer, replace(state, completed_updates=2))
    assert a.load_update(run / "latest").state.completed_updates == 2
    interrupted = run / "updates/.update-interrupted"
    interrupted.mkdir()
    (interrupted / "policy.pt").write_bytes(b"partial")
    assert a.load_update(run / "latest").state.completed_updates == 2


def test_missing_development_best_rejected_on_resume(tmp_path, training):
    a = api()
    model, optimizer, state = training
    run = tmp_path / "run"
    first = a.save_update(run, model, optimizer, state)
    best_id = json.loads((first / "metadata.json").read_text())["id"]
    second = a.save_update(run, model, optimizer, replace(state, completed_updates=2, development_best_id=best_id))
    import shutil
    shutil.rmtree(first)
    with pytest.raises(a.ArtifactError): a.load_update(second)


def test_portable_manifest_rejects_private_extra_fields(tmp_path, training):
    a = api()
    model, optimizer, state = training
    path = a.export_policy(a.save_update(tmp_path / "run", model, optimizer, state), tmp_path / "export")
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["master_seed"] = 123
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(a.ArtifactError): a.inspect_policy(path)


def test_all_written_files_validated_before_commit(tmp_path, training, monkeypatch):
    a = api()
    model, optimizer, state = training
    run = tmp_path / "run"
    a.save_update(run, model, optimizer, state)
    original = a._write_bytes
    def truncate_controller(path, data):
        return original(path, data[:10] if Path(path).name == "controller.pt" else data)
    monkeypatch.setattr(a, "_write_bytes", truncate_controller)
    with pytest.raises(a.ArtifactError):
        a.save_update(run, model, optimizer, replace(state, completed_updates=2))
    assert a.load_update(run / "latest").state.completed_updates == 1


def test_unavailable_accelerator_resume_on_cpu_reports_nonexact_continuation(tmp_path, training, monkeypatch):
    a = api()
    model, optimizer, state = training
    gpu_config = replace(state.config, runtime=replace(state.config.runtime, device="cuda"))
    path = a.save_update(tmp_path / "run", model, optimizer, replace(state, config=gpu_config))
    controller = torch.load(path / "controller.pt", weights_only=False)
    # Simulated provenance only: no real GPU execution is claimed by this test.
    controller["rng_states"]["torch_cuda"] = [controller["rng_states"]["torch_cpu"].clone()]
    torch.save(controller, path / "controller.pt")
    rewrite_manifest(path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    loaded = a.load_update(path, device="cpu", config=state.config)
    actual = draws()
    assert loaded.rng_restore_notes and "CUDA" in loaded.rng_restore_notes[0]
    assert "not exact" in loaded.rng_restore_notes[0]
    assert "torch_cuda" in loaded.state.rng_states  # preserve source provenance
    available = dict(controller["rng_states"])
    available.pop("torch_cuda")
    a.restore_rng_states(available)
    assert_draws(actual, draws())
    assert loaded.state.next_game_id == 8
    assert all(torch.equal(v, loaded.model.state_dict()[k]) for k,v in model.state_dict().items())
    with pytest.raises(a.ArtifactError):
        a.restore_rng_states(controller["rng_states"])  # default exact restore stays strict
    with pytest.raises(a.ArtifactError):
        a.load_update(path, device="cuda")  # unavailable active target still fails
    resumed = torch.optim.Adam(loaded.model.parameters())
    resumed.load_state_dict(loaded.optimizer_state)
    next_path = a.save_update(tmp_path / "run", loaded.model, resumed, replace(loaded.state, completed_updates=2))
    next_load = a.load_update(next_path, "cpu")
    assert not next_load.rng_restore_notes
    assert "torch_cuda" not in next_load.state.rng_states


def test_same_shaped_optimizer_parameter_permutation_rejected_before_commit(tmp_path, training):
    a = api()
    model, optimizer, state = training
    run = tmp_path / "run"
    first = a.save_update(run, model, optimizer, state)
    named = list(model.named_parameters())
    names = [name for name, _ in named]
    left, right = names.index("location_ids.weight"), names.index("card_ids.weight")
    parameters = [parameter for _, parameter in named]
    assert parameters[left].shape == parameters[right].shape
    parameters[left], parameters[right] = parameters[right], parameters[left]
    permuted = torch.optim.Adam(parameters, lr=state.config.ppo.learning_rate)
    for index, (_, parameter) in enumerate(named):
        parameter.grad = torch.full_like(parameter, (index + 1) / 100.)
    permuted.step()
    assert not torch.equal(permuted.state[model.location_ids.weight]["exp_avg"],
                           permuted.state[model.card_ids.weight]["exp_avg"])
    with pytest.raises(a.ArtifactError):
        a.save_update(run, model, permuted, replace(state, completed_updates=2))
    assert not (run / "updates/000002").exists()
    assert a.load_update(run / "latest").path == first


def test_canonical_adam_resume_preserves_actual_next_step(tmp_path, training):
    a = api()
    model, optimizer, state = training
    checkpoint = a.save_update(tmp_path / "run", model, optimizer, state)
    loaded = a.load_update(checkpoint)
    resumed = torch.optim.Adam(loaded.model.parameters(), lr=state.config.ppo.learning_rate)
    resumed.load_state_dict(loaded.optimizer_state)
    expected_parameters = dict(model.named_parameters())
    actual_parameters = dict(loaded.model.named_parameters())
    for index, (name, parameter) in enumerate(expected_parameters.items()):
        gradient = torch.full_like(parameter, (index + 1) / 100.)
        parameter.grad = gradient
        actual_parameters[name].grad = gradient.clone()
    optimizer.step()
    resumed.step()
    for name, expected in expected_parameters.items():
        actual = actual_parameters[name]
        assert torch.equal(actual, expected), name
        for field in ("step", "exp_avg", "exp_avg_sq"):
            assert torch.equal(resumed.state[actual][field], optimizer.state[expected][field]), (name, field)


def test_policy_descriptor_hashes_all_three_portable_files(tmp_path, training):
    a = api()
    model, optimizer, state = training
    checkpoint = a.save_update(tmp_path / "run", model, optimizer, state)
    portable = a.export_policy(checkpoint, tmp_path / "portable")
    metadata = a.inspect_policy(portable)
    assert set(metadata.content_hashes) == {"policy.pt", "metadata.json", "manifest.json"}
    for name, digest in metadata.content_hashes.items():
        assert digest == hashlib.sha256((portable / name).read_bytes()).hexdigest()
    manifest = json.loads((portable / "manifest.json").read_text())
    assert set(manifest["content_hashes"]) == {"policy.pt", "metadata.json"}
    assert metadata.id == json.loads((checkpoint / "metadata.json").read_text())["id"]


def test_trusted_source_provenance_is_outside_portable_identity(tmp_path, training, monkeypatch):
    a = api()
    model, optimizer, state = training
    first = a.save_update(tmp_path/'first', model, optimizer, state)
    versions = json.loads((first/'versions.json').read_text())
    assert versions['source']['kind'] in {'git', 'archive-declared', 'unavailable'}
    monkeypatch.setattr(a, 'source_provenance', lambda: {'kind': 'archive-declared', 'revision': 'f'*40})
    second = a.save_update(tmp_path/'second', model, optimizer, state)
    assert json.loads((second/'versions.json').read_text())['source']['revision'] == 'f'*40
    assert (first/'metadata.json').read_bytes() == (second/'metadata.json').read_bytes()
    left = a.export_policy(first, tmp_path/'portable-a')
    right = a.export_policy(second, tmp_path/'portable-b')
    for name in ('policy.pt', 'metadata.json', 'manifest.json'):
        assert (left/name).read_bytes() == (right/name).read_bytes()
    # Earlier completed updates without source provenance remain loadable.
    versions.pop('source')
    (first/'versions.json').write_text(json.dumps(versions))
    # Refresh trusted journal payload hash just as an old producer would have done.
    manifest = json.loads((first/'manifest.json').read_text())
    manifest['content_hashes']['versions.json'] = hashlib.sha256((first/'versions.json').read_bytes()).hexdigest()
    (first/'manifest.json').write_text(json.dumps(manifest))
    assert a.load_update(first).state.completed_updates == state.completed_updates


def test_committed_export_never_reserializes_identity(tmp_path, training, monkeypatch):
    a = api()
    model, optimizer, state = training
    checkpoint = a.save_update(tmp_path/'run', model, optimizer, state)
    original_id = json.loads((checkpoint/'metadata.json').read_text())['id']
    # A different serializer is exactly the cross-runtime failure mode: trusted
    # committed payloads must be exported as bytes, never re-identified.
    monkeypatch.setattr(a, 'tensor_bytes', lambda weights: b'different-runtime-serialization')
    destination = a.export_policy(checkpoint, tmp_path/'portable')
    assert a.inspect_policy(destination).id == original_id
    for name in ('policy.pt', 'metadata.json'):
        assert (destination/name).read_bytes() == (checkpoint/name).read_bytes()
    assert a.export_policy(checkpoint, destination) == destination
