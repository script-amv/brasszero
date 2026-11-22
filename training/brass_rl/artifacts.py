"""Single-writer completed-update journal and portable policy export.

The directory rename commits an update; latest/best files are disposable caches.
Controller/optimizer files are trusted local training data, never portable input.
No retention deletion occurs here: all referenced content-addressed weights survive.
"""
from dataclasses import asdict, dataclass, field
import io
import json
import os
from pathlib import Path
import platform
import pickle
import random
import shutil
import tempfile
from typing import Any, Mapping, Optional, Tuple

import numpy as np
import torch

from .retention import pin, present_payloads
from .source import source_provenance
from .config import TrainConfig, _SECTIONS
from .league import League, Snapshot
from .seeding import SeedStreams
from .policy_io import (ArtifactError, PolicyMetadata, POLICY_FORMAT, UPDATE_FORMAT,
                        inspect_policy, json_bytes, load_policy, policy_metadata, UPDATE_PAYLOAD_FILES,
                        read_json, sha256, tensor_bytes, verified_bytes)

UPDATE_FILES = UPDATE_PAYLOAD_FILES


@dataclass(frozen=True)
class RunState:
    config: TrainConfig
    master_seed: int
    controller_rng_states: Mapping[str, Any]
    next_game_id: int
    completed_updates: int
    league_snapshots: Tuple[Snapshot, ...] = ()
    holdouts: Tuple[Snapshot, ...] = ()
    development_best_id: Optional[str] = None
    metrics: Tuple[Mapping[str, Any], ...] = ()
    manifests: Mapping[str, Any] = field(default_factory=dict)
    # Controller may retain historical ID/digest metadata for evicted snapshots.
    league_history: Mapping[str, Any] = field(default_factory=dict)
    # Read-only resume snapshot for callers constructing objects after load.
    # save_update always captures the CURRENT global streams, not this old copy.
    rng_states: Optional[Mapping[str, Any]] = None

    dynamic_league: Optional[Mapping[str, Any]] = None

    @property
    def league_snapshot_ids(self):
        return tuple(s.id for s in self.league_snapshots)

    def __post_init__(self):
        if not isinstance(self.config, TrainConfig):
            raise ValueError("RunState requires TrainConfig")
        if type(self.master_seed) is not int or not 0 <= self.master_seed < 2**64:
            raise ValueError("master_seed must be an unsigned 64-bit integer")
        for name in ("next_game_id", "completed_updates"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(name + " must be nonnegative")
        snapshots = tuple(self.league_snapshots) + tuple(self.holdouts)
        if len({s.id for s in snapshots}) != len(snapshots):
            raise ValueError("snapshot IDs must be unique")
        if any(s.role == "holdout" for s in self.league_snapshots) or any(s.role != "holdout" for s in self.holdouts):
            raise ValueError("holdout and training snapshot roles must be separate")
        if self.dynamic_league is None and len(self.league_snapshots) > self.config.league.max_snapshots:
            raise ValueError("too many league snapshots")
        if any(s.update > self.completed_updates for s in snapshots):
            raise ValueError("snapshot cannot come from a future update")
        object.__setattr__(self, "league_snapshots", tuple(self.league_snapshots))
        object.__setattr__(self, "holdouts", tuple(self.holdouts))
        object.__setattr__(self, "metrics", tuple(self.metrics))


@dataclass(frozen=True)
class LoadedUpdate:
    model: Any
    optimizer_state: Mapping[str, Any]
    state: RunState
    path: Path
    rng_restore_notes: Tuple[str, ...] = ()
    critic_state: Optional[Mapping[str, Any]] = None
    critic_optimizer_state: Optional[Mapping[str, Any]] = None


def capture_rng_states():
    states = {"python": random.getstate(), "numpy": np.random.get_state(),
              "torch_cpu": torch.get_rng_state().clone()}
    if torch.cuda.is_initialized():
        states["torch_cuda"] = torch.cuda.get_rng_state_all()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        states["torch_mps"] = torch.mps.get_rng_state()
    return states


def restore_rng_states(states, *, device: Optional[str] = None) -> Tuple[str, ...]:
    """Restore available streams; explicit target permits inactive missing backends.

    With no target, restoration is strict. A nonempty returned note tuple means
    cross-backend continuation is supported, but exact future draws are not claimed.
    Source states are never modified or discarded by this function.
    """
    target = None if device is None else torch.device(device).type
    notes = []
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch_cpu"].cpu())
    for key, backend, available, restore in (
            ("torch_cuda", "cuda", torch.cuda.is_available, torch.cuda.set_rng_state_all),
            ("torch_mps", "mps", torch.backends.mps.is_available, torch.mps.set_rng_state)):
        if key not in states:
            continue
        if available():
            restore(states[key])
        elif target is None or target == backend:
            raise ArtifactError("saved %s RNG requires its runtime for exact resume" % backend.upper())
        else:
            notes.append("%s RNG was not restored: backend unavailable; continuation on %s is not exact."
                         % (backend.upper(), target))
    return tuple(notes)


def _serialize(value):
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


def _write_bytes(path, data):
    with Path(path).open("wb") as output:
        output.write(data)
        output.flush()
        os.fsync(output.fileno())


def _sync_dir(path):
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _pointer(run_dir, name, target):
    fd, temporary = tempfile.mkstemp(prefix="." + name + "-", dir=str(run_dir))
    os.close(fd)
    temporary = Path(temporary)
    try:
        _write_bytes(temporary, json_bytes({"update": target.name}))
        os.replace(temporary, run_dir / name)
        _sync_dir(run_dir)
    finally:
        temporary.unlink(missing_ok=True)


def _config(payload):
    payload = dict(payload)
    for name, cls in _SECTIONS.items():
        payload[name] = cls(**payload[name])
    return TrainConfig(**payload)


def _compatible_config(saved, requested):
    a, b = asdict(saved), asdict(requested)
    # Runtime device, wall time and checkpoint margin are operational overrides.
    # Eval interval is
    # strength-evaluation cadence, NOT log cadence, and is not overridable.
    for name in ("device", "max_wall_time_seconds", "checkpoint_margin_seconds"):
        a["runtime"].pop(name)
        b["runtime"].pop(name)
    if a != b:
        raise ArtifactError("resume training config mismatch (only device/wall time/checkpoint margin may change)")


def restore_league(state):
    streams = SeedStreams(state.master_seed)
    if state.dynamic_league is not None:
        from .dynamic_league import DynamicLeague
        if state.dynamic_league['next_game_id'] != state.next_game_id:
            raise ValueError('dynamic league cursor differs from training counter')
        return DynamicLeague.restore(streams, state.league_snapshots, state.holdouts,
                                     state.league_history, state.dynamic_league)
    return League.restore(streams, state.league_snapshots, state.config.league,
                          state.holdouts, state.league_history)


def _verify_league(state):
    if state.league_snapshots or state.holdouts or state.league_history or state.dynamic_league is not None:
        try:
            restore_league(state)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ArtifactError("invalid immutable league state") from exc


def store_snapshot_records(run_dir, state, model):
    """Validate the snapshot tensors and durably store original content-addressed bytes."""
    store = run_dir / "weights"
    store.mkdir(exist_ok=True)
    records = []
    expected = model.state_dict()
    for snapshot in state.league_snapshots + state.holdouts:
        try:
            data = snapshot.weights_path.read_bytes()
            weights = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
            if (not isinstance(weights, dict) or set(weights) != set(expected) or
                    any(not isinstance(weights[k], torch.Tensor) or weights[k].shape != v.shape or
                        weights[k].dtype != v.dtype or not torch.isfinite(weights[k]).all().item()
                        for k, v in expected.items())):
                raise ArtifactError("snapshot is not a compatible finite tensor state_dict")
            for name, value in model.named_buffers():
                if not torch.equal(weights[name], value.cpu()):
                    raise ArtifactError("snapshot static board mismatch")
        except (OSError, RuntimeError, EOFError, pickle.UnpicklingError) as exc:
            raise ArtifactError("cannot read immutable snapshot " + snapshot.id) from exc
        digest = sha256(data)
        target = store / (digest + ".pt")
        if target.exists():
            if sha256(target.read_bytes()) != digest:
                raise ArtifactError("content-addressed weights were modified")
        else:
            fd, temp = tempfile.mkstemp(prefix=".weights-", dir=str(store))
            os.close(fd)
            try:
                _write_bytes(temp, data)
                os.replace(temp, target)
                _sync_dir(store)
            finally:
                Path(temp).unlink(missing_ok=True)
        records.append({"id": snapshot.id, "update": snapshot.update, "role": snapshot.role, "sha256": digest})
    return records


def _controller(state, records, rng):
    controller = {"config": asdict(state.config), "master_seed": state.master_seed,
            "controller_rng_states": dict(state.controller_rng_states),
            "next_game_id": state.next_game_id, "completed_updates": state.completed_updates,
            "snapshots": records, "development_best_id": state.development_best_id,
            "league_history": dict(state.league_history), "rng_states": rng}
    if "stability_arm" in state.manifests:
        controller["stability_arm"] = state.manifests["stability_arm"]
    if state.dynamic_league is not None:
        controller["dynamic_league"] = dict(state.dynamic_league)
    return controller


def _validate_update(path, staging_update=None, *, allow_retired=False):
    manifest = read_json(path / "manifest.json")
    if manifest.get("format") != UPDATE_FORMAT or set(manifest.get("content_hashes", {})) != UPDATE_FILES:
        raise ArtifactError("not a complete update manifest")
    try:
        available = present_payloads(path, manifest)
    except ValueError as exc:
        raise ArtifactError(str(exc)) from exc
    if 'optimizer.pt' not in available and not allow_retired:
        raise ArtifactError("Adam checkpoint intentionally retired; resume requires the latest committed full checkpoint; this update is playable only: "+str(path))
    payload = {name: verified_bytes(path, manifest["content_hashes"], name) for name in available}
    for record in manifest.get("snapshots", []):
        digest = record["sha256"]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ArtifactError("invalid immutable weight reference")
        weight_path = path.parent.parent / "weights" / (digest + ".pt")
        try:
            if sha256(weight_path.read_bytes()) != digest:
                raise ArtifactError("immutable snapshot content mismatch")
        except OSError as exc:
            raise ArtifactError("missing immutable snapshot " + record["id"]) from exc
    # Full policy compatibility/tensor checks, without consuming training RNG.
    model, _, metadata = load_policy(path)
    expected_name = path.name if staging_update is None else "%06d" % staging_update
    if expected_name != "%06d" % metadata.update or manifest.get("update") != metadata.update:
        raise ArtifactError("update directory/counter mismatch")
    return manifest, payload, model, metadata


def _committed(run_dir):
    # Hidden temp directories are never candidates. A damaged committed numeric
    # directory is an error, never a silent rollback to an older random policy.
    updates = run_dir / "updates"
    if not updates.is_dir():
        raise ArtifactError("no committed updates")
    paths = sorted((p for p in updates.iterdir() if p.is_dir() and p.name.isdecimal()), key=lambda p: int(p.name))
    if not paths:
        raise ArtifactError("no committed updates")
    return paths


def _resolve_update(path):
    path = Path(path).resolve()
    if path.is_dir():
        return path
    if path.name not in ("latest", "best"):
        raise ArtifactError("expected complete update directory or latest/best pointer")
    candidates = _committed(path.parent)
    # latest is derived from the journal even when pointer missing/stale/corrupt.
    latest = candidates[-1]
    _, payload, _, _ = _validate_update(latest)
    if path.name == "latest":
        return latest
    controller = torch.load(io.BytesIO(payload["controller.pt"]), map_location="cpu", weights_only=False)
    best_id = controller["development_best_id"]
    if best_id is None:
        raise ArtifactError("no development-best checkpoint recorded")
    return _development_best(reversed(candidates), best_id)


def _development_best(candidates, best_id):
    for candidate in candidates:
        _, payload, _, metadata = _validate_update(candidate, allow_retired=True)
        if metadata.id == best_id:
            if 'optimizer.pt' not in payload:
                raise ArtifactError("development-best checkpoint is retired")
            return candidate
    raise ArtifactError("development-best checkpoint is missing")


def policy_id(model, update: int) -> str:
    """Deterministic public identity for marking this update as development-best."""
    return policy_metadata(model, update, sha256(tensor_bytes(model.state_dict())))["id"]


def _validate_optimizer(model, optimizer_state, learning_rate):
    probe = torch.optim.Adam(model.parameters(), lr=learning_rate)
    try:
        probe.load_state_dict(optimizer_state)
        for parameter, values in probe.state.items():
            for name, value in values.items():
                if isinstance(value, torch.Tensor):
                    if not torch.isfinite(value).all().item():
                        raise ArtifactError("optimizer contains nonfinite state")
                    if name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq") and value.shape != parameter.shape:
                        raise ArtifactError("optimizer tensor shape mismatch")
    except (ValueError, KeyError, TypeError) as exc:
        raise ArtifactError("invalid Adam optimizer state") from exc


def save_update(run_dir: Path, model, optimizer, state: RunState, *, inherited_policy=None,
                critic=None, critic_optimizer=None, inherit_source_rng=False) -> Path:
    """Commit one complete boundary, preserving every caller RNG stream.

    Single writer only. Repeating an identical boundary is idempotent; an existing
    counter with different policy/controller state is rejected. Global RNG on the
    first commit wins for a retry. state.rng_states is resume output, not save input.
    """
    rng = capture_rng_states()
    temporary = None
    try:
        if model.config != state.config.model or not isinstance(optimizer, torch.optim.Adam):
            raise ArtifactError("checkpoint requires matching model config and Adam optimizer")
        # Adam state is positional. Resume constructs Adam(model.parameters()),
        # so equal membership alone cannot preserve moment/parameter association.
        if (len(optimizer.param_groups) != 1 or
                [id(p) for p in optimizer.param_groups[0]["params"]] !=
                [id(p) for p in model.parameters()]):
            raise ArtifactError("optimizer requires one parameter group in canonical model order")
        _validate_optimizer(model, optimizer.state_dict(), state.config.ppo.learning_rate)
        arm = state.manifests.get("stability_arm")
        if arm == "candidate":
            from .stability_checkpoint import pack
            optimizer_payload = pack(model, optimizer, critic, critic_optimizer, state.config.ppo.learning_rate)
        elif arm in (None, "control") and critic is None and critic_optimizer is None:
            optimizer_payload = optimizer.state_dict()
        else:
            raise ArtifactError("incompatible stability arm/critic checkpoint")
        checkpoint_rng = rng
        if inherit_source_rng and inherited_policy is None:
            raise ArtifactError("inheriting RNG requires an explicit source checkpoint")
        run_dir = Path(run_dir).resolve()
        updates = run_dir / "updates"
        updates.mkdir(parents=True, exist_ok=True)
        target = updates / ("%06d" % state.completed_updates)
        if inherited_policy is None:
            weights = tensor_bytes(model.state_dict())
            metadata = policy_metadata(model, state.completed_updates, sha256(weights))
            metadata_bytes = json_bytes(metadata)
        else:
            # A full-state fork can inherit a policy serialized by another OS.
            # Re-encoding equal tensors locally may assign a different public ID.
            _, payload, inherited, identity = _validate_update(Path(inherited_policy))
            if (identity.update != state.completed_updates or inherited.config != model.config or
                    any(not torch.equal(v.cpu(), inherited.state_dict()[k]) for k, v in model.state_dict().items())):
                raise ArtifactError("inherited public policy differs from fork state")
            weights, metadata_bytes = payload["policy.pt"], payload["metadata.json"]
            metadata = json.loads(metadata_bytes)
            if inherit_source_rng:
                source_controller = torch.load(io.BytesIO(payload["controller.pt"]), map_location="cpu", weights_only=False)
                checkpoint_rng = source_controller["rng_states"]
        if state.development_best_id is not None and state.development_best_id != metadata["id"]:
            candidates = _committed(run_dir)
            _development_best(candidates, state.development_best_id)
        _verify_league(state)
        records = store_snapshot_records(run_dir, state, model)
        controller = _controller(state, records, checkpoint_rng)
        data = {"policy.pt": weights, "metadata.json": metadata_bytes,
                "controller.pt": _serialize(controller), "optimizer.pt": _serialize(optimizer_payload),
                "metrics.json": json_bytes({"records": state.metrics}), "manifests.json": json_bytes(dict(state.manifests)),
                "versions.json": json_bytes({"python": platform.python_version(), "torch": str(torch.__version__),
                                              "numpy": np.__version__, "system": platform.system(), "source": source_provenance()})}
        if target.exists():
            _, old, _, _ = _validate_update(target)
            old_controller = torch.load(io.BytesIO(old["controller.pt"]), map_location="cpu", weights_only=False)
            # Compare trusted state without global RNG; state dictionaries contain
            # arbitrary caller-owned RNG tuples/tensors, so use serialization.
            old_controller.pop("rng_states")
            comparable = dict(controller)
            comparable.pop("rng_states")
            if (_serialize(old_controller) != _serialize(comparable) or
                    any(old[k] != data[k] for k in UPDATE_FILES - {"controller.pt", "versions.json"})):
                raise ArtifactError("completed update already exists with different content")
        else:
            existing = [int(p.name) for p in updates.iterdir() if p.is_dir() and p.name.isdecimal()]
            if existing and state.completed_updates != max(existing) + 1:
                raise ArtifactError("completed update counter must advance by one")
            temporary = Path(tempfile.mkdtemp(prefix=".update-", dir=str(updates)))
            for name, content in data.items():
                _write_bytes(temporary / name, content)
            manifest = {"format": UPDATE_FORMAT, "update": state.completed_updates,
                        "content_hashes": {name: sha256(content) for name, content in data.items()}, "snapshots": records}
            _write_bytes(temporary / "manifest.json", json_bytes(manifest))
            # Hash every written payload/reference before the one commit point.
            _validate_update(temporary, staging_update=state.completed_updates)
            _sync_dir(temporary)
            os.replace(temporary, target)
            temporary = None
            _sync_dir(updates)
        _pointer(run_dir, "latest", _committed(run_dir)[-1])
        if state.development_best_id is not None:
            selected = _resolve_update(run_dir / "best")
            pin(selected)
            _pointer(run_dir, "best", selected)
        return target
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
        restore_rng_states(rng)


def _checkpoint_state(path, manifest, payload, model, metadata, config):
    controller = torch.load(io.BytesIO(payload["controller.pt"]), map_location="cpu", weights_only=False)
    saved_config = _config(controller["config"])
    if config is not None:
        _compatible_config(saved_config, config)
    snapshots = manifest["snapshots"]
    if snapshots != controller["snapshots"] or controller["completed_updates"] != metadata.update:
        raise ArtifactError("controller/update manifest mismatch")
    if saved_config.model != model.config:
        raise ArtifactError("controller model config mismatch")
    refs = tuple(Snapshot(r["id"], path.parent.parent / "weights" / (r["sha256"] + ".pt"), r["update"], r["role"])
                 for r in snapshots)
    state = RunState(config=config or saved_config, master_seed=controller["master_seed"],
                     controller_rng_states=controller["controller_rng_states"],
                     next_game_id=controller["next_game_id"], completed_updates=controller["completed_updates"],
                     league_snapshots=tuple(s for s in refs if s.role != "holdout"),
                     holdouts=tuple(s for s in refs if s.role == "holdout"),
                     development_best_id=controller["development_best_id"],
                     metrics=tuple(read_json(path / "metrics.json")["records"]),
                     manifests=read_json(path / "manifests.json"), league_history=controller["league_history"],
                     rng_states=controller["rng_states"], dynamic_league=controller.get("dynamic_league"))
    arm = state.manifests.get("stability_arm")
    if (controller.get("stability_arm") != arm or arm not in (None, "control", "candidate")
            or ("stability_arm" in controller) != ("stability_arm" in state.manifests)):
        raise ArtifactError("stability arm differs between controller and committed manifest")
    _verify_league(state)
    if state.development_best_id is not None:
        _development_best(reversed(_committed(path.parent.parent)), state.development_best_id)
    return state, saved_config


def inspect_update_state(path, config=None):
    """Trusted read-only study inspection, including the full Adam checks when available."""
    path = Path(path)
    before = capture_rng_states()
    try:
        if (path/'optimizer.pt').is_file():
            return load_update(path, config=config).state
        manifest, payload, model, metadata = _validate_update(path, allow_retired=True)
        state, _ = _checkpoint_state(path, manifest, payload, model, metadata, config)
        restore_rng_states(state.rng_states, device='cpu')
        return state
    finally:
        restore_rng_states(before)


def load_update(path: Path, device: str = "cpu", config: Optional[TrainConfig] = None) -> LoadedUpdate:
    """Load trusted local training state; restore RNG only after all construction.

    Caller builds Adam(loaded.model.parameters()), then loads optimizer_state.
    Adam construction/load do not consume RNG; create other random objects before
    this call, or restore loaded.state.rng_states again with device=device after
    their construction. Surface rng_restore_notes when continuation is nonexact.
    """
    before = capture_rng_states()
    try:
        path = _resolve_update(path)
        manifest, payload, model, metadata = _validate_update(path)
        state, saved_config = _checkpoint_state(path, manifest, payload, model, metadata, config)
        optimizer_state = torch.load(io.BytesIO(payload["optimizer.pt"]), map_location=device, weights_only=False)
        model.to(device).train().requires_grad_(True)
        critic = critic_adam = None
        if state.manifests.get("stability_arm") == "candidate":
            from .stability_checkpoint import unpack
            optimizer_state, critic, critic_adam = unpack(model, optimizer_state, state.config.ppo.learning_rate)
        elif "format" in optimizer_state:
            raise ArtifactError("unexpected versioned optimizer outside candidate stability checkpoint")
        # Validate optimizer shape/load semantics before restoring any RNG.
        _validate_optimizer(model, optimizer_state, state.config.ppo.learning_rate)
        notes = restore_rng_states(state.rng_states, device=device)
        versions = read_json(path / "versions.json")
        current = {"python": platform.python_version(), "torch": str(torch.__version__),
                   "numpy": np.__version__, "system": platform.system()}
        changed = [name for name, value in current.items() if name in versions and versions[name] != value]
        if changed:
            notes += ("Runtime changed (%s); exact numerical reproduction is not guaranteed." % ", ".join(changed),)
        if saved_config.runtime.device != torch.device(device).type:
            notes += ("Training device changed from %s to %s; continuation is not exact."
                      % (saved_config.runtime.device, torch.device(device).type),)
        return LoadedUpdate(model, optimizer_state, state, path, notes, critic, critic_adam)
    except Exception as exc:
        restore_rng_states(before)
        if isinstance(exc, ArtifactError):
            raise
        raise ArtifactError("invalid completed update: %s" % path) from exc


def export_model_policy(model, update: int, dest: Path) -> Path:
    """Pure canonical three-file export for evaluation of uncommitted weights.

    This publishes no training/best pointer. Its identity is exactly policy_id.
    """
    before = capture_rng_states()
    temporary = None
    try:
        weights = tensor_bytes(model.state_dict())
        metadata = policy_metadata(model, update, sha256(weights))
        dest = Path(dest).resolve()
        if dest.exists():
            if inspect_policy(dest).id != metadata['id']:
                raise ArtifactError('export destination already contains a different policy')
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix='.policy-', dir=str(dest.parent)))
        payload = {'policy.pt': weights, 'metadata.json': json_bytes(metadata)}
        for name, data in payload.items():
            _write_bytes(temporary/name, data)
        _write_bytes(temporary/'manifest.json', json_bytes({'format': POLICY_FORMAT,
            'content_hashes': {name: sha256(data) for name, data in payload.items()}}))
        inspect_policy(temporary)
        _sync_dir(temporary)
        os.replace(temporary, dest)
        temporary = None
        _sync_dir(dest.parent)
        return dest
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
        restore_rng_states(before)


def export_policy(update_dir: Path, dest: Path) -> Path:
    """Publish exact validated committed bytes, preserving historical identities.

    Torch serialization is not a portable canonical tensor encoding. Re-saving
    tensors may change archive bytes across runtimes, so it must never redefine
    the identity of an already committed policy.
    """
    before = capture_rng_states()
    temporary = None
    try:
        source = _resolve_update(update_dir)
        _, payload, _, metadata = _validate_update(source, allow_retired=True)
        dest = Path(dest).resolve()
        if dest.exists():
            actual = inspect_policy(dest)
            if actual.id != metadata.id or any((dest/n).read_bytes() != payload[n]
                                               for n in ('policy.pt', 'metadata.json')):
                raise ArtifactError('export destination already contains a different policy')
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix='.policy-', dir=str(dest.parent)))
        for name in ('policy.pt', 'metadata.json'):
            # Hard links are safe only for immutable public payloads.
            try:
                os.link(source/name, temporary/name)
            except OSError:
                _write_bytes(temporary/name, payload[name])
        _write_bytes(temporary/'manifest.json', json_bytes({'format': POLICY_FORMAT,
            'content_hashes': {n: sha256(payload[n]) for n in ('policy.pt', 'metadata.json')}}))
        inspect_policy(temporary)
        _sync_dir(temporary)
        os.replace(temporary, dest)
        temporary = None
        _sync_dir(dest.parent)
        return dest
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
        restore_rng_states(before)


def committed_records(run_dir: Path) -> Tuple[Mapping[str, Any], ...]:
    """Read journal metrics/manifests without constructing models or restoring RNG.

    Numeric directories are authoritative; staging directories and pointer caches
    are ignored. This focused metadata reader verifies the committed envelope and
    selected payload hashes. Full model/optimizer/reference validation remains the
    responsibility of load_update. No trusted controller pickle is opened here.
    """
    records = []
    for path in _committed(Path(run_dir)):
        try:
            manifest = read_json(path / "manifest.json")
            update = manifest.get("update")
            if (manifest.get("format") != UPDATE_FORMAT or type(update) is not int or update < 0 or
                    path.name != "%06d" % update or
                    set(manifest.get("content_hashes", {})) != UPDATE_FILES):
                raise ArtifactError("invalid committed journal envelope: %s" % path)
            present_payloads(path, manifest)
            metrics = json.loads(verified_bytes(path, manifest["content_hashes"], "metrics.json"))
            manifests = json.loads(verified_bytes(path, manifest["content_hashes"], "manifests.json"))
            if (not isinstance(metrics, dict) or set(metrics) != {"records"} or
                    not isinstance(metrics["records"], list) or
                    not all(isinstance(row, dict) for row in metrics["records"]) or
                    not isinstance(manifests, dict)):
                raise ArtifactError("invalid committed metrics/manifests: %s" % path)
            json_bytes(metrics)
            json_bytes(manifests)  # finite JSON, including trusted handcrafted input
            records.append({"update": update, "metrics": tuple(metrics["records"]), "manifests": manifests})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ArtifactError("invalid committed journal metadata: %s" % path) from exc
    return tuple(records)
