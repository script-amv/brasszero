"""Strict portable policy boundary. Never imports or unpickles controller state."""
from dataclasses import dataclass
import hashlib
import io
import json
import pickle
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import torch

from brass_ai.fingerprints import engine_fingerprint
from .retention import present_payloads, UPDATE_FILES as UPDATE_PAYLOAD_FILES
from .config import ModelConfig
from .model import BrassPolicy
from .schema import ComponentData, canonical_json

ROOT = Path(__file__).resolve().parents[2]
POLICY_FORMAT = "brass-portable-policy-v1"
UPDATE_FORMAT = "brass-complete-update-v1"
POLICY_FILES = {"policy.pt", "metadata.json", "manifest.json"}


class ArtifactError(ValueError):
    """Missing, incomplete, corrupt, or incompatible checkpoint/policy."""


@dataclass(frozen=True)
class PolicyMetadata:
    id: str
    update: int
    model_id: str
    schema_id: str
    component_hash: str
    engine_fingerprint: str
    requires_history: bool
    content_hashes: Mapping[str, str]
    portable_path: Path


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return canonical_json(value).encode("utf-8") + b"\n"


def tensor_bytes(weights):
    # File-object serialization uses a constant archive name, never a private path.
    output = io.BytesIO()
    torch.save({k: v.detach().cpu().contiguous().clone() for k, v in sorted(weights.items())}, output)
    return output.getvalue()


def read_json(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("expected JSON object")
        return value
    except (OSError, ValueError) as exc:
        raise ArtifactError("invalid or missing JSON: %s" % path) from exc


def verified_bytes(path, hashes, name):
    try:
        data = (path / name).read_bytes()
        if sha256(data) != hashes[name]:
            raise ValueError("content hash mismatch")
        return data
    except (OSError, KeyError, ValueError, TypeError) as exc:
        raise ArtifactError("invalid or missing artifact: %s" % (path / name)) from exc


def policy_identity(metadata, weights_hash):
    fields = {k: v for k, v in metadata.items() if k != "id"}
    return "policy-sha256-v1:" + sha256(json_bytes({"metadata": fields, "weights": weights_hash}))


def policy_metadata(model, update, weights_hash):
    from dataclasses import asdict
    fields = {"format": POLICY_FORMAT, "update": update, "model_id": model.model_id,
              "model_config": asdict(model.config), "schema_id": model.schema.fingerprint,
              "schema": json.loads(model.schema.canonical_json()),
              "component_hash": model.schema.component_hash,
              "engine_fingerprint": engine_fingerprint(ROOT),
              "requires_history": model.config.use_history}
    fields["id"] = policy_identity(fields, weights_hash)
    return fields


def load_policy(path: Path, device: str = "cpu", portable_only: bool = False):
    """Validate envelope and tensors on CPU before moving an inference model.

    Inference from a committed training directory reads only its policy subset.
    No pointer resolution, RNG/controller deserialization, or random fallback.
    """
    path = Path(path).resolve()
    try:
        manifest_bytes = (path / "manifest.json").read_bytes()
        manifest = json.loads(manifest_bytes)
        if not isinstance(manifest, dict):
            raise ArtifactError("invalid policy manifest")
        allowed = (POLICY_FORMAT,) if portable_only else (POLICY_FORMAT, UPDATE_FORMAT)
        if manifest.get("format") not in allowed:
            raise ArtifactError("not a complete portable policy" if portable_only else "unknown artifact format")
        hashes = manifest["content_hashes"]
        if not isinstance(hashes, dict):
            raise ArtifactError("invalid content manifest")
        if manifest["format"] == POLICY_FORMAT:
            if (set(manifest) != {"format", "content_hashes"} or
                    set(hashes) != {"policy.pt", "metadata.json"} or
                    {p.name for p in path.iterdir()} != POLICY_FILES):
                raise ArtifactError("portable export must contain only declared policy files")
        else:
            try:
                present_payloads(path, manifest)
            except ValueError as exc:
                raise ArtifactError("incomplete training update directory") from exc
        raw = verified_bytes(path, hashes, "metadata.json")
        metadata = json.loads(raw)
        expected_fields = {"format", "update", "id", "model_id", "model_config", "schema_id",
                           "schema", "component_hash", "engine_fingerprint", "requires_history"}
        if set(metadata) != expected_fields or metadata["format"] != POLICY_FORMAT:
            raise ArtifactError("invalid declared policy metadata")
        if type(metadata["update"]) is not int or metadata["update"] < 0:
            raise ArtifactError("invalid policy update")
        if metadata["id"] != policy_identity(metadata, hashes["policy.pt"]):
            raise ArtifactError("policy identity mismatch")
        components = ComponentData.load()
        if (metadata["schema_id"] != components.schema.fingerprint or
                metadata["schema"] != json.loads(components.schema.canonical_json()) or
                metadata["component_hash"] != components.component_hash or
                metadata["engine_fingerprint"] != engine_fingerprint(ROOT)):
            raise ArtifactError("incompatible policy schema, components or engine fingerprint")
        config = ModelConfig(**metadata["model_config"])
        if type(metadata["requires_history"]) is not bool or metadata["requires_history"] != config.use_history:
            raise ArtifactError("history requirement disagrees with model")
        weights_data = verified_bytes(path, hashes, "policy.pt")
        weights = torch.load(io.BytesIO(weights_data), map_location="cpu", weights_only=True)
        if not isinstance(weights, dict) or not weights or not all(
                isinstance(k, str) and isinstance(v, torch.Tensor) and torch.isfinite(v).all().item()
                for k, v in weights.items()):
            raise ArtifactError("policy must contain only finite tensor state_dict")
        # Constructor initializes random weights; validation and inspection must
        # not advance the training stream, including on rejected artifacts.
        with torch.random.fork_rng(devices=[]):
            model = BrassPolicy(config, components.schema, components)
        if model.model_id != metadata["model_id"]:
            raise ArtifactError("model identity mismatch")
        expected = model.state_dict()
        if set(weights) != set(expected) or any(weights[k].shape != v.shape or weights[k].dtype != v.dtype
                                               for k, v in expected.items()):
            raise ArtifactError("policy tensor keys, shapes or dtypes mismatch")
        # Board buffers are a native-information boundary, not trainable payload.
        for name, value in model.named_buffers():
            if not torch.equal(weights[name], value):
                raise ArtifactError("policy static board buffer mismatch")
        model.load_state_dict(weights, strict=True)
        model.to(device).eval().requires_grad_(False)
        public = PolicyMetadata(**{k: metadata[k] for k in
            ("id", "update", "model_id", "schema_id", "component_hash", "engine_fingerprint", "requires_history")},
            content_hashes=MappingProxyType({"policy.pt": hashes["policy.pt"],
                                             "metadata.json": hashes["metadata.json"],
                                             "manifest.json": sha256(manifest_bytes)}),
            portable_path=path)
        return model, components, public
    except ArtifactError:
        raise
    except (OSError, KeyError, TypeError, ValueError, RuntimeError, EOFError, pickle.UnpicklingError) as exc:
        raise ArtifactError("invalid policy artifact: %s" % path) from exc


def inspect_policy(path: Path) -> PolicyMetadata:
    """Read-only complete portable export inspection using the inference validator."""
    return load_policy(path, "cpu", portable_only=True)[2]
