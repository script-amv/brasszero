# SPDX-License-Identifier: Apache-2.0
"""Build reviewable model and replay distributions locally. Originals are opened read-only."""

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import date
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine/tools"))
from brass_ai.replay_archive import Archive, canonical_game, seat_identities  # noqa: E402

QA = re.compile(r"(^|[/_.\-\s])(validation|qa|tests?|fixtures?|smoke|task\d+|portability)([/_.\-\s]|$)", re.I)


def qa_metadata(value):
    if isinstance(value, str):
        return bool(QA.search(value))
    if isinstance(value, dict):
        return any(qa_metadata(x) for x in value.values())
    if isinstance(value, list):
        return any(qa_metadata(x) for x in value)
    return False


def sha(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@contextmanager
def archive_writer(path, confirm):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".build-")
    try:
        with (
            os.fdopen(fd, "wb") as raw,
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=4) as gz,
        ):
            with tarfile.open(mode="w|", fileobj=gz) as tar:
                yield tar
        if path.exists():
            if sha_file(path) == sha_file(Path(name)):
                return
            if not confirm:
                raise FileExistsError(
                    f"Different local release already exists: {path}. Review before using CONFIRM=1."
                )
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def member(tar, name, data):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = 0o644
    tar.addfile(info, io.BytesIO(data))


def write_file(path, payload):
    if path.is_file() and path.read_bytes() == payload:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        f.write(payload)
    Path(f.name).replace(path)


# The replay writer can attach an arbitrary operator label to version_id. These
# closed forms are emitted by the policy, search, league and native baseline code.
PUBLIC_VERSIONS = {
    "ai": re.compile(
        r"(?:policy-sha256-v1:|search-sha256-v1:|(?:initial|training)-[0-9]{6,}:weights-sha256:)[0-9a-f]{64}"
    ),
    "greedy": re.compile(r"greedy:native-v1:[0-9a-f]{64}"),
    "random": re.compile(r"random:native-v1:[0-9a-f]{64}"),
}


def public_record(record):
    p = record["provenance"]
    if (
        record["category"] in {"qa", "validation", "test"}
        or p.get("validation_fixture")
        or p.get("test_fixture")
        or qa_metadata(p)
    ):
        return "qa/validation", None
    if record["status"] != "completed" or not record.get("result"):
        return "incomplete", None
    if record.get("duplicate_of"):
        return "duplicate alias", None
    if record["category"] not in {"training", "evaluation"}:
        return "outside training/evaluation", None
    if (
        p.get("parent") is not None
        or p.get("cursor") is not None
        or any(x.get("ply") != 0 or x.get("seats") != record["seats"] for x in record.get("lineups", []))
    ):
        return "changed lineup or branch", None
    identities = record.get("seat_identities") or seat_identities(record["seats"], legacy=True)
    if len(identities) != 4 or any(x["kind"] not in {"ai", "random", "greedy", "bot"} for x in identities):
        return "human or unknown participant", None
    seats = []
    for row in identities:
        version = row.get("version_id")
        # Do not ship arbitrary display names, hostnames or paths as agent IDs.
        form = PUBLIC_VERSIONS.get(row["kind"])
        if form is None or not isinstance(version, str) or not form.fullmatch(version):
            version = None
        seats.append({"kind": row["kind"], "version_id": version})
    game = canonical_game(record)
    if set(game["initial"]) != {"seed"}:
        return "explicit starting state", None
    # Only native seed/actions/results and whitelisted scientific identifiers go out.
    # Historical provenance may contain local paths, human labels and provider receipts.
    result = record["result"]
    return "included", {
        "format": "brasszero-public-game/1",
        "game_id": record["game_id"],
        "engine_id": record["engine_id"],
        "category": record["category"],
        "initial": game["initial"],
        "seats": seats,
        "moves": game["moves"],
        "result": {k: result[k] for k in ("ranking", "vp", "winner")},
    }


def dataset(confirm=False):
    games = ROOT / "replays/games"
    if not games.is_dir():
        raise ValueError("local recording archive is required")
    out = ROOT / "dist/dataset"
    archive = Archive(games.parent)
    ids = sorted(
        {
            p.name.split(".")[0]
            for p in games.iterdir()
            if re.fullmatch(r"[a-f0-9]{32,64}\.json(?:\.gz|l)", p.name)
        }
    )
    name = "brasszero-games-" + date.today().isoformat() + ".tar.gz"
    counts = Counter()
    manifest = []
    with archive_writer(out / name, confirm) as tar:
        for key in ids:
            record = archive.read(key)
            reason, game = public_record(record)
            counts[reason] += 1
            if not game:
                continue
            if game["game_id"] != key:
                raise ValueError("record ID does not match its archive path")
            data = (json.dumps(game, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
            name_in = "games/" + key + ".json"
            member(tar, name_in, data)
            manifest.append(sha(data) + "  " + name_in)
        if not counts["included"]:
            raise ValueError("no eligible games")
        sheet = (
            (ROOT / "docs/DATASHEET.md").read_text()
            + "\n## Snapshot population\n\n"
            + "".join(f"- {k}: {v}\n" for k, v in sorted(counts.items()))
        )
        licensing = (ROOT / "docs/datasheets/CC-BY-4.0.txt").read_bytes()
        for path, data in [("DATASHEET.md", sheet.encode()), ("LICENSE.txt", licensing)]:
            member(tar, path, data)
            manifest.append(sha(data) + "  " + path)
        hashes = ("\n".join(manifest) + "\n").encode()
        member(tar, "MANIFEST.sha256", hashes)
    for file, data in [
        ("MANIFEST.sha256", hashes),
        ("DATASHEET.md", sheet.encode()),
        ("LICENSE.txt", licensing),
    ]:
        write_file(out / file, data)
    print("Archive", out / name, sha_file(out / name), dict(sorted(counts.items())))
    print(
        "Review the archive and datasheet. In Zenodo create a new version, upload the archive and checksum manifest, choose CC BY 4.0, then publish after review."
    )


def lineage_notes(run, experiment, branch):
    """State the opponent mixture and parent changes recorded by the run itself."""
    notes = []
    config_path = run / "config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_bytes())
        mix = config.get("league", {}).get("lineup_probabilities")
        if mix:
            own, frozen, mixed = (round(100 * x) for x in mix)
            notes.append(
                f"Each training lineup was drawn with probability {own}% for self-play in all four seats, "
                f"{frozen}% for one learning seat against three frozen snapshots, and {mixed}% for one learning "
                "seat against three opponents drawn from the snapshots and the rule-based Random and Greedy "
                "players. Greedy is therefore also a training opponent."
            )
    changes = experiment.get("config_overrides") or {}
    if changes:
        listed = ", ".join(f"`{k}` from `{v['from']}` to `{v['to']}`" for k, v in sorted(changes.items()))
        notes.append(f"Compared with its parent run, this branch changed {listed}.")
        if branch == "BC" and {"ppo.reward", "ppo.critic_target"} <= set(changes):
            notes.append(
                "`BC` is the study's label for these two changes. It does not mean behaviour cloning."
            )
    return " ".join(notes)


def model_evaluation(run, metadata):
    """Use a finalized, ID-bound matched study when the local journal supplies one."""
    paths = [
        run.parent / "result.json",
        run.parent / "study.json",
        run.parent / "evaluation" / run.name / "suite.json",
    ]
    fallback = (
        "TODO: confirm source experiment game count and league.",
        "TODO: attach a frozen evaluation identified by the exact policy and suite.",
    )
    if not all(p.is_file() for p in paths):
        return fallback
    result, study, suite = (json.loads(p.read_bytes()) for p in paths)
    endpoint = result.get("endpoints", {}).get(run.name, {})
    if endpoint.get("id") != metadata["id"]:
        return fallback
    rows = result.get("policies", {}).get(run.name, {})
    metrics = []
    for name, row in sorted(rows.items()):
        metrics.append(
            f"| {name} | {row['deals']} | {row['games']} | {row['mean_vp']:.2f} | {row['strict_wins']} |"
        )
    if not metrics:
        return fallback
    evaluation = [
        f"Frozen suite `{suite['suite_id']}` used {suite['rotations']} seat rotations per deal.",
        "",
        "| Opponents | Deals | Games | Mean VP | Sole wins |",
        "| --- | ---: | ---: | ---: | ---: |",
        *metrics,
    ]
    comparison = (
        result.get("paired", {}).get(run.name, {}).get("control", {}).get("frozen688", {}).get("mean_vp")
    )
    if comparison:
        lo, hi = comparison["bootstrap_95"]
        evaluation.extend(
            [
                "",
                f"Against frozen 688, the mean-VP difference from the control was {comparison['mean_difference']:+.2f} "
                f"(paired 95% interval {lo:+.2f} to {hi:+.2f}). The interval crosses zero, so this is a trend, not an established win."
                if lo <= 0 <= hi
                else f"Against frozen 688, paired mean-VP difference from control was {comparison['mean_difference']:+.2f} (95% interval {lo:+.2f} to {hi:+.2f}).",
            ]
        )
    compared = sorted(result.get("policies", {}))
    if len(compared) > 1:
        evaluation.extend(
            [
                "",
                "The same deals compared the policies "
                + ", ".join(compared)
                + ". Numbers for a policy picked after such a comparison may be optimistic.",
            ]
        )
    evaluation.extend(["", "Deal seeds: " + ", ".join(str(v) for v in suite["deal_seeds"]) + "."])
    training = f"The branch played {study['games_per_arm']} training games during the paired training study. Earlier games belong to the inherited parent lineage."
    return training, "\n".join(evaluation)


def model(identifier, confirm=False):
    if not identifier:
        raise ValueError("set MODEL to a catalog label or exact policy ID")
    catalog = json.loads((ROOT / "players/models/catalog.json").read_bytes())
    rows = catalog["models"] + catalog.get("other_preserved_snapshots", [])
    match = [r for r in rows if identifier in (r.get("policy_id"), r.get("name"))]
    if len(match) != 1:
        raise ValueError("model identifier is absent or ambiguous")
    row = match[0]
    label = row["name"]
    if not re.fullmatch("[A-Za-z0-9_.-]+", label):
        raise ValueError("unsafe catalog label")
    folder = (ROOT / row["path"]).resolve()
    if (ROOT / "players/models").resolve() not in folder.parents:
        raise ValueError("model outside local catalog")
    paths = [folder / name for name in ("manifest.json", "metadata.json", "policy.pt")]
    if sum(p.stat().st_size for p in paths) > 2 * 1024**3:
        raise ValueError("model exceeds 2 GiB")
    sums = {p.name: sha_file(p) for p in paths}
    metadata = json.loads(paths[1].read_bytes())
    expected = json.loads(paths[0].read_bytes())["content_hashes"]
    if (
        sums != row["files"]
        or metadata["id"] != row["policy_id"]
        or any(sums[n] != s for n, s in expected.items())
    ):
        raise ValueError("catalog, manifest and model disagree")
    cfg = metadata["model_config"]
    version = metadata["update"]
    experiment = {}
    provenance = row.get("provenance") or {}
    run = ROOT / (provenance.get("run") or "training/runs/absent")
    if (ROOT / "training/runs").resolve() in run.resolve().parents and (run / "experiment.json").is_file():
        experiment = json.loads((run / "experiment.json").read_bytes())
    parent = experiment.get("parent") or {}
    parent_label = (
        f"Update {parent['update']}, policy SHA-256 {parent['policy_sha256']}"
        if "update" in parent and "policy_sha256" in parent
        else "TODO: verify parent from journal"
    )
    card = (ROOT / "docs/MODEL_CARD_TEMPLATE.md").read_text()
    training, evaluation = model_evaluation(run, metadata)
    values = {
        "MODEL": label,
        "POLICY": metadata["id"],
        "UPDATE": str(version),
        "LINEAGE": str(row.get("lineage", "TODO")),
        "PARENT": parent_label,
        "DATE": (
            experiment["created_at"][:10]
            if experiment.get("created_at")
            else "TODO: verify experiment start date"
        ),
        "BRANCH": str(row.get("branch", "TODO")),
        "ENGINE": metadata["engine_fingerprint"],
        "SCHEMA": metadata["schema_id"],
        "CONFIG": json.dumps(cfg, sort_keys=True),
        "WEIGHTS": sums["policy.pt"],
        "PARAMETERS": str(parameter_count(paths[-1])),
        "EVALUATION": evaluation,
        "TRAINING": " ".join(filter(None, [training, lineage_notes(run, experiment, row.get("branch"))])),
    }
    for key, value in values.items():
        card = card.replace("{{" + key + "}}", value)
    out = ROOT / "dist/models"
    archive = out / (label + ".tar.gz")
    card_path = ROOT / "docs/model-cards" / (label + ".md")
    if card_path.exists() and card_path.read_text() != card and not confirm:
        raise FileExistsError(
            "Generated card differs from its reviewed copy. Preserve your edits or use CONFIRM=1."
        )
    with archive_writer(archive, confirm) as tar:
        for p in paths:
            # The 2 GiB gate is above. Stream tensor storage so packing does not duplicate it in memory.
            info = tarfile.TarInfo(label + "/policy/" + p.name)
            info.size = p.stat().st_size
            with p.open("rb") as stream:
                tar.addfile(info, stream)
        member(tar, label + "/MODEL_CARD.md", card.encode())
        member(tar, label + "/LICENSE", (ROOT / "LICENSE").read_bytes())
    write_file(card_path, card.encode())
    write_file(out / (archive.name + ".sha256"), (sha_file(archive) + "  " + archive.name + "\n").encode())
    print(f"Prepared {archive.relative_to(ROOT)}, weights {sums['policy.pt']}")
    print(
        f'gh release upload "${{RELEASE_TAG:?Set the reviewed release tag}}" dist/models/{archive.name} dist/models/{archive.name}.sha256'
    )


def parameter_count(weights):
    # Read the verified, local catalog weight file in the optional inference runtime.
    # The release builder must remain usable with only the standard library.
    import subprocess

    py = ROOT / "training/.venv/bin/python"
    if not py.is_file():
        return "TODO: count parameters in the verified inference environment"
    code = "import sys; from brass_rl.agent import load_agent; a=load_agent(sys.argv[1],1,device='cpu'); print(sum(p.numel() for p in a.model.parameters()))"
    return subprocess.check_output(
        [py, "-c", code, weights.parent],
        text=True,
        env=dict(os.environ, PYTHONPATH=os.pathsep.join(str(ROOT / p) for p in ("engine/tools", "training"))),
    ).strip()


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("kind", choices=["dataset", "model"])
    cli.add_argument("id", nargs="?")
    cli.add_argument(
        "--confirm", action="store_true", help="replace a different existing local release draft"
    )
    args = cli.parse_args()
    if args.kind == "model":
        model(args.id, args.confirm)
    else:
        dataset(args.confirm)
