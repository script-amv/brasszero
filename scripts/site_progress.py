# SPDX-License-Identifier: Apache-2.0
"""Refresh public site data atomically from the private workspace and the approved ledger."""

import argparse
from collections import Counter
import datetime as dt
import json
from pathlib import Path
import re
import subprocess
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "site/public/data/training-record.json"
CLAIMS = ROOT / "site/src/data/claims.json"
LEDGER = ROOT / "lab/outreach/CLAIMS.md"
HOME = ROOT / "site/src/content/home.md"


def write_json(path, data):
    payload = (json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if path.is_file() and path.read_bytes() == payload:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        tmp.write(payload)
    Path(tmp.name).replace(path)


def approved():
    if not LEDGER.exists():
        return json.loads(CLAIMS.read_text())
    text = LEDGER.read_text()
    date = re.search(r"Last verified: (\d{4}-\d{2}-\d{2})", text).group(1)
    rows = {}
    for line in text.split("## Conditional")[0].splitlines():
        fields = [f.strip() for f in line.strip("|").split("|")]
        if len(fields) == 4 and re.fullmatch(r"C-\d+", fields[0]):
            rows[fields[0]] = fields[1]
    # C-07 contains a disputed expansion of the treatment name and is withheld.
    rows.pop("C-07", None)
    if not rows:
        raise ValueError("approved claims ledger has no entries")
    # Extract approved values only from their named, dated scientific claims.
    score = re.search(r"from ([\d.]+) VP .*? to ([\d.]+) VP", rows["C-16"])
    actions = re.search(r"from ([\d.]+)% .*? to ([\d.]+)%", rows["C-17"])
    record = re.search(r": (\d+) victory points", rows["C-09"])
    if not (score and actions and record):
        raise ValueError("headline ledger changed wording; review the approved measurements")
    headlines = {
        "score": f"Mean score rose from {score[1]} to {score[2]} victory points",
        "actions": f"Passing fell from {actions[1]}% to {actions[2]}% of main actions",
        "record": f"{record[1]} victory points",
    }
    patterns = {
        "C-01": (r"plays ([0-9,]+–[0-9,]+) complete random games", "games per second on one laptop CPU core"),
        "C-04": (r"^([0-9,]+) self-play training games", "self-play training games so far"),
        "C-06": (r"^([0-9,]+) playable model versions", "playable model versions in the dated inventory"),
    }
    facts = {}
    for claim, (pattern, label) in patterns.items():
        match = re.search(pattern, rows[claim])
        if not match:
            raise ValueError("review the homepage fact for " + claim)
        facts[claim] = {"value": match[1], "label": label}
    if "no paid compute" not in rows["C-14"]:
        raise ValueError("review the C-14 compute claim")
    facts["C-14"] = {"value": "0", "label": "paid compute on consumer hardware and free cloud CPUs"}
    return {"as_of": date, "claims": rows, "headlines": headlines, "facts": facts}


def frontmatter(path):
    lines = path.read_text().splitlines()
    if not lines or lines[0] != "---" or "---" not in lines[1:]:
        raise ValueError(f"missing Markdown front matter in {path.name}")
    return yaml.safe_load("\n".join(lines[1 : lines.index("---", 1)]))


def validate_current():
    rendered = json.loads(CLAIMS.read_text())
    if LEDGER.is_file() and rendered != approved():
        raise ValueError("rendered claims differ from the private ledger projection; run make site-data")
    validate(json.loads(RECORD.read_text()), rendered)


def validate(record, ledger):
    meta = record["meta"]
    home = frontmatter(HOME)
    status_date = dt.date.fromisoformat(str(home["status"]["updated"]))
    if status_date < dt.date.fromisoformat(meta["exported"]):
        raise ValueError("status date predates the newest chart data")
    scores = record["scores"]
    if any(
        type(u) is not int or u < 0 or type(v) is not int or c not in {"training", "evaluation", "other"}
        for u, v, c in scores
    ):
        raise ValueError("invalid update, score or QA/validation category")
    if len(scores) != meta["ai_seat_scores"] or max(u for u, v, c in scores) != meta["latest_update"]:
        raise ValueError("score sample coverage mismatch")
    by_update = Counter(u for u, _, _ in scores)
    audit = record["audit"]
    if [[u, n] for u, n in sorted(by_update.items())] != audit["known_update_counts"]:
        raise ValueError("unknown or missing update plotted as a measured observation")
    if audit["qa_rows"] != 0:
        raise ValueError("QA or validation row in exported population")
    for cell in record["score_mean"]:
        values = [v for u, v, _ in scores if cell["lo"] <= u <= cell["hi"]]
        if not values or len(values) != cell["n"] or round(sum(values) / len(values), 2) != cell["mean"]:
            raise ValueError("score chart sample or aggregate mismatch")
    required_metrics = {
        "actions",
        "built",
        "sources",
        "scored_twice",
        "loans",
        "income",
        "network",
        "unflipped",
    }
    for window in record["windows"]:
        if not window["seat_games"] or not window["policies"]:
            raise ValueError("chart window lacks sample size")
        if set(window.get("samples", {})) != required_metrics:
            raise ValueError("metric denominator missing")
        for metric, samples in window["samples"].items():
            current = window[metric]
            expected_keys = (
                set(current)
                if isinstance(current, dict)
                else {"value"}
                if metric == "scored_twice"
                else set(samples)
            )
            if not samples or set(samples) != expected_keys:
                raise ValueError("one or more plotted series lacks a denominator")
            for key, n in samples.items():
                value = current if key == "value" else (current or {}).get(key)
                if type(n) is not int or n < 0 or (value is None) != (n == 0):
                    raise ValueError("sample coverage inconsistent with plotted value")
    seen = set()
    for fact in home["facts"]:
        claim = fact["claim"]
        expected = ledger["facts"].get(claim)
        if claim in seen or expected is None or fact != {"claim": claim, **expected}:
            raise ValueError("homepage fact disagrees with approved " + claim)
        seen.add(claim)
    if seen != set(ledger["facts"]):
        raise ValueError("homepage fact coverage differs from the approved ledger")
    # User-facing Markdown titles feed Hero, Section and Roadmap components.
    # Numerical titles must be exact approved headlines, never freehand claims.
    for p in (ROOT / "site/src/content").rglob("*.md"):
        data = frontmatter(p)
        titles = [
            data.get("title"),
            data.get("kicker"),
            *(row["title"] for row in data.get("milestones", [])),
        ]
        for title in (value for value in titles if value):
            if re.search(r"\d", title) and title not in ledger["headlines"].values():
                raise ValueError("unapproved Markdown headline: " + p.name)
    checkpoint = str(home["status"]["checkpoint"])
    if "update " + checkpoint + "." not in ledger["claims"]["C-05"]:
        raise ValueError("unapproved status checkpoint")
    for p in (ROOT / "site/src/pages").glob("*.astro"):
        for heading in re.findall(r"<h[12]\b[^>]*>(.*?)</h[12]>", p.read_text(), re.S):
            if re.search(r"\d", re.sub(r"\{[^{}]*\}", "", heading)) or re.search(
                r"\{(?!approvedHeadlines\.)[^}]*\}", heading
            ):
                raise ValueError(f"unreviewed numeric or dynamic headline in {p.name}")
    print(
        f"Validated {len(scores):,} seat scores, {len(record['windows'])} behaviour windows, {len(ledger['claims'])} dated claims."
    )


def refresh():
    ledger = approved()
    helper = ROOT / "gui/site-data"
    games = ROOT / "replays/games"
    has_data = games.is_dir() and any(games.iterdir())
    if has_data and not helper.is_file():
        raise ValueError("A local archive is present but its verified dashboard exporter is unavailable.")
    if has_data:
        with tempfile.TemporaryDirectory() as tmp:
            dashboard = Path(tmp) / "dashboard.json"
            subprocess.run([helper, dashboard], check=True)
            record = Path(tmp) / "record.json"
            subprocess.run(
                [
                    "python3",
                    ROOT / "site/scripts/export_training_record.py",
                    "--input",
                    dashboard,
                    "--out",
                    record,
                ],
                check=True,
            )
            values = json.loads(record.read_text())
            validate(values, ledger)
            write_json(RECORD, values)
    else:
        print("Using the reviewed aggregate distributed with this checkout.")
        validate(json.loads(RECORD.read_text()), ledger)
    write_json(CLAIMS, ledger)


def publish(confirm):
    refresh()
    paths = ["site/public/data", "site/src/data", "site/src/content/home.md"]
    subprocess.run(["git", "diff", "--stat", "HEAD", "--", *paths], cwd=ROOT, check=True)
    changed = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *paths], cwd=ROOT).returncode
    if changed not in (0, 1):
        raise RuntimeError("could not inspect public diff")
    if not changed:
        print("No progress changes to commit.")
        return
    branch = "progress/" + dt.date.today().isoformat()
    if confirm:
        # Refuse to sweep another work item's staged changes into this commit.
        subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT, check=True)
        if (
            subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
            != branch
        ):
            subprocess.run(["git", "switch", "-c", branch], cwd=ROOT, check=True)
        subprocess.run(["git", "add", "--", *paths], cwd=ROOT, check=True)
        subprocess.run(["git", "commit", "-m", "Refresh reviewed training charts"], cwd=ROOT, check=True)
    print(f"Review the diff, then git push -u origin {branch}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["data", "validate", "publish"])
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    if args.action == "data":
        refresh()
    elif args.action == "publish":
        publish(args.confirm)
    else:
        validate_current()
