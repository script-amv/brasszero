# CPU training transport

Use the CPU environment and locked dependencies described in the training requirements. Package source with `bash training/scripts/cloud_bootstrap.sh --archive /tmp/brasszero-source.tar.gz`. The archive contains the committed public engine, training and baseline code.

Keep checkpoints, provider notebooks and datasets private. Pin the source revision and manifest of each transferred experiment. Run the package verify command before any import or resume. For a new operator-controlled campaign, set the trusted HTTPS origin using `BRASS_CONTROL_ENDPOINT` on the owner machine. Runner and owner capabilities have distinct privileges. Only the run-scoped runner capability belongs with a private job. Never publish a capability, provider receipt or raw run. Existing sealed campaigns keep their original manifests.

Future jobs whose sealed source includes the per-game logger print the same `GAME` lines as local training. The Kaggle controller forwards child standard output and runs Python unbuffered. Results therefore enter the notebook/job output as each update commits, one line per game. Kaggle determines when that output becomes visible to its client. `control status` still reports campaign phase and update; it is not a live per-game terminal stream.

The source revision is sealed when preparing a job. Commit the logging change before packaging a future job from the current repository. Historical frozen checkouts, including the private `lab/pinned-control` recovery source, retain their original output behavior. Existing uploaded or running jobs do not acquire local code changes. No remote session is submitted automatically by adding logging.

## Supported BC continuation

Use a new private job and the reviewed source revision. Set `SOURCE` to the chosen numeric full checkpoint, `PRIVATE` to an existing private working directory, `KERNEL` to a fresh account/slug, and `BRASS_CONTROL_ENDPOINT` to the trusted control origin. The seed command forks the supported learner, verifies its base and seals `seed.tar.gz` plus `seed.tar.json`. It consumes zero games. The following example gives the child up to three hours of training and a separate provider window of three hours and 55 minutes.

```sh
export PYTHONPATH="$PWD/engine/tools:$PWD/training:$PWD/players/search"
py=training/.venv/bin/python
control=training/scripts/kaggle_control
"$py" "$control/prepare.py" seed-bc "$SOURCE" "$PRIVATE/seed.tar.gz" --max-wall-time-seconds 10800
deadline=$(( $(date +%s) + 14100 ))
"$py" "$control/prepare.py" prepare "$PRIVATE/seed.tar.gz" "$PRIVATE/seed.tar.json" "$PRIVATE/job" "$KERNEL" "$deadline" --stage resume
```

Register the prepared `job/registration.json` at the control service's run-scoped KV key `RUN/config` and upload only `job/dataset` to its named private Kaggle input dataset. Keep owner credentials local. The existing CPU-only submit and owner commands operate on this same job directory:

```sh
"$py" "$control/prepare.py" submit "$PRIVATE/job"
"$py" "$control/owner.py" "$PRIVATE/job" status
"$py" "$control/owner.py" "$PRIVATE/job" stop --wait-download
```

For a deadline stop, use `watch --wait-download` in place of the manual stop. After provider publication, `download` retrieves and verifies the pinned return. The child stores its 10,800-second training wall budget in its immutable base config. The trainer admits whole updates within that budget, accounting for checkpoint margin and observed update duration. The example provider window adds 15 minutes for setup, 30 minutes for export and 10 minutes for an update. Longer setup, queue delays or slow updates can shorten the training window. The control receipt reports a scheduled wall-budget cutoff as `deadline`. The capacity guard also accounts for free space and the size of the return archive. Stop commands allow an admitted update to finish.

`job/verified/verified.json` identifies the experiment, source manifest, base, tip, game cursor, actor and critic Adam summaries and replay count. Verification checks immutable prefix files, full latest optimizer, per-update playable exports, replay actions and native results. It reconstructs the eight sampled lineups from each previous checkpoint and checks the next exposure table against their native places. Further authorized continuation can use the returned `continuation.tar.gz` and `verified.json` as the two inputs to a fresh `prepare --stage resume` job. Historical 843 seed/proof procedures keep their original source pins and checks.
