# CPU training transport

Use the CPU environment and locked dependencies described in the training requirements. Package source with `bash training/scripts/cloud_bootstrap.sh --archive /tmp/brasszero-source.tar.gz`. The archive contains the committed public engine, training and baseline code.

Keep checkpoints, provider notebooks and datasets private. Pin the source revision and manifest of each transferred experiment. Run the package verify command before any import or resume. For a new operator-controlled campaign, set the trusted HTTPS origin using `BRASS_CONTROL_ENDPOINT` on the owner machine. Runner and owner capabilities have distinct privileges. Only the run-scoped runner capability belongs with a private job. Never publish a capability, provider receipt or raw run. Existing sealed campaigns keep their original manifests.

Future jobs whose sealed source includes the per-game logger print the same `GAME` lines as local training. The Kaggle controller forwards child standard output and runs Python unbuffered. Results therefore enter the notebook/job output as each update commits, one line per game. Kaggle determines when that output becomes visible to its client. `control status` still reports campaign phase and update; it is not a live per-game terminal stream.

The source revision is sealed when preparing a job. Commit the logging change before packaging a future job from the current repository. Historical frozen checkouts, including the private `lab/pinned-control` recovery source, retain their original output behavior. Existing uploaded or running jobs do not acquire local code changes. No remote session is submitted automatically by adding logging.
