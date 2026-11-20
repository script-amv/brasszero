# CPU training transport

Use the CPU environment and locked dependencies described in the training requirements. Package source with `bash training/scripts/cloud_bootstrap.sh --archive /tmp/brasszero-source.tar.gz`. The archive contains the committed public engine, training and baseline code.

Keep checkpoints, provider notebooks and datasets private. Pin the source revision and manifest of each transferred experiment. Run the package verify command before any import or resume. For a new operator-controlled campaign, set the trusted HTTPS origin using `BRASS_CONTROL_ENDPOINT` on the owner machine. Runner and owner capabilities have distinct privileges. Only the run-scoped runner capability belongs with a private job. Never publish a capability, provider receipt or raw run. Existing sealed campaigns keep their original manifests.
