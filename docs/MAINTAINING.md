# Maintaining BrassZero

| When you want to… | Run | Result |
| --- | --- | --- |
| Prepare a checkout | `make setup` | Builds native bindings, installs Python and npm dependencies and the pre-commit hook. The private GUI is installed when present. |
| Check code | `make test lint check-public` | Runs the fresh-clone checks. The optional numerical suite uses `make setup-training test-training`. |
| Play locally | `make play` | Opens the private server on loopback port 8773 when present. |
| Preview the site | `make site-dev` | Refreshes aggregates, starts Astro on loopback port 4321 and polls data changes. Override with `SITE_PORT=4322`. |
| Update charts | `make publish-progress` | Checks scope, sample sizes, approved claims and dates, then shows the diff. Add `CONFIRM=1` to commit changed exports on a progress branch. It prints the push command. |
| Build a site preview or deployment | `make test-site` | Validates and builds static files. A PR builds a preview artifact. A same-repository PR can deploy to a preview branch. Main deploys production. Missing Pages secrets leave the artifact available. |
| Prepare a model | `make release-model MODEL=MODEL_ID` | Verifies the local catalog hashes, builds a card and archive, and prints the upload command for owner review. |
| Snapshot game records | `make dataset-snapshot` | Builds a derived machine-game archive, datasheet, licence and SHA-256 manifest under `dist/dataset/`. |
| Prepare a version | `make release VERSION=0.1.0` | Checks a clean tree, citation and changelog. Add `CONFIRM=1` to update citation metadata and create the annotated local tag. The tag workflow creates the release after an owner pushes it. |
| Check private workspace boundaries | `make sync-check` | Verifies ignored local repositories, game records, trained models and reference material. |

Repeated packaging keeps identical outputs. Replacing a different local model card or distribution archive requires `CONFIRM=1` after reviewing the changes.

Keep `PUBLIC_LAUNCHED=false` until launch. One repository variable controls both the robots meta tag and the HTTP noindex header. Set `SITE_URL` to the site's public origin when ready. Use Pages Access for restricted preview audiences. Model weights and repository code carry Apache-2.0. Published game records carry CC BY 4.0.

At each dependency, binding or interpreter change and before tagging a release, review the two retained PyO3 advisory exceptions in [SECURITY.md](../SECURITY.md). Run `python3 scripts/check_binding_audit.py` and `cargo deny check advisories licenses`.

For a model upload, set `RELEASE_TAG` to the reviewed tag before using the printed GitHub command. The archive contains a standalone `policy/` directory, model card and license. See [usage](USAGE.md) for the checksum and extracted-model evaluation commands. A production site build ships the browser dependencies’ full notices at `/licenses/client-dependencies.md`.

For supported BC, preserve the explicitly selected full source and its ancestors when moving an experiment. Compact activation reads the parent without changing its bytes or adding a retention pin there. Retain an independent full copy if the parent will continue training past an off-cadence source. The child keeps its activation boundary, latest boundary, ten-update cadence, pins and development selections as full checkpoints. Between these boundaries, rolling retention can retire the dual optimizer payload, which includes critic weights. Every actor export and committed controller journal remains available. The local activation and cloud transport commands are in [Usage](USAGE.md) and [Kaggle](KAGGLE.md).
