# Contributing

Install Rust, uv and Node 22.23.4 or later. Run `make setup`, edit a focused change, then run `make test lint check-public`. Optional inference and search regressions run with `make setup-training test-training` on the supported CPU profiles. Work on a branch and open a pull request against `main`; it is squash-merged once CI passes. Include the commands and evidence in your pull request. [AGENTS.md](AGENTS.md) says where each kind of file belongs. Report rules disagreements with a minimal native position and independently written interpretation.

Submit only material you may distribute under its declared licence. Keep rulebook text, reference implementations, artwork, personal details and private run data out of commits. Read the [code of conduct](CODE_OF_CONDUCT.md).

Add each new public path to `.public-files` after reviewing its licensing, provenance and privacy. The public boundary check rejects any candidate or staged file outside that exact manifest.
