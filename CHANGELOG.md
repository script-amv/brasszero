# Changelog

Changes follow Keep a Changelog and semantic versioning.

## [Unreleased]

## [0.1.0] - 2026-11-27

First public release.

### Added

- Independent four-player rules engine, native Python bindings and replay archive.
- Random, Greedy, search and language-model player interfaces.
- Optional self-play tooling: PPO training, the `brass-bc/1` learner, evaluation suites and verified experiment transport.
- Static research site with measured aggregates and their sample sizes.
- Builders for model cards, portable weight archives and replay snapshots.
- BC 788, the recommended model, as a release archive with a model card and checksum (Apache-2.0).
- [Selected checkpoints](https://huggingface.co/brasszero/brasszero-ppo) and a [game record snapshot](https://huggingface.co/datasets/brasszero/brasszero-games) (CC BY 4.0) on Hugging Face.

### Known limitations

- Playing strength against human players is unmeasured.
- Checkpoints load only with the matching engine identity. Loading released checkpoints on Linux has not been tested.
- Two PyO3 advisories apply to the frozen dependency graph. [SECURITY.md](https://github.com/script-amv/brasszero/blob/main/SECURITY.md) records the review.
