# BrassZero

BrassZero is an independent research project that teaches a program to play Brass: Birmingham through self-play. The four-player game combines hidden hands, shared markets and decisions that score much later. The project includes a verified Rust rules engine, public-information players, graph-network policies and tools for measuring them on matched deals.

**Research status, 27 November 2026.** The recommended main-line model is BC 788. The latest published main-line checkpoint is 4097; its playing strength has not been evaluated, and a higher update number does not mean a stronger model. Human playing strength remains unmeasured. [Model weights](https://huggingface.co/brasszero/brasszero-ppo) and [game records](https://huggingface.co/datasets/brasszero/brasszero-games) are on Hugging Face.

| Result | Population | Reproduce or inspect |
| --- | --- | --- |
| Highest recorded AI score, 124 VP (approved C-09) | One evaluation game, experiment branch against three AI opponents | `make site-data`, inspect `record` in `site/public/data/training-record.json` |
| Mean score rose from 7.7 to 63.5 VP (approved C-16) | Updates 0–19, 2,570 seat scores, mostly the untrained model. Updates 820–839, 185 seat scores. There are 13,414 seat scores in the snapshot. This is a training trend. | `make site-data`, inspect `score_mean` and each window's sample count |
| Pass fell from 29.9% to 1.1% of main actions (approved C-17) | Updates 0–39 and 800–839 | `make site-data`, inspect `windows[].actions.Pass` and the action denominators |

## Quickstart

Install Rust with rustup, Python 3.9 or later, uv and Node 22.23.4 or later. Start at the repository root.

```sh
make setup
make test lint check-public
make site-dev
```

For frozen-model inference, install the optional CPU runtime and download a published checkpoint. The [model page](https://huggingface.co/brasszero/brasszero-ppo) gives the commands, and [Usage](docs/USAGE.md) covers evaluation.

## Repository map

- `engine/` contains the rules, native bindings, replay archive and differential adapters.
- `players/` contains baselines, frozen search and language-model player interfaces.
- `training/` contains optional PPO, evaluation and artifact tools.
- `site/` contains the static research site and compact measured aggregates.
- `docs/` contains [results](docs/RESULTS.md), [training history](docs/TRAINING-HISTORY.md), the [dataset description](docs/DATASHEET.md) and [maintenance commands](docs/MAINTAINING.md).

Selected model checkpoints are published on [Hugging Face](https://huggingface.co/brasszero/brasszero-ppo) under Apache-2.0, with a model card and checksums. A snapshot of machine-only games from 27 November 2026 is published as a [Hugging Face dataset](https://huggingface.co/datasets/brasszero/brasszero-games) under CC BY 4.0. Version 0.1.0 on the [GitHub releases page](https://github.com/script-amv/brasszero/releases) carries the BC 788 archive. The same snapshot is archived on Zenodo with a citable DOI: [10.5281/zenodo.24578707](https://doi.org/10.5281/zenodo.24578707) names this snapshot, and [10.5281/zenodo.24578706](https://doi.org/10.5281/zenodo.24578706) always resolves to the latest dataset version. Raw training runs stay private. Cite the software using [CITATION.cff](CITATION.cff) and the game records by their DOI.

Developed by the author with AI coding assistants (Claude Code, Codex).

BrassZero is an independent research project. Brass: Birmingham is a trademark of Roxley Games. This project is not affiliated with, sponsored by or endorsed by Roxley Games.
