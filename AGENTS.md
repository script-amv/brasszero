# Engineering conventions

Read the usage, results and maintenance guides before changing a public interface. Run `make test lint check-public`. Use `make test-training` when modifying optional training code.

Only `GameState::apply` mutates a game. A player receives its own observation and sanitized public events. Hidden state or recoverable deal seeds must stay outside player code. Report sole first place after native tiebreakers and disclose the number of independent deals.

Treat model weights, update envelopes and replay records as immutable. Verify checksums at transport boundaries. Changing a source-identity input creates a new compatibility identity. Keep exact source bytes for preserved formats unless you implement and test a versioned migration. Keep private reference engines, GUI, third-party game material and local writer state outside public history. No automatic publishing or training continuation.

## Where things go

| Kind | Place |
| --- | --- |
| Engine rules and native bindings | `engine/` |
| Bots, search and LLM players | `players/` |
| Training code, configs and tests | `training/` |
| Publication and boundary tools | `scripts/` |
| Website | `site/` |
| Guides for users and maintainers | `docs/` (update an existing guide before adding a new one) |
| Model cards and datasheets | `docs/model-cards/`, `docs/datasheets/` |
| Task plans, design specs, session notes | `lab/notes/plans/` (private, never public) |
| Experiment outputs, reports and logs | `lab/research/<name>-<yyyymmdd>/` (private) |
| Temporary files | `lab/scratch/` (private, delete when done) |

Planning tools that default to `docs/superpowers/` must save to `lab/notes/plans/` instead; the public check rejects `docs/superpowers/` and `docs/plans/`. Do not add a top-level directory or a new `docs/` file without a reason that a later reader needs. Every new public file goes into `.public-files`.

## Branches and commits

`main` is protected and public. Work on a branch (`feat/`, `fix/`, `docs/`, `exp/`), in a worktree under `.worktrees/` for long-running work. Open a pull request; it is squash-merged once CI passes, so the pull request title becomes the one commit on `main`: write it as a short imperative sentence. Never push `exp/` branches or experiment results; merge only the code an experiment proves useful. Never force-push `main` or rewrite published history. Never push tags; the owner cuts releases.
