# Working with BrassZero

Start from the repository root. `make setup` builds the native Python extension and installs development tools. `make test` exercises the engine, Python interfaces and static site. Rust players can be tested independently with `make test-rust`.

To play a terminal game, use `PYTHONPATH=engine/tools .venv/bin/python -m brass_ai play --help`. Use `python -m brass_ai match --agents model greedy greedy greedy --deals 1 --seed 123` with the same interpreter and import path for seeded tournaments.

For inference and optional training, install the separate runtime with `make setup-training`. On Intel macOS this uses Python 3.9 and CPU Torch. Linux x86-64 uses Python 3.10 and the CPU wheel. In a shell set `export PYTHONPATH="$PWD/engine/tools:$PWD/training:$PWD/players/search"`. The `training/.venv/bin/python -m brass_rl runtime` command reports the active engine and runtime. For a downloaded portable policy extracted under `dist/models/`, run:

```sh
training/.venv/bin/python -m brass_rl arena --agent model=dist/models/MODEL_DIRECTORY --mode argmax -- match --agents model greedy greedy greedy --deals 1 --seed 123
```

The command registers the exact export and rotates the model through four seats on one seeded deal. Use `python -m brass_rl eval --help` for an immutable evaluation suite. Use `pack`, `verify` and `import` for experiment transport. Restoring optimizer state requires a trusted complete checkpoint and explicit training authorization. The maintenance guide covers site preview, publishing, datasets and releases.
