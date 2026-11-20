# Working with BrassZero

Start from the repository root. `make setup` builds the native Python extension and installs development tools. `make test` exercises the engine, Python interfaces and static site. Rust players can be tested independently with `make test-rust`.

To play a terminal game, use `PYTHONPATH=engine/tools .venv/bin/python -m brass_ai play --help`. Use `PYTHONPATH=engine/tools .venv/bin/python -m brass_ai match --agents random greedy greedy greedy --deals 1 --seed 123` for seeded tournaments.

For inference and optional training, install the separate runtime with `make setup-training`. On Intel macOS this uses Python 3.9 and CPU Torch. Linux x86-64 uses Python 3.10 and the CPU wheel. In a shell set `export PYTHONPATH="$PWD/engine/tools:$PWD/training:$PWD/players/search"`. The `training/.venv/bin/python -m brass_rl runtime` command reports the active engine and runtime. Download the reviewed archive and its adjacent checksum into `dist/models/`. Verify and unpack the release there, then run the downloaded policy against the native Greedy player. The release includes a card and license alongside its standalone `policy/` directory.

```sh
export PYTHONPATH="$PWD/engine/tools:$PWD/training:$PWD/players/search"
model_name=ppo-788-BC-3466c7765c2d
(cd dist/models && shasum -a 256 -c "$model_name.tar.gz.sha256" && tar -xzf "$model_name.tar.gz")
recordings=$(mktemp -d)
BRASS_REPLAY_ROOT="$recordings" training/.venv/bin/python -m brass_rl arena --agent "model=dist/models/$model_name/policy" --mode argmax -- match --agents model greedy greedy greedy --deals 1 --seed 123 --out "$recordings/match.jsonl"
```

The command registers the exact export and rotates the model through four seats on one seeded deal. Use `python -m brass_rl eval --help` for an immutable evaluation suite. Use `pack`, `verify` and `import` for experiment transport. Restoring optimizer state requires a trusted complete checkpoint and explicit training authorization. The maintenance guide covers site preview, publishing, datasets and releases.
