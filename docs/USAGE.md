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

Training prints one flushed `GAME` line per newly committed training game. Each line identifies the experiment, stable game number, model update that played, saved update, all four players, VP, native places and the sole winner. For example:

```text
GAME run=exp-example game=3836 model_update=1918 saved_update=1919 | P1=AI:62VP/#1 P2=Frozen(training-001890):59VP/#2 P3=Greedy:29VP/#4 P4=Random:41VP/#3 | winner=P1
```

Seats and places start at 1. `AI` marks each trainable seat; self-play displays all four. Frozen opponents retain their snapshot IDs. Places include native tiebreakers. A tied first place is reported as `winner=none(tie:P1,P2)`. `model_update` is the version that made the decisions; `saved_update` is the checkpoint after learning from the batch. Lines are printed after the full checkpoint commits, so a two-game batch produces two lines together. Resume prints only newly committed games and keeps their stable IDs. A failed console write does not stop the training loop.

On Mac, lines appear in the training terminal. To retain them, append `2>&1 | tee -a training.log` to the existing training command in Bash or Zsh with `set -o pipefail`; another terminal can use `tail -f training.log`. The checkpoint manifests and replay archive remain the authoritative records if console output is interrupted.

### Continue BC training

Use `train-bc` for the supported `brass-bc/1` learner. Set `SOURCE` to a trusted, full numeric checkpoint directory and select a new run. For example, the first command activates from update 4071 without playing any games. The second advances to update 4072, collecting eight complete games. The absolute ceiling always counts completed updates.

```sh
export PYTHONPATH="$PWD/engine/tools:$PWD/training:$PWD/players/search"
run=training/runs/bc-current
training/.venv/bin/python -m brass_rl train-bc --source "$SOURCE" --out "$run" --max-updates 4071
training/.venv/bin/python -m brass_rl train-bc --resume latest --out "$run" --max-updates 4072
```

Activation preserves actor and optimizer, original seed streams, game and update counters, retained frozen membership, holdouts and immutable identity history. An independent source carries its critic weights and trained Adam moments into the child. A shared source starts the critic as an exact independent copy with fresh Adam. The new identity, parent digest, config changes, initialization checks and prior development selection are recorded in `experiment.json`. Earlier journals and recordings remain in the parent. The child begins its own journal and development selection at the activation boundary. A returned seed can also be resumed with `brass_rl train --config "$run/config.json" --out "$run" --resume latest --max-updates N`, replacing N with the absolute ceiling.

The profile uses BC own-VP rewards, relative four-player terminal critic targets and actor GAE with gamma 1 and lambda .95. The actor takes decision-weighted steps on fragments from at most four distinct games. It runs up to four epochs, checking a representative cross-game KL after each actor step and full chronological decision-weighted KL after each completed actor epoch. Both thresholds are .05. The representative estimate can differ from full KL, and a completed step can overshoot the threshold. The independent critic runs all four epochs with its own Adam and public-history memory.

The sampled lineup probabilities are 50% self-play, 35% frozen and 15% mixed. For each frozen identity, counts record twice the learner's wins plus ties and the total native-rank encounters. A duplicate frozen seat counts again. For a win score of one per win and half per tie, the smoothed probability is `p = (2 + score) / (4 + encounters)`. Frozen draw weights are `1 + 4*(1-p)^2`, normalized to sum to the active pool size. Mixed Random and Greedy each have weight one. Opponent seats draw with replacement. Counts and the game cursor commit once with the whole update. A source with no exposure starts empty. The eight-member retention and ten-update snapshot cadence continue.

See `training/configs/bc-current.json` for the standard profile. Activation inherits the source's model, seed and operational settings. Generic untagged training continues using shared PPO. To continue the historical BC788 main line explicitly, use `train-bc --legacy --max-updates N`. Its optional `--dynamic-league` flag belongs to that historical route. New actors published under `training/runs/bc-current` enter the main-line catalog after the source boundary. Existing recommended policies retain their selection.

### Local PPO stability experiment

For an explicitly authorized stability study, `brass_rl.stability_run.fork_stability(source, dest, candidate)` creates a compact run from a trusted numeric BC checkpoint and returns its new base checkpoint path. Pass `False` for the two-game shared-network control or `True` for the eight-game candidate. The parent manifest digest, configuration changes, initialization checks and runtime differences are recorded in `experiment.json`. Each new run retains the exact parent actor export, active league, counters, actor Adam and RNG state. Earlier history stays in the parent.

`train_stability(run, additional_games, stop_requested=callback)` resumes the latest complete update. Supply a whole number of two-game or eight-game updates. The optional callback requests a stop at an update boundary. The return value is the usual `TrainReport`. Ordinary training requires its own untagged checkpoints.

The candidate starts its critic as a separate copy of the source model with a fresh Adam. Its value estimates and recurrent boundaries come from replaying each viewer's public events from zero under the frozen critic. Its fixed actor advantages use own-seat GAE on those estimates. Each actor Adam step combines up to four recurrent fragments from distinct games, weighting the policy and entropy terms by eligible decisions. The critic independently trains for four epochs on the observer-relative terminal score vector, weighting fragments by all value decisions. Candidate `optimizer.pt` contains both Adam states and the critic weights at each resumable boundary. The portable actor export retains the legacy value head for format compatibility. Its values do not represent the independently trained critic.

Each update's `metrics.json` records `actor_steps`, `critic_steps`, decision visits, `kl_checks`, `monitor_policy_decisions` and `actor_stop_reason`. The control records its shared Adam steps in both step counts. The candidate checks `monitor_kl` against .03 after every actor step on a fixed fragment from every game, spreading the selection across temporal regions. This estimate uses the recorded actor boundary plus burn-in. Both arms report `final_kl` and `final_clip_fraction` on all eligible decisions using full public histories replayed under the final actor. The final metric can exceed the guard threshold. `critic_mse_before` and `critic_mse_after` measure four-component errors on the same training games.

### Local matched PPO stability study

The pre-registered overnight study uses a dedicated directory under the data project's `training/runs`. From the reviewed source checkout, use the local single-process Mac compute guard for `run` and `resume`. The command runs in the foreground. Set `PROJECT` to the data project that contains the original 4055 checkpoint, the sealed registration packet, replay archive, and playable-model catalog.

```sh
out="$PROJECT/training/runs/ppo-stability-20261122"
pre="$PROJECT/lab/research/ppo-stability-20261122/preregistration.json"
export BRASS_REPLAY_ROOT="$PROJECT/replays" PYTHONPATH="$PWD/engine/tools:$PWD/training"
training/.venv/bin/python -m brass_rl.stability_study run --project "$PROJECT" --out "$out" --preregistration "$pre"
python3 -m brass_rl.stability_study status --project "$PROJECT" --out "$out" --preregistration "$pre"
python3 -m brass_rl.stability_study stop --project "$PROJECT" --out "$out" --preregistration "$pre"
training/.venv/bin/python -m brass_rl.stability_study resume --project "$PROJECT" --out "$out" --preregistration "$pre"
```

`protocol.json` binds the source revision, embedded checkpoint configuration, deadline, opponents, and final deals. `progress.json` tracks the active arm and phase. Status reads numeric checkpoint boundaries, including work committed since the last progress write. Stop requests take effect after the current update or evaluation game. Explicit resume clears the stop request and continues from committed work. Use the same checkout revision on resume.

Control and candidate advance in matched blocks of eight games. Only after both reach 128 games does the runner seal its resource-based choice in `endpoint-decision.json`. An extension to 256 requires time for both arms plus a 90-minute evaluation reserve. Budget checks allow 4 GiB of new data, keep at least 2 GiB free, and reserve a further 64 MiB for atomic writes. Accounting includes the run directory, the private study packet and growth in the shared replay and model directories. Ordinary training publication registers new actors in the experimental catalog and retains their action recordings in the project archive.

At equal endpoints the final suite rotates each policy through all four seats on 16 pre-registered deals against three Greedy players and against fixed updates 688, 738 and 793. Evaluation rows under `evaluation/` link to native-verified recordings. `report.json` records exact coverage on a graceful stop. After a stop the runner reconciles archived games whose outcome rows were interrupted. The partial report lists recorded slots awaiting native verification separately and audits the actor exports for committed updates. A complete report includes optimizer work, full-collection KL and clipping, critic training error, and paired whole-deal intervals for all three policy comparisons. The two lineups share the same 16 independent deals and one training repetition.
