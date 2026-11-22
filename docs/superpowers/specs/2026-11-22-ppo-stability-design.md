# Overnight PPO stability experiment

The owner authorized implementation and overnight verification of the previously discussed combined treatment: eight independent complete games per update, gradients accumulated over four recurrent fragments from different games where available, fixed representative cross-game KL monitoring, and independently trainable actor and critic. This authorization replaces another design/permission round. Do not change production defaults or promote a model automatically.

## Scope and constraints

Work in this isolated source worktree; keep the running GUI and canonical main source untouched. Use only local Mac CPU, a single numerical process, the existing numerical guard, and scoped sleep prevention. Do not rent/restart paid compute or launch cloud jobs. Preserve original exports, checkpoints, recordings, and the Linux/Mac export-hash issue. All new runs are experimental, not main-history league members. BC score reward, terminal four-seat relative critic targets, own-seat GAE gamma=1/lambda=.95, PPO clipping, learning rate, entropy, legal actions, and information boundaries are unchanged.

## Treatment

Control retains the original two-game shared-network PPO. Candidate uses eight games per update, with four whole recurrent fragments accumulated per actor Adam step, preferring distinct game IDs; tail groups may be smaller. Weight policy/entropy by eligible decisions and value loss by value decisions, rather than giving short fragments extra influence. Never backpropagate a graph from one game through another.

Candidate KL protection uses a deterministic fixed representative subset of collected sequences, covering every collected game and varied temporal regions. Check it after actor steps using the rollout old log probabilities; also measure decision-weighted KL and clipping over the full collection at final actor weights. Keep target KL .03 unchanged. Expose measurement denominators, check frequency, actual work, and stop reason. A lower fraction of KL stops is not itself a success criterion; do not disable protection or increase its threshold to obtain it.

Critic has separate parameters, optimizer, and public-event recurrent state. Initialize its function as an exact clone of the source shared network. No actor parameter may change in a critic-only update; no critic parameter may change in an actor-only update. Critic learns the existing final relative BC payoff vectors for four epochs independently of actor early stopping. GAE uses critic values recorded under the frozen collection-time critic; actor advantages stay fixed throughout each update. Critic never receives a hidden state or deal seed. Avoid unnecessary legal-action scoring in critic inference if practical and verified.

Keep portable actor policies in the current format so existing GUI workers can load them without critic/training state. Resumable experiment boundaries atomically include independent critic weights, optimizer, and relevant RNG state. Preserve strict ordinary resume configuration checks and support old checkpoints without rewriting them. An explicit experimental fork records changed settings, source manifest, preserved actor/Adam/RNG/league/counters and critic initialization. Interrupted collection/PPO repeats only uncommitted work; save/play every policy and keep full resume state at each tenth update plus latest/source as before.

## Study chosen by the implementer

Start both arms from latest full checkpoint `training/runs/vast-legacy1919-20261122/updates/004055`, not a silent rollback. Its manifest SHA256 is `0edc774d71641b2c22c362e865c0801aff98cc2579fd3a3d01d9752c12adae1e`. Compact forks reference immutable ancestry and avoid copying the entire old run. Preserve the source old league and per-update snapshot cadence; the different games/update necessarily changes snapshot timing per game and is disclosed as part of this combined treatment.

Train in paired eight-game blocks: control four updates, candidate one update. Primary endpoint is 128 new games per arm. A predeclared resource-only extension to 256 per arm is permitted only if measured speed leaves enough time for both arms and evaluation. No choice based on observed strength. Never compare endpoints with unequal new game counts. Freeze protocol and evaluation deals before scientific training. Time limit is eight hours from task start, with a conservative evaluation/checkpoint reserve and a 4 GiB new-study disk ceiling; reaching a bound stops gracefully and preserves all data without silent deletion. Use separate run IDs/directories and durable progress, stop and resume commands.

After training, compare frozen source and both equal-game endpoints against (a) three Greedy and (b) fixed AI 688/738/793. Sixteen fresh deals, every candidate seat: 64 games per model per lineup, 384 total, paired exact deals and physical-seat sampling seeds. Select fresh deals disjoint from planned training and available archived seeds. If the eight-hour budget prevents completing the suite, retain partial data and report precise coverage without pretending it is final. Evaluate only after completing both arms at the selected endpoint. Save all action replays and native-verify outcomes.

Report mean/median/Q25/max VP, sole native-ranked wins, average place, deal-block paired differences with uncertainty versus control and source, actor/critic steps, final global KL/clipping, stop frequencies, throughput, time, and disk. Distinguish own-rollout critic fitting from held-out predictive ability. One combined treatment and one training repetition cannot identify which component caused an effect or prove general improvement. No automatic promotion, further branches, or weight averaging.

## Alternatives considered

Continuing only the existing algorithm would not test the authorized changes. Separate component ablations would improve causal attribution but multiply the overnight budget. A combined controlled test matches the owner's request; record the attribution limitation and leave component ablations for a later decision.

## Verification

Use failing-then-passing targeted tests for distinct-game grouping/decision weights, common-sample KL detection and stopping, actor/critic parameter isolation, critic continuation after actor stop, four-seat target ordering, old checkpoint loading, full experiment save/resume equality, interrupted-update rollback including all optimizers/RNG, compact independent forks, paired deal schedule, and runner stop/disk/time limits. Run appropriate complete existing suites and `make test lint check-public` before claiming completion. Archive scientific source revision and leave main/GUI sessions intact. No push.

## Pre-training guard refinement

A disposable eight-game QA update finished four actor epochs with fixed-sample KL 0.02899 and full-collection KL 0.04022 against the registered threshold of 0.03. No scientific training or evaluation had begun. Protocol `brass-ppo-stability/2` keeps the fixed cross-game check after each actor step and also replays every eligible decision chronologically under current actor weights at each completed actor epoch. A full-epoch KL above 0.03 ends the actor update. The independent critic still completes all four epochs. Record per-step probe work, full-epoch check counts and decision visits, last full-epoch KL, stopping reason, and final full-collection KL and clipping. The optimizer tensor layout, registered deals, endpoint budget and control objective are unchanged.
