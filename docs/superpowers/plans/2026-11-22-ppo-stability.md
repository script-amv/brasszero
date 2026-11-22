# Overnight PPO Stability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and measure the approved combined PPO stability treatment in a bounded paired local study.

**Architecture:** Keep existing actor-only portable policy interfaces; add an opt-in experimental trainer with independent critic state and cross-game update diagnostics. A durable study runner makes compact forks, trains paired blocks, evaluates frozen endpoints, and writes auditable results.

**Tech Stack:** Existing Python 3.9, CPU PyTorch, native Rust engine, JSON checkpoint/replay infrastructure.

**Spec:** `docs/superpowers/specs/2026-11-22-ppo-stability-design.md`.

## Global Constraints

- BC score reward, terminal four-seat relative critic targets, gamma=1/lambda=.95, clipping, learning rate, entropy, game rules and information boundaries remain unchanged.
- No paid resources, GUI restart, original artifact rewrites, source main merge, model promotion, hash-bug fix, push, or extra scientific branches.
- Candidate eight games/update, four independent-game recurrent fragments per gradient step where available, common-sample KL with target .03, separate actor/critic parameters and recurrent states.
- Exact source 4055 and old league; same new training deals; equal endpoint game counts; primary128/arm, resource-only extension256/arm; frozen source/control/candidate evaluation384 games total.
- Eight-hour task wall budget, scoped caffeinate, one numerical process, 4GiB new-study ceiling, durable graceful stop/resume; no archive deletion.

## Review Focus

- Actor Adam moments and source policy identity survive the explicit fork even when critic architecture/training state changes.
- Independent critic recurrent boundaries must never use actor hidden states or public information from a different viewer/game.
- Sampled KL can understate full-batch drift; final full-batch diagnostic must remain visible and attributed to final weights.
- A stop during the larger eight-game collection leaves previous committed state resumable and does not double-count already archived games.
- Eight-game grouping changes update clocks; reports and evaluation comparisons use played game counts and retain exact model versions.

### Task 1: Opt-in stable PPO and independent critic

**Files:** Existing `training/brass_rl/{ppo,train,artifacts,config,rollout,model}.py` only as necessary; focused new experimental modules; targeted `training/tests/test_stability*.py`; usage documentation under docs/.

**Interfaces:** Preserve existing `train` and `update_policy` behavior by default and actor policy format. Provide explicit `fork_stability(source: Path, dest: Path, candidate: bool)` and a resume-capable `train_stability(run: Path, additional_games: int, *, stop_requested, ...)` (or document a comparably simple settled API for Task2). Candidate optimizer/checkpoint state atomically includes its critic. Strict ordinary resume must not silently reinterpret an experiment.

- [ ] Write and run failing tests proving weighted multi-game gradients, cross-game KL stopping, actor/critic isolation, independent recurrent state, rollback and exact resume.
- [ ] Implement the smallest opt-in modules and compatibility integration satisfying the spec. Legacy numerical path remains unchanged.
- [ ] Run targeted and existing PPO/artifact/train/config/rollout tests; record exact commands and failures resolved.
- [ ] Commit only Task1 source/tests/docs and report settled public interfaces, checkpoint format/provenance and test evidence.

### Task 2: Bounded matched study runner and reporting

**Files:** New `training/brass_rl/stability_study.py`, targeted controller/report tests, a small CLI launcher if needed; docs/ usage and immutable ignored run protocol/results.

**Interfaces:** Use Task1 explicit compact fork and training APIs; existing `combination_evaluation.evaluate`, `EvalSuite`, `Archive`, and native replay validation. CLI supports `run`, `status`, `stop`, and `resume` with stable study path. Atomic progress file records active arm/phase, deadlines, committed games and output paths.

- [ ] Write failing tests for paired eight-game blocks, budget-only endpoint extension, stale/missing state refusal, stop/resume and disk/time bounds.
- [ ] Implement runner, freeze source identity/deals/opponent IDs and verify source/arm independence before training.
- [ ] Reuse existing evaluation reporting where possible; add paired whole-deal uncertainty and explicit one-training-repeat limitation.
- [ ] Run controller tests, real bounded QA training/stop/resume and portable inference checks, keeping QA games separate.
- [ ] Commit source/tests/docs. Run full required verification. Obtain independent whole-branch review, resolve important findings before scientific training.

### Task 3: Execute and report

**Files:** Ignored canonical `training/runs/ppo-stability-20261122/`, private `lab/research/ppo-stability-20261122/` logs/report, final docs update if appropriate.

- [ ] Run under shared Mac numerical guard and scoped sleep prevention; keep all new replay and policy publications available without restarting GUI.
- [ ] Monitor actual speed, invariant checks, disk/time reserves and failures; preserve resumable checkpoints on every exit.
- [ ] Finish matched frozen evaluation within bound; native-verify recordings and report exact measured counts, coverage, source, speed, strength and uncertainties.
- [ ] Leave experiment isolated and provide concrete status/stop/resume commands. No automatic promotion or merge.
