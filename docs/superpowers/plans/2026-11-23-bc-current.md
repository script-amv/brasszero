# Supported BC learner integration implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate the approved eight-game independent-critic PPO, loss-priority league and KL0.05 into main and start its authorized bounded free Kaggle continuation.

**Architecture:** Reuse the reviewed stability runtime and the researched loss-priority formulas. Expose an explicit supported BC profile/activation/resume path through existing training/checkpoint/transport infrastructure while preserving old formats and numerical paths.

**Tech Stack:** Existing Python3.9+ CPU Torch, native Rust engine, existing Kaggle CLI/control worker; no new service.

**Spec:** docs/superpowers/specs/2026-11-23-bc-current-design.md

## Global Constraints

- Eight games/update; distinct-game gradient groups up to four; target KL0.05; four independent critic epochs; BC rewards and GAE unchanged.
- Loss-priority only, old50/35/15 and eight snapshots, weighted draws with replacement; p=(2+W+T/2)/(4+N), weight1+4(1-p)^2, frozen mass normalized to pool size, baselines1each.
- Preserve independent source critic/Adam, actor/Adam, RNG and counters. No rewrite of immutable source/config under existing runs; old ordinary and stability study resume remains valid.
- No engine/information-boundary/hash-bug changes; no GUI restart; no push, paid compute, new model recommendation or extra scientific training.
- One eight-game local integration QA; one subsequent free private Kaggle trainer up to10800seconds with separate providerheadroom14100seconds (40mincheckpoint/export+15minsetup), source selected update4071.

## Review Focus

- Supported checkpoint passed to generic/cloud train resumes independent critic rather than rejecting/downgrading it; target config recorded in both activation and state.
- Legacy config missing newly introduced optional state loads unchanged; old study control/candidate remains numerically compatible with its declared protocol.
- Fork from independent source preserves nontrivial critic Adam and exact RNG; fresh initialization is allowed only for legacy shared sources.
- Loss counts use native rank, preserve half-ties and duplicate seats, advance exactly once across interrupted commit/resume, do not change RNG domains.
- Cloud prepare/train/recovery/export/import carries independent critic and league statistics, respects sparse full-state retention and protects manual graceful stop.

---

### Task 1: Supported BC runtime, loss-priority, CLI and cloud compatibility

**Files:** Modify training/brass_rl/{config,train,artifacts,main_bc,league,__main__,model_inventory}.py as required; reuse stability.py/stability_checkpoint.py; add focused profile/activation and loss-priority modules as the smallest needed units; add supported BC config, docs/USAGE.md, focused training/tests and .public-files entries. Modify training/scripts/kaggle_control only where required for supported state; preserve old jobs.

**Interfaces:** Consumes existing load_update/save_update and update_candidate. Produces a documented supported profile, explicit full-state activation/fork API and normal train/resume/CLI routes that work for both legacy and independent sources. Implementer settles exact helper/flag names and records them in report before handoff. Public portable actor format unchanged.

- [ ] Write focused failing tests for profile settings and both source types, preserved nontrivial independent optimizer/RNG state, legacy compatibility, native rank loss/tie/duplicate/mixed weights and idempotent cursor, training dispatch, whole-update stop/resume and transport/cloud dispatch.
- [ ] Verify intended failures with existing canonical numerical environment under shared heavy lock.
- [ ] Implement minimum supported runtime/config/activation and port only the accepted league behavior, retaining original seed streams. Make source/destination initialization explicit and production default discoverable; never silently continue the old BC788 run.
- [ ] Run targeted meaningful old/new regression suites and lint/public checks. Update usage/maintenance docs with exact local/cloud commands and known KL limits. Commit; self-review; deliver report and settled API/flags.
- [ ] Controller creates exact base..head diff and obtains independent task review; resolve findings before execution.

### Task 2: Integration proof, local main and authorized Kaggle launch

**Files:** private lab/research/bc-current-20261123 packet; docs as needed for usage; existing operator/package scripts. Root coordinates this task using Task1 APIs.

**Interfaces:** Consumes Task1 documented activation/resume route and existing Cloudflare stop-control/Kaggle tooling. Produces verified integration receipts and one live private three-hour Kaggle session; no scientific comparison.

- [ ] Verify original source hashes and canonical main/GUI baseline; record immutable source4071 and startup behavior. Run one isolated full eight-game QA update, request stop during it, resume/load at zero extra games; verify actor/critic/Adam/RNG/league and all eight native replays, old GUI actor loader where needed.
- [ ] Complete required `make test lint check-public` on final source, one numerical process. No redundant unchanged suites.
- [ ] Broad final review covers Task1 plus actual operator/transport proofs; preserve evidence. Apply at most the scoped necessary fix wave with covering checks.
- [ ] Merge into local main after verifying current base is unchanged and safe; preserve old worktrees/artifacts/GUI. Verify merged HEAD exactly matches tested code and smoke/status checks; no push.
- [ ] Prepare private pinned source/full-state input, checksum all transport boundaries, register scoped remote graceful-stop control without exposing secrets, check free Kaggle availability, submit one CPU job with a10800secondtrainerlimit and14100secondproviderdeadline includingcheckpoint/export/setupreserve. Recover/retry only if no training has been admitted, never duplicate a running trainer.
- [ ] Confirm actual source preflight and training admission/progress. Provide notebook link, exact settings/source, monitor/stop/retrieve commands, export locations and limitations. State whether output remains remote or has been downloaded. Never claim a run started from submission alone.
