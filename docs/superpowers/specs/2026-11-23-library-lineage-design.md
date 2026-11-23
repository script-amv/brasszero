# Automatic local library ingestion and visible training branches

## Intent and selected approach

The owner wants every available model and replay retained in the shared local
libraries after training or import, independently of whether the run is accepted
mainline or an experiment. GUI filtering changes visibility, never retention.
Publication here means local library registration only, never an external upload
or a public release.
The owner selected extending the existing shared-library approach, with
`mainline`/`experimental` tags and a Mainline/All GUI switch.

Alternatives considered: leaving Kaggle download and local publication as two
manual operations retains the current missing-data problem; a separate library
or service duplicates working storage and is unnecessary. Extend the existing
trainer, verified importer, catalog and dashboard. No new service or database.

## Current state

- Normal `publish_policies` writes portable models and updates the shared model
  inventory; training records games through the existing replay archive.
- Generic `import_package` verifies and moves models, replays and run journals,
  but does not rebuild their inventory labels through `update_inventory`.
- The pinned Kaggle owner downloads a different continuation envelope, verifies
  it and leaves it in `job/verified/`; it does not ingest the canonical project.
- Existing catalog entries carry lineage/branch, but mainline determination
  partly depends on run location. This is insufficient across cloud imports.
- The dashboard reads all catalog sections and the shared replay archive. Its
  smooth curves currently combine different model histories into one trend.

## Authoritative lineage and identity

Keep portable models, experiment/checkpoint bytes and recorded actions immutable.
Store local classification in versioned, atomically written registry metadata
beside the shared catalog; exact filename and small schema are builder choices.
Identify a run by experiment ID, a model by exact policy ID, and parent links by
the recorded parent policy/checkpoint digest. Update numbers alone are never IDs.

The registry explicitly identifies the accepted continuation chain and branch
membership; paths and maximum update number are not authoritative. Existing
mainline catalog membership seeds migration. The accepted chain now includes the
stability candidate4056–4071 and the new supported Kaggle continuation4072–4097;
the competing stability control remains experimental through4119. Follow exact
recorded parents for ancestry; do not promote sibling experiments. Preserve all
other historical branches and previous best/recommended selections.

Future launches register an explicit lineage intent in the portable ingestion
metadata. Generic forks default experimental, including forks from a mainline
tip. The supported mainline continuation command defaults to mainline intent
only from the declared tip with the accepted learner profile; its experimental
override remains available. Imports preserve the declared intent; missing intent
defaults experimental until an explicit local adoption. Registry updates check
the expected predecessor: a competing/divergent child cannot silently replace
the mainline tip. Exact flag and metadata names are builder choices. These are
controller proposals for an unambiguous producer/import contract, not extra
learning algorithm changes. Explicit adoption changes local registry/catalog
metadata, not old artifacts.
Duplicate policies can have multiple run occurrences without duplicated weights;
membership/parent edges must retain those occurrences rather than losing them
to first-publication metadata. Unknown provenance remains visible in All, with
an unknown-history label rather than an invented parent.

## One ingestion contract

All supported training publication and import/download entrypoints finish by
idempotently reconciling the shared `players/models/` and `replays/` libraries,
catalog, lineage metadata and replay index from verified available artifacts.
Preserve the generic importer's conflict checks, native replay validation,
immutable units, locking and recoverable publication; no global transaction is
claimed. Retrying after interruption repairs derived metadata and creates no
duplicate models, games or branch edges. A failed ingestion is reported explicitly
and retriable, never treated as successful complete import.

For Kaggle, future owner download/watch-download paths perform verification then
canonical ingestion automatically. Do not change a sealed running job or hash
pinned operator files. Provide the smallest supported host-side integration so
this already downloaded packet can be ingested through the same verified
contract, and future prepared operators include that integration by default.
No export is considered trusted merely because it is present on disk.

On this task, reconcile existing registered runs and the verified208-game Kaggle
return, adding all available missing models/replays and reporting counts/gaps.
Use valid full old recordings only; do not regenerate missing historical games.
Keep originals and transfer archives intact. QA artifacts may be stored but keep
existing statistical exclusions. Outside an explicitly isolated QA archive, a
supported launch must not silently skip shared-library publication.

## GUI behavior

Add a **Mainline / All** scope control, default Mainline (implementer proposal).
Apply it consistently to dashboard models, observations, record cards, selectors
and totals. Existing category/opponent/cohort filters remain composable. The
opponent selector keeps its existing explicit lineage labels and exact versions;
no new filter there is required. Existing sessions/selections remain unchanged.
The replay library retains all
recordings. An experimental opponent in
a mainline game's seat does not remove the mainline player's observation; use
model occurrence membership, not an all-seats-mainline requirement.

Show explicit branch names and identity suffixes, so control4097 and current4097
are distinct. “Latest mainline” means declared chain tip, not numerical maximum;
it is separate from “recommended/best”. A current tip with no recorded decisions
still appears in inventory/selection with no fabricated metric. Training batch N
contains decisions by its pre-update actor, normally N−1; preserve exact binding.

## Graph behavior

Keep X=model update, present lightweight SVG layout, no horizontal scrolling,
subtle exact observations and smooth trend curves. In All, VP gets separate
named/color-coded trend series for mainline and each experimental branch. Never
smooth across sibling branches or collapse equal-update distinct policies.
Add a parent/fork connector only with a proven policy parent and matching visible
metric under current filters. A smoothed value or a missing parent observation
is not a measured fork anchor. Explain that these are recorded-play trends,
not controlled strength comparisons. Limit smoothing support to available branch
observations; an isolated point stays a point.

Apply separate branch series to other simple line metrics where legible. For
stacked action percentages/VP composition, retain one meaningful branch at a time
with a small branch selector when All is active; do not stack or average unrelated
branches together. Keep exact replay inspection and existing metric definitions.

## Constraints and non-goals

No new training or evaluation, paid resources, push, game/PPO/reward/league changes,
Linux/Mac export-hash fixes, weight averaging, deletion or automatic best-model
promotion. Do not restart active GUI sessions without saving and preserving them.
Prefer hot refresh of catalogs; if server changes need restart, preserve/recover
all active sessions through existing session mechanisms first.

## Validation and delivery

Targeted tests cover local publication, generic import, Kaggle return ingestion,
repeat ingestion, interrupted reconciliation, source hashes, provenance conflicts,
mainline migration, sibling exclusion and duplicate policy occurrences. Verify
the last actor4097 is selectable although its own decisions may be absent.
Backend/frontend tests cover composed scopes, model-less games, zero-game models,
same-update separate branches, absent/visible parent anchors, per-branch smoothing,
and nonmixed stacked charts. Test actual GUI at desktop/narrow width, exact replay
opening, scope controls and session preservation. Run appropriate repository
checks, including `make test lint check-public` and training tests for changed
training code. Update `docs/`; report imported/deduplicated/gap counts, exact
mainline tip, verification and remaining limitations. No push.
