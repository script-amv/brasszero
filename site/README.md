# BrassZero research site

The Astro site publishes a project overview and a measured training record. Markdown in `src/content/` supplies page copy. An approved claim ID accompanies each homepage fact. Numeric headlines come from the dated, reviewed ledger projection in `src/data/claims.json`. Each chart uses the compact download at `public/data/training-record.json`. The replay page (`/replay`) steps through one recorded game: `scripts/export_replay.py` replays its moves through the native engine and writes the public position after every move to `public/data/replays/`, without the deal seed or any hands. Town positions on the replay board come from real Midlands geography (`src/replay/geography.ts`).

From the repository root, run `make site-dev` to regenerate available local aggregates and start the preview. A public checkout uses its committed aggregates. To validate and build the static site, run `make test-site`. Per-metric sample counts appear in the chart tooltips and downloaded windows. Public-domain art credits are in `src/assets/art/CREDITS`. Font notices are included in `public/licenses/`.

The default build includes both robots meta and HTTP noindex settings. Set the same `PUBLIC_LAUNCHED` flag to true to remove both when approved. The maintenance guide documents Pages previews and production deployment.
