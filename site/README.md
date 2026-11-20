# BrassZero research site

The Astro site publishes a project overview and a measured training record. Markdown in `src/content/` supplies page copy. An approved claim ID accompanies each homepage fact. Numeric headlines come from the dated, reviewed ledger projection in `src/data/claims.json`. Each chart uses the compact download at `public/data/training-record.json`.

From the repository root, run `make site-dev` to regenerate available local aggregates and start the preview. A public checkout uses its committed aggregates. To validate and build the static site, run `make test-site`. Per-metric sample counts appear in the chart tooltips and downloaded windows. Public-domain art credits are in `src/assets/art/CREDITS`. Font notices are included in `public/licenses/`.

The default build includes both robots meta and HTTP noindex settings. Set the same `PUBLIC_LAUNCHED` flag to true to remove both when approved. The maintenance guide documents Pages previews and production deployment.
