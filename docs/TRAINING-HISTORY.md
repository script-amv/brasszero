# Training history

The dated public snapshot retains the main line and experiment branches. Earlier score windows include evaluations of the untrained policy. Later windows mix self-play and frozen evaluation opponents. Each score window gives the observed seat count. Each behaviour window gives seat and policy counts together with per-metric denominators. Missing update positions remain unknown and are omitted from the numerical axis.

See [the results](RESULTS.md) for the dated model recommendation and measured opponents. Rebuild the local chart with `make site-data` and review it with `make site-dev`. Move a headline only after checking its evidence and approving the corresponding ledger row.
