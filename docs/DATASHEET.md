# BrassZero game records

The archive supports replay, strategy analysis, imitation learning and multi-agent evaluation. Each published JSON contains the independent engine identity, starting seed, four seat identities, ordered native moves, final VP and native ranks. Published identities are limited to content-addressed policy and search hashes, training or initial weight digests with a numbered update, and versioned native Random/Greedy hashes. Unresolved identities and operator-provided aliases are represented by null.

Recorders capture moves from self-play and evaluation. Only completed games with a stable machine-only lineup enter the publication snapshot. The build excludes QA, fixtures, validation, live human games and branches. It removes local provenance paths, display names, provider receipts and operating-system identifiers during export. Original private records stay intact. A version's datasheet reports every inclusion and exclusion count. The sample reflects historical training and evaluation choices. It is not a controlled estimate of playing strength.

The game files carry CC BY 4.0. Attribute the BrassZero project, snapshot date and version. Each version includes its SHA-256 manifest and licence. Future snapshots may add games or document corrections. Historical gaps and missing model identities stay visible.
