# Reporting a security issue

Use the repository's private security advisory form when enabled. Contact TODO-project-address when a private report is needed before the form is available. Include the affected version and reproduction details. Keep secrets, private game records and model artifacts out of public issues.

## Frozen native dependency review

PyO3 0.28.3 is part of the byte-identified dependency graph for the current policy and replay format. Two advisories apply to the dependency version. The Rust extension neither calls list/tuple iterators' `nth` or `nth_back` methods nor constructs a `PyCFunction` with `new_closure`. It registers static classes and a static `#[pyfunction]`. The supported inference runtimes use standard GIL-enabled CPython. The two exact exceptions in `deny.toml` are scoped by a CI check over the binding manifest, all its Rust sources and the entire locked graph. Any changed input requires reassessment. All other advisories remain enabled. A migration to patched PyO3 requires versioned compatibility testing with the preserved artifacts.

The review covers [RUSTSEC-2026-0176](https://rustsec.org/advisories/RUSTSEC-2026-0176.html) and [RUSTSEC-2026-0177](https://rustsec.org/advisories/RUSTSEC-2026-0177.html). Reassess at every binding, lockfile, native configuration or supported interpreter change, and before the next tagged release. PyO3 0.29.0 or later contains upstream fixes.
