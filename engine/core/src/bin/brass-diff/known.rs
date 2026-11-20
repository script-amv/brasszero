//! Known, triaged differences between our engine and an oracle, from
//! `oracles/known_differences.toml` (section `[ts]` for the TS oracle, `[yuiffy]` for Yuiffy).
//!
//! Each entry names a rule the harness applies (in `ts_limits.rs` / `yuiffy_limits.rs`) to
//! step around a difference that is already decided, and must cite the RQ that decided it. An entry the
//! harness does not implement is an error, so the file cannot list differences that are not
//! enforced; an implemented rule only applies while its entry is present. Every application
//! is counted and reported, and entries never applied in a run are reported as unused.

use std::collections::BTreeMap;

pub struct Known {
    entries: BTreeMap<String, String>,
    fired: BTreeMap<String, u64>,
    /// Differences handled structurally (by the adapter, the protocol or a recorded vendor
    /// patch), not by a harness rule.
    structural: &'static [&'static str],
}

/// True if `s` contains `RQ-` followed by three digits.
fn cites_rq(s: &str) -> bool {
    s.as_bytes()
        .windows(6)
        .any(|w| w.starts_with(b"RQ-") && w[3..].iter().all(u8::is_ascii_digit))
}

impl Known {
    /// Load section `[section]` of the TOML file at `path`, accepting only the names in
    /// `implemented` and `structural`.
    pub fn load(
        path: &std::path::Path,
        section: &str,
        implemented: &[&str],
        structural: &'static [&'static str],
    ) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        let table: toml::Table =
            toml::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut entries = BTreeMap::new();
        if let Some(sec) = table.get(section) {
            let sec = sec
                .as_table()
                .ok_or(format!("[{section}] must be a table"))?;
            for (k, v) in sec {
                let why = v
                    .as_str()
                    .ok_or(format!("[{section}] {k}: value must be a string"))?;
                if !cites_rq(why) {
                    return Err(format!("[{section}] {k}: must cite an RQ-NNN: {why:?}"));
                }
                if !implemented.contains(&k.as_str()) && !structural.contains(&k.as_str()) {
                    return Err(format!(
                        "[{section}] {k}: no such harness rule (known: {implemented:?}, \
                         structural: {structural:?})"
                    ));
                }
                entries.insert(k.clone(), why.to_string());
            }
        }
        Ok(Known {
            entries,
            fired: BTreeMap::new(),
            structural,
        })
    }

    /// Whether the difference `name` is listed (so its harness rule applies).
    pub fn active(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    /// Count one application of rule `name`.
    pub fn fire(&mut self, name: &str) {
        debug_assert!(self.active(name), "rule {name} fired while not listed");
        *self.fired.entry(name.to_string()).or_default() += 1;
    }

    /// Lines describing how often each listed difference applied.
    pub fn report(&self) -> Vec<String> {
        self.entries
            .iter()
            .map(|(k, why)| {
                if self.structural.contains(&k.as_str()) {
                    format!("  {k}: structural ({why})")
                } else {
                    match self.fired.get(k) {
                        Some(n) => format!("  {k}: applied {n} times ({why})"),
                        None => format!("  {k}: UNUSED in this run ({why})"),
                    }
                }
            })
            .collect()
    }
}
