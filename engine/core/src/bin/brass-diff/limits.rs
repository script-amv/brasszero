//! Oracle selection for the known-difference rules: each oracle has its own rule module
//! (`ts_limits.rs`, `yuiffy_limits.rs`) and its own section of `oracles/known_differences.toml`.

use std::collections::BTreeSet;

use brass_engine::moves::Move;
use brass_engine::record::{ActionRecord, DiffSnapshot};
use brass_engine::state::GameState;
use serde_json::Value;

use crate::known::Known;
use crate::{ts_limits, yuiffy_limits};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Limits {
    Ts,
    Yuiffy,
}

impl Limits {
    /// The `known_differences.toml` section.
    pub fn section(self) -> &'static str {
        match self {
            Limits::Ts => "ts",
            Limits::Yuiffy => "yuiffy",
        }
    }

    /// Structural differences the oracle's section may list (no harness rule).
    pub fn structural(self) -> &'static [&'static str] {
        match self {
            Limits::Ts => ts_limits::STRUCTURAL,
            Limits::Yuiffy => yuiffy_limits::STRUCTURAL,
        }
    }

    /// Harness rules the oracle's section may list.
    pub fn implemented(self) -> &'static [&'static str] {
        match self {
            Limits::Ts => ts_limits::IMPLEMENTED,
            Limits::Yuiffy => yuiffy_limits::IMPLEMENTED,
        }
    }

    /// Whether the snapshots differ only by a known divergence that ends the comparable game.
    pub fn known_divergence(
        self,
        known: &mut Known,
        ours: &DiffSnapshot,
        theirs: &DiffSnapshot,
    ) -> bool {
        match self {
            Limits::Ts => ts_limits::known_divergence(known, ours, theirs),
            Limits::Yuiffy => yuiffy_limits::known_divergence(known, ours, theirs),
        }
    }

    /// A request that repairs the oracle after an action whose snapshots differ exactly by a
    /// known difference it cannot avoid but that does not end the comparable game (the
    /// harness sends it and compares again); `before` is our state before `rec`.
    pub fn resync(
        self,
        known: &mut Known,
        before: &GameState,
        rec: &ActionRecord,
        ours: &DiffSnapshot,
        theirs: &DiffSnapshot,
    ) -> Option<Value> {
        match self {
            Limits::Ts => None,
            Limits::Yuiffy => yuiffy_limits::resync(known, before, rec, ours, theirs),
        }
    }

    /// The moves of `legal` the oracle can express (never empty when `legal` is not).
    pub fn candidates(
        self,
        known: &mut Known,
        g: &GameState,
        so_far: &[Move],
        legal: Vec<Move>,
    ) -> Vec<Move> {
        match self {
            Limits::Ts => ts_limits::candidates(known, g, so_far, legal),
            Limits::Yuiffy => yuiffy_limits::candidates(known, g, so_far, legal),
        }
    }

    /// Drop legal-summary keys that differ for a known reason.
    pub fn filter_keys(
        self,
        known: &mut Known,
        g: &GameState,
        ours: &mut BTreeSet<String>,
        theirs: &mut BTreeSet<String>,
    ) {
        match self {
            Limits::Ts => ts_limits::filter_keys(known, g, ours, theirs),
            Limits::Yuiffy => yuiffy_limits::filter_keys(known, g, ours, theirs),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every entry of each oracle's section names a rule its module implements (or a
    /// structural difference) and cites an RQ.
    #[test]
    fn known_differences_file_loads_for_every_oracle() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../oracles/known_differences.toml");
        for limits in [Limits::Ts, Limits::Yuiffy] {
            let known = Known::load(
                &path,
                limits.section(),
                limits.implemented(),
                limits.structural(),
            )
            .unwrap_or_else(|e| panic!("{e}"));
            for rule in limits.implemented() {
                assert!(
                    known.active(rule),
                    "[{}] does not list {rule}",
                    limits.section()
                );
            }
        }
    }
}
