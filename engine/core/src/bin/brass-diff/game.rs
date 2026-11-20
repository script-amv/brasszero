//! The shared game loop: one random game played on our engine and the oracle side by side.

use std::collections::BTreeMap;

use brass_engine::data::{Era, LINKS, LOCATIONS, MERCHANT_SPACES, SLOTS};
use brass_engine::moves::{Move, Phase};
use brass_engine::record::{ActionRecord, InitSpec, action_complete, summary_key};
use brass_engine::state::{GameState, from_init};
use rand::{RngExt, SeedableRng};
use rand_pcg::Pcg64;
use serde_json::{Value, json};

use crate::compare::{Mismatch, compare_legal, compare_reshuffle, compare_snapshots, move_kind};
use crate::known::Known;
use crate::limits::Limits;
use crate::oracle::Oracle;

/// Our board indexing, sent with `init` so the adapter translates by name.
pub fn tables() -> Value {
    let loc = |l: u8| LOCATIONS[l as usize].id;
    let mut per_loc = vec![0usize; LOCATIONS.len()];
    let slots: Vec<Value> = SLOTS
        .iter()
        .map(|s| {
            let i = per_loc[s.loc as usize];
            per_loc[s.loc as usize] += 1;
            json!([loc(s.loc), i])
        })
        .collect();
    let links: Vec<Value> = LINKS
        .iter()
        .map(|l| json!([loc(l.a), loc(l.b), l.extra.map(loc)]))
        .collect();
    let merchant_slots: Vec<Value> = MERCHANT_SPACES
        .iter()
        .flat_map(|m| (0..m.tile_slots).map(move |t| json!([loc(m.loc), t])))
        .collect();
    json!({
        "locations": LOCATIONS.iter().map(|l| l.id).collect::<Vec<_>>(),
        "slots": slots,
        "links": links,
        "merchant_slots": merchant_slots,
    })
}

/// A random complete action for the current player, as the oracle can express it.
fn choose_action(
    g: &GameState,
    rng: &mut Pcg64,
    limits: Limits,
    known: &mut Known,
) -> (ActionRecord, GameState) {
    let mut sim = g.clone();
    let player = sim.current_player().expect("game not over");
    let mut moves: Vec<Move> = Vec::new();
    loop {
        let legal = limits.candidates(known, &sim, &moves, sim.legal_moves());
        assert!(!legal.is_empty(), "no legal move in a live state");
        let m = if sim.phase == Phase::ChooseAction && moves.is_empty() {
            // A uniform key, then a uniform move with that key: keeps rare actions (Loan,
            // Pass, Develop) from drowning in the many Build and Scout moves.
            let mut by_key: BTreeMap<String, Vec<&Move>> = BTreeMap::new();
            for m in &legal {
                by_key.entry(summary_key(&sim, m)).or_default().push(m);
            }
            let keys: Vec<&Vec<&Move>> = by_key.values().collect();
            let group = keys[rng.random_range(0..keys.len())];
            group[rng.random_range(0..group.len())].clone()
        } else {
            legal[rng.random_range(0..legal.len())].clone()
        };
        sim.apply(&m).expect("candidate move is legal");
        moves.push(m);
        if action_complete(&sim) {
            break;
        }
    }
    (ActionRecord { player, moves }, sim)
}

/// How a game ended.
pub enum Outcome {
    /// Played to the end with no mismatch.
    Complete,
    /// Stopped at a known divergence (see `Limits::known_divergence`).
    KnownDivergence,
}

/// The `sourcing` discrepancies of an `apply` response (`oracles/protocol.md`): an oracle
/// that declared `sourcing_check` at `init` must send the array (possibly empty); one that
/// did not must leave the field out (or `null`). Anything else is a protocol error.
fn sourcing_of(applied: &Value, sourcing_check: bool) -> Result<Vec<String>, String> {
    match (&applied["sourcing"], sourcing_check) {
        (Value::Null, false) => Ok(Vec::new()),
        (Value::Null, true) => Err("apply response has no `sourcing` array, but the oracle \
                                    declared sourcing_check"
            .into()),
        (v, true) => serde_json::from_value(v.clone())
            .map_err(|e| format!("apply response `sourcing` is not a string array: {e}")),
        (v, false) => Err(format!(
            "apply response has `sourcing` ({v}), but the oracle declared no sourcing_check"
        )),
    }
}

/// The first rule in the `relaxed` list of an `apply` response that is not a listed known
/// difference of this oracle; each listed one is counted as applied. A malformed list counts
/// as unlisted.
fn unlisted_relaxation(applied: &Value, limits: Limits, known: &mut Known) -> Option<String> {
    let list = match &applied["relaxed"] {
        Value::Null => return None,
        Value::Array(l) => l,
        v => return Some(format!("malformed({v})")),
    };
    for r in list {
        let Some(rule) = r["rule"].as_str() else {
            return Some(format!("malformed({r})"));
        };
        if !(known.active(rule) && limits.implemented().contains(&rule)) {
            return Some(rule.to_string());
        }
        known.fire(rule);
    }
    None
}

/// Play game `seed` on both engines, recording the actions compared in `history`; `Err` on
/// the first mismatch.
#[allow(clippy::result_large_err)]
pub fn play_game(
    seed: u64,
    spec: &InitSpec,
    oracle: &mut Oracle,
    limits: Limits,
    known: &mut Known,
    history: &mut Vec<ActionRecord>,
) -> Result<Outcome, Mismatch> {
    let mut g = from_init(spec);
    let mut rng = Pcg64::seed_from_u64(seed ^ 0xD1FF_0000_0000_0001);
    let init = oracle
        .request(&json!({"cmd": "init", "spec": spec, "tables": tables()}))
        .unwrap_or_else(|e| panic!("oracle init failed: {e}"));
    let sourcing_check = init["sourcing_check"].as_bool().unwrap_or_else(|| {
        panic!("protocol error: init response must declare `sourcing_check` (true/false)")
    });
    let setup = ActionRecord {
        player: 0,
        moves: vec![],
    };
    match compare_snapshots(None, &g, oracle, &setup, limits, known) {
        Ok(()) => {}
        Err(None) => return Ok(Outcome::KnownDivergence),
        Err(Some(m)) => {
            return Err(Mismatch {
                category: format!("init:{}", m.category),
                pending: None,
                ..m
            });
        }
    }
    while !g.is_over() {
        if let Some(m) = compare_legal(&g, oracle, limits, known) {
            return Err(m);
        }
        let (rec, next) = choose_action(&g, &mut rng, limits, known);
        let applied = match oracle.request(&json!({"cmd": "apply", "action": rec})) {
            Ok(v) => v,
            Err(e) => {
                return Err(Mismatch {
                    category: format!("apply-rejected:{}", move_kind(&rec.moves[0])),
                    ours: json!(null),
                    theirs: json!(e),
                    diff: vec![e],
                    pending: Some(rec),
                });
            }
        };
        // Our explicit resource sources against the oracle's canonical ones (an oracle that
        // accepts any individually legal source cannot reject a priority violation).
        let sourcing = match sourcing_of(&applied, sourcing_check) {
            Ok(s) => s,
            Err(e) => {
                return Err(Mismatch {
                    category: "oracle-error:apply".into(),
                    ours: json!(rec),
                    theirs: applied,
                    diff: vec![e],
                    pending: Some(rec),
                });
            }
        };
        if !sourcing.is_empty() {
            return Err(Mismatch {
                category: format!("sourcing:{}", move_kind(&rec.moves[0])),
                ours: json!(rec),
                theirs: applied["action"].clone(),
                diff: sourcing,
                pending: Some(rec),
            });
        }
        // Choices the oracle cannot express but accepted because a listed known difference
        // allows them (`oracles/protocol.md`).
        if let Some(rule) = unlisted_relaxation(&applied, limits, known) {
            return Err(Mismatch {
                category: format!("relaxed:{rule}"),
                ours: json!(rec),
                theirs: applied,
                diff: vec![format!(
                    "oracle relaxed `{rule}`, which [{}] does not list",
                    limits.section()
                )],
                pending: Some(rec),
            });
        }
        let before = std::mem::replace(&mut g, next);
        let era_before = before.era;
        if era_before == Era::Canal && g.era == Era::Rail {
            // The oracle reshuffles with its own RNG, so only the card multiset can be
            // compared; then it is re-dealt from ours.
            if let Some(m) = compare_reshuffle(&g, oracle, &rec) {
                return Err(m);
            }
            let hands: Vec<_> = g.players.iter().map(|p| p.hand.clone()).collect();
            oracle
                .request(&json!({"cmd": "deal", "hands": hands, "deck": g.deck}))
                .unwrap_or_else(|e| panic!("oracle deal failed: {e}"));
        }
        match compare_snapshots(Some(&before), &g, oracle, &rec, limits, known) {
            Ok(()) => history.push(rec),
            Err(None) => {
                history.push(rec);
                return Ok(Outcome::KnownDivergence);
            }
            Err(Some(m)) => return Err(m),
        }
    }
    Ok(Outcome::Complete)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sourcing_field_must_match_the_declared_capability() {
        let with = |v: Value| json!({"ok": true, "action": {}, "sourcing": v});
        let without = json!({"ok": true, "action": {}});
        assert_eq!(sourcing_of(&with(json!([])), true), Ok(vec![]));
        assert_eq!(
            sourcing_of(&with(json!(["coal"])), true),
            Ok(vec!["coal".to_string()])
        );
        assert!(
            sourcing_of(&without, true).is_err(),
            "missing array is an error"
        );
        assert!(sourcing_of(&with(json!("x")), true).is_err());
        assert_eq!(sourcing_of(&without, false), Ok(vec![]));
        assert_eq!(sourcing_of(&with(Value::Null), false), Ok(vec![]));
        assert!(
            sourcing_of(&with(json!([])), false).is_err(),
            "undeclared array is an error"
        );
    }
}
