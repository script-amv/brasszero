//! Comparing our engine with the oracle: legal summaries, snapshots and the Canal-to-Rail
//! reshuffle, with field-level JSON diffs of any disagreement.

use std::collections::BTreeSet;

use brass_engine::data::Card;
use brass_engine::moves::Move;
use brass_engine::record::{ActionRecord, DiffSnapshot, diff_snapshot, legal_summary};
use brass_engine::state::GameState;
use serde_json::{Value, json};

use crate::known::Known;
use crate::limits::Limits;
use crate::oracle::Oracle;

/// One disagreement, with everything needed to reproduce it.
pub struct Mismatch {
    pub category: String,
    pub ours: Value,
    pub theirs: Value,
    pub diff: Vec<String>,
    pub pending: Option<ActionRecord>,
}

/// Field-level differences between two JSON values, as `path: ours=.. oracle=..` lines.
pub fn json_diff(path: &str, a: &Value, b: &Value, out: &mut Vec<String>) {
    match (a, b) {
        (Value::Object(x), Value::Object(y)) => {
            let keys: BTreeSet<&String> = x.keys().chain(y.keys()).collect();
            for k in keys {
                let (va, vb) = (
                    x.get(k).unwrap_or(&Value::Null),
                    y.get(k).unwrap_or(&Value::Null),
                );
                json_diff(&format!("{path}.{k}"), va, vb, out);
            }
        }
        (Value::Array(x), Value::Array(y)) if x.len() == y.len() => {
            for (i, (va, vb)) in x.iter().zip(y).enumerate() {
                json_diff(&format!("{path}[{i}]"), va, vb, out);
            }
        }
        _ if a != b => out.push(format!("{path}: ours={a} oracle={b}")),
        _ => {}
    }
}

/// Top-level fields named in a diff (`.money[2]: ..` -> `money`).
pub fn diff_fields(diff: &[String]) -> Vec<String> {
    let mut f: Vec<String> = diff
        .iter()
        .map(|l| {
            l.trim_start_matches('.')
                .split(['.', '[', ':'])
                .next()
                .unwrap_or("")
                .to_string()
        })
        .collect();
    f.sort();
    f.dedup();
    f
}

pub fn move_kind(m: &Move) -> &'static str {
    match m {
        Move::Build { .. } => "build",
        Move::Canal { .. } => "canal",
        Move::Rail { .. } => "rail",
        Move::Develop { .. } => "develop",
        Move::Sell { .. } => "sell",
        Move::Loan { .. } => "loan",
        Move::Scout { .. } => "scout",
        Move::Pass { .. } => "pass",
        Move::SecondRail { .. } => "second-rail",
        Move::DevelopSecond { .. } => "develop-second",
        Move::SellAnother { .. } => "sell-another",
        Move::BonusDevelop { .. } => "bonus-develop",
        Move::ShortfallRemove { .. } => "shortfall",
        Move::EndAction => "end",
    }
}

/// The oracle's snapshot, normalized (hands and discards sorted as ours are).
fn their_snapshot(oracle: &mut Oracle) -> Result<DiffSnapshot, String> {
    let v = oracle.request(&json!({"cmd": "snapshot"}))?;
    let mut s: DiffSnapshot = serde_json::from_value(v["snapshot"].clone())
        .map_err(|e| format!("oracle snapshot does not parse: {e}"))?;
    for h in s.hands.iter_mut().chain(s.discards.iter_mut()) {
        h.sort_unstable();
    }
    Ok(s)
}

/// `Ok` if the snapshots agree; `Err(None)` at a known divergence (the game stops without a
/// mismatch); `Err(Some(..))` otherwise.
// A mismatch ends the game, so the size of the error variant does not matter.
#[allow(clippy::result_large_err)]
pub fn compare_snapshots(
    before: Option<&GameState>,
    g: &GameState,
    oracle: &mut Oracle,
    pending: &ActionRecord,
    limits: Limits,
    known: &mut Known,
) -> Result<(), Option<Mismatch>> {
    let ours = diff_snapshot(g);
    let theirs = match their_snapshot(oracle) {
        Ok(s) => s,
        Err(e) => {
            return Err(Some(Mismatch {
                category: "oracle-error:snapshot".into(),
                ours: serde_json::to_value(&ours).unwrap(),
                theirs: json!(e),
                diff: vec![e],
                pending: Some(pending.clone()),
            }));
        }
    };
    if ours == theirs {
        return Ok(());
    }
    if limits.known_divergence(known, &ours, &theirs) {
        return Err(None);
    }
    // A known difference the oracle can be repaired from (`Limits::resync`): repair it and
    // compare again.
    let mut theirs = theirs;
    if let Some(req) = before.and_then(|b| limits.resync(known, b, pending, &ours, &theirs)) {
        let repaired = oracle.request(&req).and_then(|_| their_snapshot(oracle));
        match repaired {
            Ok(s) if s == ours => return Ok(()),
            Ok(s) => theirs = s,
            Err(e) => {
                return Err(Some(Mismatch {
                    category: "oracle-error:resync".into(),
                    ours: serde_json::to_value(&ours).unwrap(),
                    theirs: json!(e),
                    diff: vec![e],
                    pending: Some(pending.clone()),
                }));
            }
        }
    }
    let (a, b) = (
        serde_json::to_value(&ours).unwrap(),
        serde_json::to_value(&theirs).unwrap(),
    );
    let mut diff = Vec::new();
    json_diff("", &a, &b, &mut diff);
    let kind = pending.moves.first().map_or("none", move_kind);
    Err(Some(Mismatch {
        category: format!("snapshot:{kind}:{}", diff_fields(&diff).join("+")),
        ours: a,
        theirs: b,
        diff,
        pending: Some(pending.clone()),
    }))
}

pub fn compare_legal(
    g: &GameState,
    oracle: &mut Oracle,
    limits: Limits,
    known: &mut Known,
) -> Option<Mismatch> {
    let mut ours = legal_summary(g);
    let mut theirs: BTreeSet<String> = match oracle.request(&json!({"cmd": "legal"})) {
        Ok(v) => serde_json::from_value(v["keys"].clone()).expect("keys array"),
        Err(e) => {
            return Some(Mismatch {
                category: "oracle-error:legal".into(),
                ours: json!(ours),
                theirs: json!(e),
                diff: vec![e],
                pending: None,
            });
        }
    };
    limits.filter_keys(known, g, &mut ours, &mut theirs);
    if ours == theirs {
        return None;
    }
    let only_ours: Vec<&String> = ours.difference(&theirs).collect();
    let only_theirs: Vec<&String> = theirs.difference(&ours).collect();
    let family = |k: &&String| k.split(':').next().unwrap_or("").to_string();
    let mut cats = BTreeSet::new();
    cats.extend(only_ours.iter().map(|k| format!("ours-only-{}", family(k))));
    cats.extend(
        only_theirs
            .iter()
            .map(|k| format!("oracle-only-{}", family(k))),
    );
    let mut diff: Vec<String> = only_ours
        .iter()
        .map(|k| format!("ours only: {k}"))
        .collect();
    diff.extend(only_theirs.iter().map(|k| format!("oracle only: {k}")));
    Some(Mismatch {
        category: format!("legal:{}", cats.into_iter().collect::<Vec<_>>().join("+")),
        ours: json!(ours),
        theirs: json!(theirs),
        diff,
        pending: None,
    })
}

/// At the Canal-to-Rail transition, before re-dealing the oracle: both engines must hold the
/// same multiset of cards in hands plus deck, deal the same hand sizes, and keep no discards
/// (every pile went into the new deck).
pub fn compare_reshuffle(
    g: &GameState,
    oracle: &mut Oracle,
    rec: &ActionRecord,
) -> Option<Mismatch> {
    let ours_sizes: Vec<usize> = g.players.iter().map(|p| p.hand.len()).collect();
    let mut ours_pool: Vec<Card> = g.deck.clone();
    g.players.iter().for_each(|p| ours_pool.extend(&p.hand));
    ours_pool.sort_unstable();
    let ours_discards: usize = g.players.iter().map(|p| p.discard.len()).sum();
    let ours = json!({"hand_sizes": ours_sizes, "pool": ours_pool, "discards": ours_discards});
    let theirs = match oracle.request(&json!({"cmd": "cards"})) {
        Ok(v) => {
            let hands: Vec<Vec<Card>> = serde_json::from_value(v["hands"].clone()).expect("hands");
            let deck: Vec<Card> = serde_json::from_value(v["deck"].clone()).expect("deck");
            let pile: Vec<Card> =
                serde_json::from_value(v["discard_pile"].clone()).expect("discard pile");
            let sizes: Vec<usize> = hands.iter().map(Vec::len).collect();
            let mut pool = deck;
            hands.into_iter().for_each(|h| pool.extend(h));
            pool.sort_unstable();
            json!({"hand_sizes": sizes, "pool": pool, "discards": pile.len()})
        }
        Err(e) => json!(e),
    };
    if ours == theirs {
        return None;
    }
    let mut diff = Vec::new();
    for field in ["hand_sizes", "discards"] {
        json_diff(
            &format!(".{field}"),
            &ours[field],
            &theirs[field],
            &mut diff,
        );
    }
    if ours["pool"] != theirs["pool"] {
        // The multiset difference, not two 64-card lists.
        let as_cards =
            |v: &Value| serde_json::from_value::<Vec<Card>>(v.clone()).unwrap_or_default();
        let (mut a, mut b) = (as_cards(&ours["pool"]), as_cards(&theirs["pool"]));
        a.retain(|c| match b.iter().position(|x| x == c) {
            Some(i) => {
                b.remove(i);
                false
            }
            None => true,
        });
        diff.push(format!(
            ".pool: ours only {}, oracle only {}",
            json!(a),
            json!(b)
        ));
    }
    Some(Mismatch {
        category: format!("reshuffle:{}", diff_fields(&diff).join("+")),
        ours,
        theirs,
        diff,
        pending: Some(rec.clone()),
    })
}
