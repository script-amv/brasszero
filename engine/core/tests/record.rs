// SPDX-License-Identifier: Apache-2.0
//! Canonical records and the cross-engine legal summary (plan Task 12).
//!
//! Pure Rust: nothing here needs node or the vendored oracles.

use std::collections::BTreeSet;

use brass_engine::data::*;
use brass_engine::moves::{Move, Phase};
use brass_engine::record::{
    ActionRecord, action_complete, diff_snapshot, legal_summary, replay, summary_key,
};
use brass_engine::state::{GameState, from_init, init_spec};
use rand::{RngExt, SeedableRng};
use rand_pcg::Pcg64;

/// Play a random game from `init_spec(seed)`, grouping the moves into `ActionRecord`s. Calls
/// `visit` on every state reached (before each move).
fn record_game(seed: u64, mut visit: impl FnMut(&GameState)) -> (GameState, Vec<ActionRecord>) {
    let mut g = from_init(&init_spec(seed));
    let mut rng = Pcg64::seed_from_u64(seed ^ 0x005E_C04D);
    let mut records = Vec::new();
    while !g.is_over() {
        let mut rec = ActionRecord {
            player: g.current_player().expect("game not over"),
            moves: Vec::new(),
        };
        loop {
            visit(&g);
            let legal = g.legal_moves();
            let m = legal[rng.random_range(0..legal.len())].clone();
            g.apply(&m).expect("legal move applies");
            rec.moves.push(m);
            if action_complete(&g) {
                break;
            }
        }
        records.push(rec);
    }
    (g, records)
}

#[test]
fn action_record_replay_reproduces_state() {
    for seed in 0..8 {
        let (end, records) = record_game(seed, |_| {});
        assert!(end.is_over());
        // Every record is one action: a ChooseAction move first, then follow-ups.
        let mut g = from_init(&init_spec(seed));
        for rec in &records {
            assert_eq!(
                g.phase,
                Phase::ChooseAction,
                "seed {seed}: record starts mid-action"
            );
            assert_eq!(g.current_player(), Some(rec.player));
            for m in &rec.moves {
                g.apply(m).unwrap();
            }
        }
        let replayed = replay(&init_spec(seed), &records).expect("replay succeeds");
        assert_eq!(
            replayed, end,
            "seed {seed}: replay differs from the recorded game"
        );
        assert_eq!(diff_snapshot(&replayed), diff_snapshot(&end));

        // Records survive a JSON round trip.
        let json = serde_json::to_string(&records).unwrap();
        let back: Vec<ActionRecord> = serde_json::from_str(&json).unwrap();
        assert_eq!(back, records);
    }
}

#[test]
fn replay_rejects_wrong_player_and_illegal_moves() {
    let (_, records) = record_game(3, |_| {});
    let spec = init_spec(3);
    let mut bad = records.clone();
    bad[0].player = (bad[0].player + 1) % 4;
    assert!(replay(&spec, &bad).is_err());
    let mut bad = records.clone();
    bad[0].moves.push(Move::EndAction);
    assert!(replay(&spec, &bad).is_err());
}

fn industry_name(ind: Industry) -> String {
    serde_json::to_value(ind)
        .unwrap()
        .as_str()
        .unwrap()
        .to_string()
}

/// Every key `legal_summary` may produce, built from the static data.
fn key_universe() -> BTreeSet<String> {
    let mut u: BTreeSet<String> = ["loan", "scout", "pass", "end"]
        .into_iter()
        .map(String::from)
        .collect();
    for s in SLOTS {
        let loc = LOCATIONS[s.loc as usize].id;
        for &ind in s.industries {
            u.insert(format!("build:{loc}:{}", industry_name(ind)));
            u.insert(format!("shortfall:{loc}:{}", industry_name(ind)));
            if matches!(
                ind,
                Industry::Cotton | Industry::Manufacturer | Industry::Pottery
            ) {
                u.insert(format!("sell:{loc}:{}", industry_name(ind)));
            }
        }
    }
    for l in LINKS {
        let mut ends = [LOCATIONS[l.a as usize].id, LOCATIONS[l.b as usize].id];
        ends.sort();
        if l.canal {
            u.insert(format!("canal:{}-{}", ends[0], ends[1]));
        }
        if l.rail {
            u.insert(format!("rail:{}-{}", ends[0], ends[1]));
        }
    }
    for ind in Industry::ALL {
        u.insert(format!("develop:{}", industry_name(ind)));
    }
    u
}

#[test]
fn legal_summary_keys_well_formed() {
    let universe = key_universe();
    let mut seen = BTreeSet::new();
    for seed in 0..6 {
        record_game(seed, |g| {
            let keys = legal_summary(g);
            assert!(!keys.is_empty(), "no keys in a live state");
            for k in &keys {
                assert!(universe.contains(k), "malformed summary key {k:?}");
            }
            // The summary is exactly the keys of the legal moves.
            let from_moves: BTreeSet<String> =
                g.legal_moves().iter().map(|m| summary_key(g, m)).collect();
            assert_eq!(keys, from_moves);
            match g.phase {
                Phase::ChooseAction => {
                    assert!(keys.contains("pass"), "Pass is always legal");
                    assert!(!keys.contains("end"));
                }
                Phase::SecondRail | Phase::DevelopMore | Phase::SellMore => {
                    assert!(keys.contains("end"));
                }
                _ => assert!(!keys.contains("end")),
            }
            seen.extend(
                keys.iter()
                    .map(|k| k.split(':').next().unwrap().to_string()),
            );
        });
    }
    // The random games reach every key family.
    for family in [
        "build", "canal", "rail", "develop", "sell", "loan", "scout", "pass", "end",
    ] {
        assert!(seen.contains(family), "no {family} key seen");
    }
}

#[test]
fn diff_snapshot_projects_state() {
    let (end, _) = record_game(5, |_| {});
    let d = diff_snapshot(&end);
    assert!(d.game_over);
    // Turn-position fields are zeroed once the game is over.
    assert_eq!((d.round, d.turn_idx, d.actions_left), (0, 0, 0));
    for p in 0..4 {
        assert_eq!(d.money[p], end.players[p].money);
        assert_eq!(d.vp[p], end.players[p].vp);
        let mut discard = end.players[p].discard.clone();
        discard.sort();
        assert_eq!(d.discards[p], discard);
        let mut hand = end.players[p].hand.clone();
        hand.sort();
        assert_eq!(d.hands[p], hand);
    }
    let g = from_init(&init_spec(5));
    let d = diff_snapshot(&g);
    assert_eq!((d.round, d.turn_idx, d.actions_left), (1, 0, 1));
    assert_eq!(d.deck_size as usize, g.deck.len());
    assert_eq!(d.discards[0], vec![init_spec(5).discard_bottoms[0]]);
}
