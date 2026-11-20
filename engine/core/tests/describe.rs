// SPDX-License-Identifier: Apache-2.0
//! Move descriptions and observation rendering (Stage 1 Task 7).

use brass_engine::data::loc::BIRMINGHAM;
use brass_engine::data::*;
use brass_engine::describe::{card_name, describe_move, render_observation};
use brass_engine::moves::Move;
use brass_engine::resources::ResourcePlan;
use brass_engine::state::{GameState, new_game};
use rand::SeedableRng;
use rand::seq::IndexedRandom;
use rand_pcg::Pcg64;

#[test]
fn loan_and_end_action() {
    let loan = Move::Loan {
        card: Card::Location(BIRMINGHAM),
    };
    assert_eq!(describe_move(&loan), "loan (discard birmingham)");
    assert_eq!(describe_move(&Move::EndAction), "end action");
}

#[test]
fn build_with_empty_plans() {
    let s = SLOTS
        .iter()
        .position(|d| d.loc == BIRMINGHAM && d.industries.contains(&Industry::Cotton))
        .unwrap() as u8;
    let m = Move::Build {
        card: Card::Location(BIRMINGHAM),
        industry: Industry::Cotton,
        slot: s,
        coal: ResourcePlan::default(),
        iron: ResourcePlan::default(),
    };
    assert_eq!(
        describe_move(&m),
        format!("build cotton at birmingham (slot {s}) (discard birmingham)")
    );
}

#[test]
fn develop_with_market_iron() {
    let m = Move::Develop {
        card: Card::WildIndustry,
        industry: Industry::Iron,
        iron: ResourcePlan {
            from_tiles: vec![],
            from_market: 1,
        },
    };
    assert_eq!(
        describe_move(&m),
        "develop iron (discard wild-industry); iron: market x 1"
    );
}

#[test]
fn card_names() {
    assert_eq!(card_name(Card::WildLocation), "wild-location");
    assert_eq!(
        card_name(Card::Industry(IndustryCard::CottonOrManufacturer)),
        "cotton-or-manufacturer"
    );
}

#[test]
fn render_shows_own_hand_and_money() {
    let g = new_game(2);
    let o = g.observe(0);
    let text = render_observation(&o);
    for c in &o.my_hand {
        assert!(text.contains(&card_name(*c)), "missing {c:?}");
    }
    assert!(text.contains("£17"));
}

fn sample_states() -> Vec<GameState> {
    let mut v = Vec::new();
    for seed in 0..4u64 {
        let mut g = new_game(seed);
        let mut rng = Pcg64::seed_from_u64(seed);
        for step in 0..400 {
            if g.is_over() {
                break;
            }
            if step % 10 == 0 {
                v.push(g.clone());
            }
            let legal = g.legal_moves();
            let m = legal.choose(&mut rng).unwrap().clone();
            g.apply(&m).unwrap();
        }
    }
    v
}

#[test]
fn every_legal_move_has_a_distinct_description() {
    for g in sample_states() {
        let descs: Vec<String> = g.legal_moves().iter().map(describe_move).collect();
        let mut sorted = descs.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), descs.len(), "duplicates in {descs:#?}");
    }
}

#[test]
fn render_works_on_sample_states() {
    for g in sample_states() {
        if let Some(p) = g.current_player() {
            assert!(!render_observation(&g.observe(p)).is_empty());
        }
    }
}
