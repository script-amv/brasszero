// SPDX-License-Identifier: Apache-2.0
//! Setup tests. Rules source: docs/rules-reference.md section 5 (Setup, 4 players) [R p.4-5],
//! section 6 (first Canal round has 1 action), section 1 track constants.

use brass_engine::data::*;
use brass_engine::state::{from_init, init_spec, new_game};

// docs/rules-reference.md section 5 steps 3, 6, 7: 8 cards + 1 discard each, deck 64-36=28,
// 17 pounds, income space 10, coal 13, iron 8, wilds 4+4 [R p.4-5].
#[test]
fn setup_deals_8_cards_and_one_discard_each() {
    let g = new_game(1);
    for p in &g.players {
        assert_eq!(p.hand.len(), 8);
        assert_eq!(p.discard.len(), 1);
        assert_eq!(p.money, 17);
        assert_eq!(p.income_space, 10);
        assert_eq!(p.vp, 0);
        assert_eq!(p.spent, 0);
    }
    assert_eq!(g.deck.len(), 28);
    assert_eq!((g.coal_market, g.iron_market), (13, 8));
    assert_eq!((g.wild_location_supply, g.wild_industry_supply), (4, 4));
    // Section 6: first Canal round is 1 action each.
    assert_eq!((g.era, g.round, g.actions_left), (Era::Canal, 1, 1));
    assert!(!g.game_over);
    assert_eq!(g.slots.len(), SLOTS.len());
    assert!(g.slots.iter().all(Option::is_none));
    assert_eq!(g.links.len(), LINKS.len());
    assert!(g.links.iter().all(Option::is_none));
}

// Section 5 step 3: all 64 cards are dealt from one deck, none lost or duplicated.
#[test]
fn all_64_cards_accounted_for() {
    let g = new_game(7);
    let mut all = g.deck.clone();
    for p in &g.players {
        all.extend(&p.hand);
        all.extend(&p.discard);
    }
    let mut want = deck_4p();
    all.sort();
    want.sort();
    assert_eq!(all, want);
}

// Section 5 step 5: beer beside every non-blank merchant tile only [FAQ: do-blank-merchant-tiles-get-beer].
#[test]
fn merchant_beer_on_every_non_blank_tile() {
    let g = new_game(3);
    assert_eq!(g.merchant_tiles.len(), 9);
    assert_eq!(g.merchant_beer.len(), 9);
    for i in 0..9 {
        assert_eq!(
            g.merchant_beer[i],
            g.merchant_tiles[i] != MerchantTile::Blank
        );
    }
    let mut got = g.merchant_tiles.clone();
    let mut want = MERCHANT_TILES_4P.to_vec();
    got.sort();
    want.sort();
    assert_eq!(got, want);
}

// Section 5 step 7: 45 industry tiles per mat, from the TILES counts.
#[test]
fn mat_matches_tile_counts() {
    let g = new_game(5);
    for p in &g.players {
        let total: u32 = p.mat_remaining.iter().flatten().map(|&n| n as u32).sum();
        assert_eq!(total, 45);
        for t in TILES {
            assert_eq!(
                p.mat_remaining[t.industry as usize][t.level as usize - 1],
                t.count
            );
        }
    }
}

#[test]
fn turn_order_is_a_permutation() {
    let g = new_game(11);
    let mut o = g.turn_order;
    o.sort();
    assert_eq!(o, [0, 1, 2, 3]);
    assert_eq!(g.turn_idx, 0);
}

#[test]
fn same_seed_same_state_different_seed_differs() {
    let a = serde_json::to_string(&new_game(42)).unwrap();
    let b = serde_json::to_string(&new_game(42)).unwrap();
    let c = serde_json::to_string(&new_game(43)).unwrap();
    assert_eq!(a, b);
    assert_ne!(a, c);
}

#[test]
fn state_serde_roundtrips() {
    let g = new_game(9);
    let s = serde_json::to_string(&g).unwrap();
    let back: brass_engine::state::GameState = serde_json::from_str(&s).unwrap();
    assert_eq!(g, back);
}

#[test]
fn from_init_roundtrips_new_game() {
    for s in [0u64, 1, 99] {
        assert_eq!(new_game(s), from_init(&init_spec(s)));
    }
}
