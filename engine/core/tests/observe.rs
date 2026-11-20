// SPDX-License-Identifier: Apache-2.0
//! Observation and determinization (plan Task 11).

use brass_engine::data::{Card, Era, PlayerId};
use brass_engine::moves::{Move, Phase};
use brass_engine::state::{GameState, check_invariants, new_game};
use rand::seq::SliceRandom;
use rand::{RngExt, SeedableRng};
use rand_pcg::Pcg64;

fn is_wild(c: &Card) -> bool {
    matches!(c, Card::WildLocation | Card::WildIndustry)
}

/// Random play from `new_game(seed)`; `scout_bias` favours Scout so Wilds show up in hands.
/// Stops after `max_moves` moves or when `stop` holds.
fn play(
    seed: u64,
    max_moves: usize,
    scout_bias: bool,
    stop: impl Fn(&GameState) -> bool,
) -> GameState {
    let mut g = new_game(seed);
    let mut rng = Pcg64::seed_from_u64(seed ^ 0xABCD);
    for _ in 0..max_moves {
        if g.is_over() || stop(&g) {
            break;
        }
        let legal = g.legal_moves();
        let scouts: Vec<&Move> = legal
            .iter()
            .filter(|m| matches!(m, Move::Scout { .. }))
            .collect();
        let m = if scout_bias && !scouts.is_empty() && rng.random_range(0..2) == 0 {
            scouts[rng.random_range(0..scouts.len())].clone()
        } else {
            legal[rng.random_range(0..legal.len())].clone()
        };
        g.apply(&m).unwrap();
    }
    g
}

/// A spread of mid-game states: early, mid Canal, and Rail Era.
fn sample_states() -> Vec<GameState> {
    let mut v = Vec::new();
    for seed in 0..6 {
        v.push(new_game(seed));
        v.push(play(seed, 30, true, |_| false));
        v.push(play(seed, 2000, true, |g| {
            g.era == Era::Rail && g.round == 3
        }));
    }
    // The last Rail round (a player has already emptied their hand), an empty deck, and a
    // finished game.
    v.push(play(6, 5000, true, |g| {
        g.era == Era::Rail && g.deck.is_empty() && g.players.iter().any(|p| p.hand.is_empty())
    }));
    v.push(play(7, 5000, true, |g| g.deck.is_empty()));
    v.push(play(8, 5000, true, |_| false));
    v
}

/// Unseen-card multiset from `p`'s point of view plus all face-down cards.
fn hidden_pool(g: &GameState, p: PlayerId) -> Vec<Card> {
    let mut v = g.deck.clone();
    for (q, ps) in g.players.iter().enumerate() {
        if g.era == Era::Canal {
            v.push(ps.discard[0]);
        }
        if q != p as usize {
            v.extend(ps.hand.iter().filter(|c| !is_wild(c)));
        }
    }
    v.sort_unstable();
    v
}

#[test]
fn observe_hides_opponent_hands_and_deck() {
    for g in sample_states() {
        for p in 0..4u8 {
            let base = serde_json::to_string(&g.observe(p)).unwrap();
            // Any state agreeing on the public information and p's hand looks identical.
            for seed in 0..5 {
                let d = g.determinize(p, seed);
                assert_eq!(serde_json::to_string(&d.observe(p)).unwrap(), base);
            }
            // Directly swap an opponent's hidden card with a deck card.
            let q = ((p + 1) % 4) as usize;
            let mut h = g.clone();
            if let Some(i) = h.players[q].hand.iter().position(|c| !is_wild(c))
                && !h.deck.is_empty()
            {
                std::mem::swap(&mut h.players[q].hand[i], &mut h.deck[0]);
                assert_eq!(serde_json::to_string(&h.observe(p)).unwrap(), base);
            }
        }
    }
}

#[test]
fn observe_hides_face_down_discards() {
    let g = play(3, 30, false, |_| false);
    assert_eq!(g.era, Era::Canal);
    for q in 0..4usize {
        // Replace q's face-down card with some other card.
        let mut h = g.clone();
        let old = h.players[q].discard[0];
        let new = h.players[(q + 1) % 4].discard[0];
        let new = if new == old { h.deck[0] } else { new };
        assert_ne!(new, old);
        h.players[q].discard[0] = new;
        for p in 0..4u8 {
            let (a, b) = (g.observe(p), h.observe(p));
            assert_eq!(a, b);
            assert_eq!(
                serde_json::to_string(&a).unwrap(),
                serde_json::to_string(&b).unwrap()
            );
        }
    }
    let o = g.observe(0);
    for q in 0..4 {
        let ps = &g.players[q];
        assert!(o.public.players[q].has_face_down_discard);
        assert_eq!(o.public.players[q].discard, ps.discard[1..].to_vec());
    }
    // Rail Era: every discard is visible and no face-down card exists.
    let r = play(3, 2000, false, |g| g.era == Era::Rail && g.round == 4);
    assert_eq!(r.era, Era::Rail);
    let o = r.observe(0);
    for q in 0..4 {
        assert!(!o.public.players[q].has_face_down_discard);
        assert_eq!(o.public.players[q].discard, r.players[q].discard);
    }
}

#[test]
fn determinize_preserves_public_state_and_my_hand() {
    for g in sample_states() {
        for p in 0..4u8 {
            let d = g.determinize(p, 7);
            assert_eq!(d.public_snapshot(), g.public_snapshot());
            assert_eq!(d.players[p as usize].hand, g.players[p as usize].hand);
            assert_eq!(d.observe(p), g.observe(p));
            assert_eq!(d.deck.len(), g.deck.len());
            for q in 0..4 {
                assert_eq!(d.players[q].hand.len(), g.players[q].hand.len());
                assert_eq!(d.players[q].discard.len(), g.players[q].discard.len());
            }
            check_invariants(&d).unwrap();
            // The determinized state plays on.
            let mut d = d;
            if let Some(m) = d.legal_moves().first().cloned() {
                d.apply(&m).unwrap();
                check_invariants(&d).unwrap();
            }
        }
    }
}

#[test]
fn determinize_conserves_unseen_card_multiset() {
    for g in sample_states() {
        for p in 0..4u8 {
            for seed in 0..4 {
                let d = g.determinize(p, seed);
                assert_eq!(hidden_pool(&d, p), hidden_pool(&g, p));
            }
        }
    }
}

#[test]
fn determinize_keeps_opponent_wilds() {
    let mut checked = 0;
    let mut wilds_seen = 0;
    for seed in 0..20 {
        let g = play(seed, 40, true, |_| false);
        for p in 0..4u8 {
            let d = g.determinize(p, 11);
            for q in 0..4 {
                // Each hand keeps its Wilds (as a multiset: positions within an opponent's
                // hand are not part of the observation).
                let wilds = |h: &[Card]| {
                    let mut w: Vec<Card> = h.iter().copied().filter(is_wild).collect();
                    w.sort_unstable();
                    w
                };
                let w = wilds(&g.players[q].hand);
                assert_eq!(w, wilds(&d.players[q].hand));
                if q != p as usize {
                    checked += w.len();
                }
            }
            let (po, pd) = (g.public_snapshot(), d.public_snapshot());
            for q in 0..4 {
                let n = |c: Card| g.players[q].hand.iter().filter(|&&x| x == c).count() as u8;
                assert_eq!(po.players[q].wild_location_in_hand, n(Card::WildLocation));
                assert_eq!(po.players[q].wild_industry_in_hand, n(Card::WildIndustry));
                assert_eq!(po.players[q], pd.players[q]);
                wilds_seen += po.players[q].wild_location_in_hand as usize
                    + po.players[q].wild_industry_in_hand as usize;
            }
            assert_eq!(d.wild_location_supply, g.wild_location_supply);
            assert_eq!(d.wild_industry_supply, g.wild_industry_supply);
        }
    }
    assert!(wilds_seen > 0);
    assert!(checked > 0, "no opponent held a Wild in any sampled state");
}

#[test]
fn determinize_varies_with_seed() {
    let g = play(1, 30, false, |_| false);
    let a = g.determinize(0, 1);
    assert_eq!(a, g.determinize(0, 1), "same (state, p, seed) must agree");
    let b = g.determinize(0, 2);
    assert_ne!(a.deck, b.deck);
    assert_ne!(a.players[1].hand, b.players[1].hand);
    assert_ne!(g.determinize(0, 1), g.determinize(1, 1));
    // The true future is not recoverable from the re-seeded RNG.
    assert_ne!(a.rng_seed, g.rng_seed);
    assert_ne!(a.rng_seed, b.rng_seed);
    check_invariants(&a).unwrap();
}

#[test]
fn determinize_resamples_face_down_discards() {
    let g = play(5, 30, false, |_| false);
    assert_eq!(g.era, Era::Canal);
    for p in 0..4u8 {
        let mut own_changed = false;
        let mut other_changed = false;
        for seed in 0..30 {
            let d = g.determinize(p, seed);
            for q in 0..4 {
                // Visible discards stay put; only discard[0] moves.
                assert_eq!(d.players[q].discard[1..], g.players[q].discard[1..]);
                if d.players[q].discard[0] != g.players[q].discard[0] {
                    if q == p as usize {
                        own_changed = true;
                    } else {
                        other_changed = true;
                    }
                }
            }
        }
        assert!(own_changed, "p's own face-down card is never resampled");
        assert!(other_changed);
    }
}

#[test]
fn observation_legal_moves_only_for_current_player() {
    let mut g = play(2, 30, false, |_| false);
    for _ in 0..40 {
        let cur = g.current_player().unwrap();
        for p in 0..4u8 {
            let o = g.observe(p);
            if p == cur {
                assert_eq!(o.legal_moves, g.legal_moves());
                assert!(!o.legal_moves.is_empty());
            } else {
                assert!(o.legal_moves.is_empty());
            }
        }
        if matches!(g.phase, Phase::Shortfall { .. }) {
            break;
        }
        let m = g.legal_moves()[0].clone();
        g.apply(&m).unwrap();
    }
}

/// Copy of `g` with every hidden card permuted: opponents' non-Wild hand cards, the deck,
/// and (Canal Era) face-down discards, shuffled together with `seed`. Public state is the same.
fn scramble_hidden(g: &GameState, p: u8, seed: u64) -> GameState {
    let mut h = g.clone();
    let face_down = h.era == Era::Canal;
    let mut pool: Vec<Card> = h.deck.clone();
    for (q, ps) in h.players.iter().enumerate() {
        if face_down && let Some(&c) = ps.discard.first() {
            pool.push(c);
        }
        if q != p as usize {
            pool.extend(ps.hand.iter().filter(|c| !is_wild(c)));
        }
    }
    pool.shuffle(&mut Pcg64::seed_from_u64(seed));
    let mut pool = pool.into_iter();
    for c in h.deck.iter_mut() {
        *c = pool.next().unwrap();
    }
    for (q, ps) in h.players.iter_mut().enumerate() {
        if face_down && let Some(c) = ps.discard.first_mut() {
            *c = pool.next().unwrap();
        }
        if q != p as usize {
            for c in ps.hand.iter_mut().filter(|c| !is_wild(c)) {
                *c = pool.next().unwrap();
            }
        }
    }
    assert!(pool.next().is_none());
    h
}

#[test]
fn sample_states_cover_late_rail_and_game_over() {
    let v = sample_states();
    assert!(v.iter().any(|g| g.era == Era::Rail
        && g.deck.is_empty()
        && !g.is_over()
        && g.players.iter().any(|p| p.hand.is_empty())));
    assert!(v.iter().any(|g| g.deck.is_empty() && !g.is_over()));
    assert!(v.iter().any(|g| g.is_over()));
}

#[test]
fn observation_determinize_depends_only_on_observation() {
    for g in sample_states() {
        for p in 0..4u8 {
            for s in 0..3 {
                let h = scramble_hidden(&g, p, 99 + s);
                assert_eq!(g.observe(p), h.observe(p));
                assert_eq!(g.observe(p).determinize(s), h.observe(p).determinize(s));
                assert_eq!(g.determinize(p, s), h.determinize(p, s));
            }
        }
    }
}

#[test]
fn observation_determinize_valid_in_every_era() {
    for g in sample_states() {
        for p in 0..4u8 {
            let o = g.observe(p);
            let d = o.determinize(5);
            check_invariants(&d).unwrap();
            assert_eq!(d.observe(p), o);
            assert_eq!(hidden_pool(&d, p), hidden_pool(&g, p));
            assert_eq!(d.is_over(), g.is_over());
        }
    }
}

#[test]
fn observation_determinize_is_deterministic_and_varies_with_seed() {
    let g = play(4, 40, false, |_| false);
    let o = g.observe(1);
    assert_eq!(o.determinize(3), o.determinize(3));
    assert!((0..8).any(|s| o.determinize(s) != o.determinize(s + 100)));
}
