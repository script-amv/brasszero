// SPDX-License-Identifier: Apache-2.0
//! Turn, round and era flow; scoring; game end.
//! Rules: docs/rules-reference.md section 6 (turn structure, hand refill), 6.1 (end of round:
//! turn order, income, shortfall), 6.2 (era-end scoring), 6.3 (Canal-to-Rail transition), 6.4
//! (game end, tiebreak); RULES-QUESTIONS.md RQ-004 (shortfall), RQ-011 (tiebreak), RQ-026
//! (Scout and era length), RQ-043 (links return at era end).

mod common;

use brass_engine::data::*;
use brass_engine::flow::{era_end_vp, ranking, winners};
use brass_engine::moves::{Move, Phase};
use brass_engine::state::{GameState, new_game};
use common::*;
use rand::{Rng, RngExt, SeedableRng};
use rand_pcg::Pcg64;

use Industry::{Brewery, Coal, Cotton, Manufacturer, Pottery};

const ANY: Card = Card::Location(loc::BIRMINGHAM);

/// The player to act. Panics if nobody is.
fn me(g: &GameState) -> PlayerId {
    g.current_player().expect("someone is to act")
}

/// Pass with the first card in the acting player's hand.
fn pass(g: &mut GameState) {
    let card = g.players[me(g) as usize].hand[0];
    g.apply(&Move::Pass { card }).unwrap();
}

/// A fresh game positioned at the last action of round `round` in `era`: seat
/// `turn_order[3]` holds one card and has one action left. With `last_round`, the deck and
/// every other hand are empty, so that action ends the era.
fn at_last_action(era: Era, round: u8, last_round: bool) -> (GameState, PlayerId) {
    let mut g = empty_game();
    g.era = era;
    g.round = round;
    g.turn_idx = 3;
    g.actions_left = 1;
    if last_round {
        g.deck.clear();
        for ps in &mut g.players {
            ps.hand.clear();
        }
    }
    let p = g.turn_order[3];
    g.players[p as usize].hand = vec![ANY];
    (g, p)
}

/// Space of the top of income `level`.
fn income_at(level: i8) -> u8 {
    level_top_space(level)
}

// ---------------------------------------------------------------- turns and rounds

// Section 6: the first Canal round gives 1 action per turn; every later turn has 2.
#[test]
fn first_round_one_action_then_two() {
    let mut g = new_game(3);
    let order = g.turn_order;
    for (i, &p) in order.iter().enumerate() {
        assert_eq!((g.round, g.turn_idx, g.actions_left), (1, i as u8, 1));
        assert_eq!(me(&g), p);
        pass(&mut g);
    }
    assert_eq!(
        (g.era, g.round, g.turn_idx, g.actions_left),
        (Era::Canal, 2, 0, 2)
    );
    let first = me(&g);
    pass(&mut g);
    assert_eq!((me(&g), g.actions_left), (first, 1));
    pass(&mut g);
    assert_eq!((g.turn_idx, g.actions_left), (1, 2));
    assert_ne!(me(&g), first);
}

// Section 6: the hand is refilled to 8 after the turn's last action, never between actions;
// once the deck is empty there is no refill. A player whose hand empties ends the turn early.
#[test]
fn refill_after_turn_not_after_action() {
    let mut g = new_game(4);
    // Round 1 (1 action): refilled right after the only action.
    let p = me(&g);
    pass(&mut g);
    assert_eq!(g.players[p as usize].hand.len(), 8);
    assert_eq!(g.deck.len(), 27);
    for _ in 0..3 {
        pass(&mut g);
    }
    // Round 2: no refill after the first action, refill after the second.
    let p = me(&g);
    pass(&mut g);
    assert_eq!(g.players[p as usize].hand.len(), 7);
    assert_eq!(g.deck.len(), 24);
    pass(&mut g);
    assert_eq!(g.players[p as usize].hand.len(), 8);
    assert_eq!(g.deck.len(), 22);

    // Deck with 1 card: the hand is topped up as far as it goes.
    let p = me(&g);
    g.deck.truncate(1);
    pass(&mut g);
    pass(&mut g);
    assert_eq!(g.players[p as usize].hand.len(), 7);
    assert!(g.deck.is_empty());
    // Empty deck: the next player's hand shrinks.
    let q = me(&g);
    pass(&mut g);
    pass(&mut g);
    assert_eq!(g.players[q as usize].hand.len(), 6);

    // A hand emptied by the first action ends the turn early (the round's last turn, so the
    // next round starts).
    let r = me(&g);
    assert_eq!((g.round, g.turn_idx), (2, 3));
    g.players[r as usize].hand = vec![ANY];
    pass(&mut g);
    assert_eq!((g.round, g.turn_idx, g.actions_left), (3, 0, 2));
    assert!(g.players[r as usize].hand.is_empty());
}

// 6.1 step 1: least spent goes first; equal spenders keep their relative order; spent resets.
#[test]
fn turn_order_by_spent_stable_ties() {
    let (mut g, _) = at_last_action(Era::Canal, 2, false);
    let [a, b, c, d] = g.turn_order;
    for (p, s) in [(a, 5), (b, 0), (c, 5), (d, 0)] {
        g.players[p as usize].spent = s;
    }
    pass(&mut g);
    assert_eq!(g.turn_order, [b, d, a, c]);
    assert!(g.players.iter().all(|ps| ps.spent == 0));
    assert_eq!((g.round, g.turn_idx), (3, 0));
    assert_eq!(me(&g), b);

    // All different: sorted ascending.
    let (mut g, _) = at_last_action(Era::Canal, 2, false);
    let [a, b, c, d] = g.turn_order;
    for (p, s) in [(a, 9), (b, 3), (c, 1), (d, 4)] {
        g.players[p as usize].spent = s;
    }
    pass(&mut g);
    assert_eq!(g.turn_order, [c, b, d, a]);
}

// 6.1 step 2: income equals the income level (not the space), paid every round including the
// last Canal round, but not after the final round of the game.
#[test]
fn income_paid_each_round_but_not_final() {
    let levels: [i8; 4] = [5, 0, -2, 30];
    let set = |g: &mut GameState| {
        for (ps, &l) in g.players.iter_mut().zip(&levels) {
            ps.income_space = income_at(l);
            ps.money = 20;
        }
    };
    // An ordinary round.
    let (mut g, _) = at_last_action(Era::Canal, 2, false);
    set(&mut g);
    pass(&mut g);
    let money: Vec<u16> = g.players.iter().map(|ps| ps.money).collect();
    assert_eq!(money, vec![25, 20, 18, 50]);
    // Paying never moves the income marker.
    assert_eq!(g.players[2].income_space, income_at(-2));

    // The last Canal round still pays income.
    let (mut g, _) = at_last_action(Era::Canal, 8, true);
    set(&mut g);
    pass(&mut g);
    assert_eq!(g.era, Era::Rail);
    let money: Vec<u16> = g.players.iter().map(|ps| ps.money).collect();
    assert_eq!(money, vec![25, 20, 18, 50]);

    // The final round of the game does not.
    let (mut g, _) = at_last_action(Era::Rail, 8, true);
    set(&mut g);
    pass(&mut g);
    assert!(g.is_over());
    assert_eq!(g.phase, Phase::GameOver);
    assert!(g.players.iter().all(|ps| ps.money == 20));
    assert!(g.legal_moves().is_empty());
    assert_eq!(g.current_player(), None);
}

// 6.1 step 3 / RQ-004: a player who cannot pay negative income pays what they can, then removes
// own tiles for half their cost (rounded down), stopping once covered and keeping the excess.
// Income is not reduced and the round then continues normally.
#[test]
fn negative_income_shortfall_sells_tiles() {
    let (mut g, _) = at_last_action(Era::Canal, 2, false);
    let q = 1;
    g.players[q as usize].income_space = income_at(-5);
    g.players[q as usize].money = 2;
    let cotton = place(&mut g, "birmingham", 0, q, Cotton, 1, 0);
    // An opponent's tile is never offered.
    place(&mut g, "birmingham", 1, (q + 1) % 4, Cotton, 1, 0);
    pass(&mut g);
    assert_eq!(g.phase, Phase::Shortfall { player: q, owed: 3 });
    assert_eq!(me(&g), q);
    assert_eq!(g.players[q as usize].money, 0);
    assert_eq!(
        g.legal_moves(),
        vec![Move::ShortfallRemove { slot: cotton }]
    );
    g.apply(&Move::ShortfallRemove { slot: cotton }).unwrap();
    // Cotton I costs £12: removing it raises £6, £3 pays the debt, £3 is kept.
    assert_eq!(g.players[q as usize].money, 3);
    assert!(g.slots[cotton as usize].is_none());
    assert_eq!(g.players[q as usize].income_space, income_at(-5));
    assert_eq!(g.players[q as usize].vp, 0);
    assert_eq!((g.phase, g.round, g.turn_idx), (Phase::ChooseAction, 3, 0));
    assert_eq!(g.actions_left, 2);

    // The first removal covers the debt while another removable tile remains: the shortfall
    // ends at once and the second tile stays on the board.
    let (mut g, _) = at_last_action(Era::Canal, 2, false);
    g.players[q as usize].income_space = income_at(-5);
    g.players[q as usize].money = 0;
    let cotton = place(&mut g, "birmingham", 0, q, Cotton, 1, 0); // £12 -> £6
    let coal = place(&mut g, "dudley", 0, q, Coal, 1, 2); // £5 -> £2
    pass(&mut g);
    assert_eq!(g.phase, Phase::Shortfall { player: q, owed: 5 });
    g.apply(&Move::ShortfallRemove { slot: cotton }).unwrap();
    assert_eq!(g.phase, Phase::ChooseAction);
    assert_eq!(g.players[q as usize].money, 1);
    assert!(g.slots[cotton as usize].is_none());
    assert!(g.slots[coal as usize].is_some());
    assert_eq!((g.round, g.turn_idx), (3, 0));

    // With enough cash no tile is touched.
    let (mut g, _) = at_last_action(Era::Canal, 2, false);
    g.players[q as usize].income_space = income_at(-5);
    g.players[q as usize].money = 5;
    let cotton = place(&mut g, "birmingham", 0, q, Cotton, 1, 0);
    pass(&mut g);
    assert_eq!(g.phase, Phase::ChooseAction);
    assert_eq!(g.players[q as usize].money, 0);
    assert!(g.slots[cotton as usize].is_some());
}

// Review Focus 4 / RQ-004: the player chooses which tile to remove, one at a time (flipped or
// not); £0 tiles are never required; only with no tiles left to raise money does the player
// lose 1 VP per £1 short, never below 0. Several short players resolve one after another.
#[test]
fn shortfall_player_chooses_tiles_then_loses_vp() {
    let (mut g, _) = at_last_action(Era::Canal, 2, false);
    let (q, r) = (0, 2);
    for p in [q, r] {
        g.players[p as usize].income_space = income_at(-10);
        g.players[p as usize].money = 0;
    }
    g.players[q as usize].vp = 5;
    g.players[r as usize].vp = 3;
    let cotton = place(&mut g, "birmingham", 0, q, Cotton, 1, 0); // £12 -> £6
    let coal = place(&mut g, "dudley", 0, q, Coal, 1, 2); // £5 -> £2
    g.slots[cotton as usize].as_mut().unwrap().flipped = true;
    let pottery = place(&mut g, "stoke-on-trent", 0, q, Pottery, 2, 0); // £0
    pass(&mut g);

    // Seat q first: both paying tiles offered, the £0 Pottery is not.
    assert_eq!(
        g.phase,
        Phase::Shortfall {
            player: q,
            owed: 10
        }
    );
    let mut offered = g.legal_moves();
    offered.sort();
    let mut expect = vec![
        Move::ShortfallRemove { slot: cotton },
        Move::ShortfallRemove { slot: coal },
    ];
    expect.sort();
    assert_eq!(offered, expect);
    assert!(
        !g.legal_moves()
            .contains(&Move::ShortfallRemove { slot: pottery })
    );
    // q removes the coal mine (cubes go with it) and still owes £8.
    g.apply(&Move::ShortfallRemove { slot: coal }).unwrap();
    assert_eq!(g.phase, Phase::Shortfall { player: q, owed: 8 });
    assert!(g.slots[coal as usize].is_none());
    assert_eq!(
        g.legal_moves(),
        vec![Move::ShortfallRemove { slot: cotton }]
    );
    g.apply(&Move::ShortfallRemove { slot: cotton }).unwrap();
    // £2 still short with only the £0 Pottery left: lose 2 VP; the Pottery stays.
    assert_eq!(g.players[q as usize].vp, 3);
    assert_eq!(g.players[q as usize].money, 0);
    assert!(g.slots[pottery as usize].is_some());

    // Seat r has no tiles: loses VP at once (no decision), floored at 0.
    assert_eq!(g.players[r as usize].vp, 0);
    assert_eq!(g.players[r as usize].money, 0);
    assert_eq!((g.phase, g.round), (Phase::ChooseAction, 3));
}

// RQ-026 era length: play a whole game by `pick` and record each era's round count. Also
// checks that every 2-action turn starts with an even hand of at least 2 cards.
fn era_rounds(seed: u64, mut pick: impl FnMut(&GameState, &[Move]) -> Move) -> (u8, u8) {
    let mut g = new_game(seed);
    let mut max = [0u8; 2];
    let mut last_turn = None;
    for _ in 0..2000 {
        if g.is_over() {
            break;
        }
        let era = g.era as usize;
        max[era] = max[era].max(g.round);
        if g.phase == Phase::ChooseAction {
            let turn = (g.era, g.round, g.turn_idx);
            if last_turn != Some(turn) && g.actions_left == 2 {
                let n = g.players[me(&g) as usize].hand.len();
                assert!(
                    n >= 2 && n.is_multiple_of(2),
                    "seed {seed} {turn:?}: hand of {n}"
                );
            }
            last_turn = Some(turn);
        }
        let moves = g.legal_moves();
        assert!(!moves.is_empty(), "seed {seed}: no legal move");
        let m = pick(&g, &moves);
        g.apply(&m).unwrap();
    }
    assert!(g.is_over(), "seed {seed} did not finish");
    (max[0], max[1])
}

// Section 6: each era is exactly 8 rounds at 4 players.
#[test]
fn canal_era_has_8_rounds_when_no_scout() {
    for seed in 0..4u64 {
        let mut rng = Pcg64::seed_from_u64(seed);
        let rounds = era_rounds(seed, |_, moves| {
            let plain: Vec<&Move> = moves
                .iter()
                .filter(|m| !matches!(m, Move::Scout { .. }))
                .collect();
            plain[rng.random_range(0..plain.len())].clone()
        });
        assert_eq!(rounds, (8, 8), "seed {seed}");
    }
}

// Review Focus 3 / RQ-026: Scout puts 3 cards out and 2 Wilds in, so a game full of Scouts
// still has exactly 8 rounds per era.
#[test]
fn era_has_8_rounds_with_scouts() {
    for seed in 0..4u64 {
        let mut rng = Pcg64::seed_from_u64(seed + 100);
        let mut scouts = 0;
        let rounds = era_rounds(seed, |_, moves| {
            if let Some(m) = moves.iter().find(|m| matches!(m, Move::Scout { .. })) {
                scouts += 1;
                return m.clone();
            }
            moves[rng.random_range(0..moves.len())].clone()
        });
        assert_eq!(rounds, (8, 8), "seed {seed}");
        assert!(scouts >= 10, "seed {seed}: only {scouts} scouts");
    }
}

// ---------------------------------------------------------------- era end

// 6.2: each link scores its owner the link icons of every location it touches: flipped tiles
// of any player and the 2 printed icons of a merchant location (even with a blank merchant
// tile); unflipped tiles add nothing. Links are removed, then flipped tiles score their VP.
#[test]
fn link_scoring_includes_merchant_icons_and_flipped_tiles() {
    let (mut g, _) = at_last_action(Era::Canal, 8, true);
    let p = 0;
    let o = 1;
    g.merchant_tiles[0] = MerchantTile::Blank; // Shrewsbury
    // Birmingham: an opponent's flipped Cotton I (1 icon), p's unflipped Manufacturer (none).
    let opp_cotton = place(&mut g, "birmingham", 0, o, Cotton, 1, 0);
    g.slots[opp_cotton as usize].as_mut().unwrap().flipped = true;
    place(&mut g, "birmingham", 1, p, Manufacturer, 1, 0);
    // Worcester: p's flipped Cotton II (2 icons); southern farm: o's flipped Brewery I (2).
    let w = place(&mut g, "worcester", 0, p, Cotton, 2, 0);
    g.slots[w as usize].as_mut().unwrap().flipped = true;
    let farm = place(&mut g, "farm-south", 0, o, Brewery, 1, 0);
    g.slots[farm as usize].as_mut().unwrap().flipped = true;
    link(&mut g, "birmingham", "oxford", p); // 1 + 2
    link(&mut g, "kidderminster", "worcester", p); // 0 + 2 + 2 (farm brewery)
    link(&mut g, "coalbrookdale", "shrewsbury", p); // 0 + 2 (blank merchant)
    link(&mut g, "birmingham", "worcester", o); // 1 + 2
    pass(&mut g);
    // p: links 3 + 4 + 2 = 9, tiles Cotton II 5 VP.
    assert_eq!(g.players[p as usize].vp, 9 + 5);
    // o: link 3, tiles Cotton I 5 + Brewery I 4.
    assert_eq!(g.players[o as usize].vp, 3 + 5 + 4);
    assert!(g.links.iter().all(Option::is_none));
}

// 6.2: `era_end_vp` is what era-end scoring would add now: link icons for link owners, then
// each flipped tile's VP for its owner.
#[test]
fn era_end_vp_counts_links_and_flipped_tiles() {
    let mut g = new_game(1);
    let (ind, lvl) = (Cotton, 1);
    let s = place(&mut g, "birmingham", 0, 1, ind, lvl, 0);
    g.slots[s as usize].as_mut().unwrap().flipped = true;
    link(&mut g, "birmingham", "oxford", 0);
    let def = TILES
        .iter()
        .find(|t| t.industry == ind && t.level == lvl)
        .unwrap();
    let icons_at = |l: LocId| -> u16 {
        let tile_icons = if l == loc::BIRMINGHAM {
            def.link_vp as u16
        } else {
            0
        };
        let merchant: u16 = MERCHANT_SPACES
            .iter()
            .filter(|m| m.loc == l)
            .map(|m| m.link_vp as u16)
            .sum();
        tile_icons + merchant
    };
    let expected_link = icons_at(loc::BIRMINGHAM) + icons_at(loc::OXFORD);
    let vp = era_end_vp(&g);
    assert_eq!(vp[1], tile(ind, lvl).vp as u16);
    assert_eq!(vp[0], expected_link);
    assert_eq!((vp[2], vp[3]), (0, 0));
}

// 6.2 step 1 / RQ-043: links return to their owners at era end, so a player who used all 14
// Link tiles in the Canal Era has all 14 again for rails.
#[test]
fn links_return_to_owners_at_era_end() {
    let (mut g, _) = at_last_action(Era::Canal, 8, true);
    let p = g.turn_order[0];
    let canal_links: Vec<usize> = (0..LINKS.len()).filter(|&i| LINKS[i].canal).collect();
    for &i in canal_links.iter().take(LINK_TILES) {
        g.links[i] = Some(p);
    }
    pass(&mut g);
    assert_eq!((g.era, me(&g)), (Era::Rail, p));
    assert!(g.links.iter().all(Option::is_none));
    // No presence, market coal: rails to merchant locations are offered again.
    assert!(
        g.legal_moves()
            .iter()
            .any(|m| matches!(m, Move::Rail { .. }))
    );
}

// 6.3: at the end of the Canal Era level-1 tiles leave the board (with their cubes), level 2+
// tiles stay with their contents, merchant beer is refilled beside non-blank tiles, all discards
// are shuffled into a new deck with the game RNG and each player draws 8. The Rail Era starts
// at round 1 with 2 actions per turn and the last Canal round's turn order.
#[test]
fn era_transition_removes_level1_and_reshuffles() {
    let build = |seed: u64| {
        let (mut g, _) = at_last_action(Era::Canal, 8, true);
        g.rng_seed = seed;
        let l1 = place(&mut g, "birmingham", 0, 0, Cotton, 1, 0);
        g.slots[l1 as usize].as_mut().unwrap().flipped = true;
        let l1_coal = place(&mut g, "dudley", 0, 1, Coal, 1, 2);
        let l2 = place(&mut g, "dudley", 1, 2, Coal, 2, 3);
        g.merchant_beer = vec![false; 9];
        // All 64 cards are in the discard piles, except the one card the last player passes with.
        let mut cards = deck_4p();
        let i = cards.iter().position(|&c| c == ANY).unwrap();
        cards.remove(i);
        for (k, ps) in g.players.iter_mut().enumerate() {
            ps.discard = cards[k * 16..((k + 1) * 16).min(cards.len())].to_vec();
        }
        (g, l1, l1_coal, l2)
    };
    let (mut g, l1, l1_coal, l2) = build(11);
    let order = g.turn_order;
    pass(&mut g);
    assert_eq!((g.era, g.round, g.turn_idx), (Era::Rail, 1, 0));
    assert_eq!((g.actions_left, g.phase), (2, Phase::ChooseAction));
    assert_eq!(g.turn_order, order);
    assert!(g.slots[l1 as usize].is_none());
    assert!(g.slots[l1_coal as usize].is_none());
    let kept = g.slots[l2 as usize].as_ref().unwrap();
    assert_eq!((kept.level, kept.resources, kept.flipped), (2, 3, false));
    let expect_beer: Vec<bool> = g
        .merchant_tiles
        .iter()
        .map(|&t| t != MerchantTile::Blank)
        .collect();
    assert_eq!(g.merchant_beer, expect_beer);
    // Every card is back in play: 8 per hand, 32 in the deck, empty discards.
    let mut all = g.deck.clone();
    for ps in &g.players {
        assert_eq!(ps.hand.len(), 8);
        assert!(ps.discard.is_empty());
        all.extend(&ps.hand);
    }
    assert_eq!(g.deck.len(), 32);
    let mut full = deck_4p();
    full.sort();
    all.sort();
    assert_eq!(all, full);
    assert_eq!(g.reshuffles, 1);

    // Deterministic for a seed; a different seed shuffles differently.
    let (mut again, ..) = build(11);
    pass(&mut again);
    assert_eq!(again, g);
    let (mut other, ..) = build(12);
    pass(&mut other);
    assert_ne!(other.deck, g.deck);
}

// Parked carry-forward: the era-2 reshuffle stream of game `s` is domain-separated from the
// setup stream of game `s + 1` (and of game `s`).
#[test]
fn reshuffle_rng_independent_of_setup_streams() {
    for s in [0u64, 1, 42, u64::MAX - 1] {
        let g = new_game(s);
        let mut reshuffle = g.reshuffle_rng();
        let r: Vec<u64> = (0..4).map(|_| reshuffle.next_u64()).collect();
        for seed in [s, s.wrapping_add(1), s.wrapping_add(2)] {
            let mut setup = Pcg64::seed_from_u64(seed);
            let x: Vec<u64> = (0..4).map(|_| setup.next_u64()).collect();
            assert_ne!(r, x, "game {s} reshuffle vs setup seed {seed}");
        }
    }
}

// ---------------------------------------------------------------- game end

// 6.4 / RQ-011: most VP wins; ties broken by income level (not space), then money; still tied
// → shared placement.
#[test]
fn tiebreak_income_then_money() {
    let mut g = empty_game();
    let set = |g: &mut GameState, p: usize, vp: u16, space: u8, money: u16| {
        g.players[p].vp = vp;
        g.players[p].income_space = space;
        g.players[p].money = money;
    };
    set(&mut g, 0, 50, income_at(10), 5);
    set(&mut g, 1, 50, income_at(10), 9);
    set(&mut g, 2, 50, income_at(8), 100);
    set(&mut g, 3, 40, income_at(30), 100);
    assert_eq!(ranking(&g), [1, 0, 2, 3]);
    assert_eq!(winners(&g), vec![1]);

    // Spaces 29 and 30 are both level 10: income ties, money decides.
    set(&mut g, 0, 50, 29, 9);
    set(&mut g, 1, 50, 30, 9);
    assert_eq!(ranking(&g), [0, 0, 2, 3]);
    assert_eq!(winners(&g), vec![0, 1]);

    // VP first, whatever the income and money.
    set(&mut g, 3, 51, income_at(-10), 0);
    assert_eq!(ranking(&g), [1, 1, 3, 0]);
    assert_eq!(winners(&g), vec![3]);
}

/// Play `seed` with uniformly random legal moves; returns the number of moves.
fn random_game(seed: u64) -> usize {
    let mut g = new_game(seed);
    let mut rng = Pcg64::seed_from_u64(seed ^ 0x9E37_79B9_7F4A_7C15);
    for n in 0..2000 {
        if g.is_over() {
            assert!(!winners(&g).is_empty());
            return n;
        }
        let moves = g.legal_moves();
        assert!(!moves.is_empty(), "seed {seed}: deadlock after {n} moves");
        let m = &moves[rng.random_range(0..moves.len())];
        g.apply(m).unwrap();
    }
    panic!("seed {seed}: not over after 2000 moves");
}

// Every random game ends within 2,000 moves, with no deadlock or panic (fast subset).
#[test]
fn random_games_always_terminate() {
    for seed in 0..50 {
        random_game(seed);
    }
}

// The full 1,000-seed run; slow in debug builds. Run with
// `cargo test --release --test flow -- --ignored`.
#[test]
#[ignore]
fn random_games_always_terminate_1000() {
    for seed in 0..1000 {
        random_game(seed);
    }
}

// 6.1 / 6.2 / RQ-021: at the end of the last Canal round, a short player settles the shortfall
// (tile removal) before era-end scoring, so the removed flipped tile no longer scores and the
// removed tile's link icons are gone. Here the flipped Cotton I is the only tile that could pay.
#[test]
fn canal_end_shortfall_before_era_scoring() {
    let (mut g, _) = at_last_action(Era::Canal, 8, true);
    let q = 1;
    g.players[q as usize].income_space = income_at(-5);
    g.players[q as usize].money = 0;
    let cotton = place(&mut g, "birmingham", 0, q, Cotton, 1, 0);
    g.slots[cotton as usize].as_mut().unwrap().flipped = true;
    pass(&mut g);
    // The era is not scored yet: the shortfall comes first.
    assert_eq!(g.phase, Phase::Shortfall { player: q, owed: 5 });
    assert_eq!(g.era, Era::Canal);
    assert_eq!(g.players[q as usize].vp, 0);
    g.apply(&Move::ShortfallRemove { slot: cotton }).unwrap();
    // Scoring ran after the removal: the tile scored nothing, and the Rail Era has begun.
    assert_eq!(g.era, Era::Rail);
    assert_eq!(g.players[q as usize].vp, 0);
    assert!(g.slots[cotton as usize].is_none());
}
