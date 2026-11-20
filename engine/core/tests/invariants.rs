// SPDX-License-Identifier: Apache-2.0
//! Invariant fuzzing: random games checked after every move (plan Task 10).
//!
//! After every move: `state::check_invariants` (card, tile, link and market conservation,
//! RQ-026 hand parity), VP never decreases except through a shortfall (section 6.1 step 3,
//! RQ-004), mat tiles never return, tile totals fall by exactly the tiles the move removes, `apply_unchecked` on a clone equals `apply`, and the state
//! survives a serde round trip. A sample of other legal moves must apply cleanly; moves that
//! were legal one step earlier but are not legal now, and randomly altered legal moves, must be
//! accepted exactly when `legal_moves()` lists them and otherwise rejected without changing the
//! state (cross-checks `apply`'s cheaper legality test against `legal_moves()`).
//!
//! The default 200 proptest cases take ~40 s in a debug build on a 2-core laptop.

use brass_engine::data::{
    Card, Era, Industry, IndustryCard, LINKS, LOCATIONS, PlayerId, SLOTS, income_level,
};
use brass_engine::moves::{Move, Phase, RuleError};
use brass_engine::state::{GameState, check_invariants, new_game};
use proptest::prelude::*;
use rand::{RngExt, SeedableRng};
use rand_pcg::Pcg64;

/// No real game comes close (random games take about 160 moves).
const MAX_MOVES: usize = 2000;
/// Other legal moves applied to clones at each step.
const SAMPLED_LEGAL: usize = 2;
/// Previously legal, now illegal moves probed at each step.
const SAMPLED_STALE: usize = 2;
/// Randomly altered legal moves probed at each step.
const SAMPLED_MUTANTS: usize = 2;

/// A random card value, including ones no deck holds (e.g. a merchant location).
fn random_card(rng: &mut Pcg64) -> Card {
    match rng.random_range(0..4) {
        0 => Card::WildLocation,
        1 => Card::WildIndustry,
        2 => Card::Industry(
            [
                IndustryCard::Iron,
                IndustryCard::Coal,
                IndustryCard::Pottery,
                IndustryCard::Brewery,
                IndustryCard::CottonOrManufacturer,
            ][rng.random_range(0..5)],
        ),
        _ => Card::Location(rng.random_range(0..LOCATIONS.len() as u8 + 2)),
    }
}

/// `m` with one parameter changed at random (slot, link, card, industry), possibly out of
/// range. The result may or may not be legal.
fn mutate(m: &Move, rng: &mut Pcg64) -> Move {
    let mut m = m.clone();
    let slot = rng.random_range(0..SLOTS.len() as u8 + 3);
    let link = rng.random_range(0..LINKS.len() as u8 + 3);
    let card = random_card(rng);
    let industry = Industry::ALL[rng.random_range(0..6)];
    let which = rng.random_range(0..2);
    match &mut m {
        Move::Build {
            card: c,
            industry: i,
            slot: s,
            ..
        } => match rng.random_range(0..3) {
            0 => *c = card,
            1 => *i = industry,
            _ => *s = slot,
        },
        Move::Canal { card: c, link: l }
        | Move::Rail {
            card: c, link: l, ..
        } => {
            if which == 0 {
                *c = card
            } else {
                *l = link
            }
        }
        Move::Sell {
            card: c, slot: s, ..
        } => {
            if which == 0 {
                *c = card
            } else {
                *s = slot
            }
        }
        Move::Develop {
            card: c,
            industry: i,
            ..
        } => {
            if which == 0 {
                *c = card
            } else {
                *i = industry
            }
        }
        Move::Loan { card: c } | Move::Pass { card: c } => *c = card,
        Move::Scout { cards } => cards[rng.random_range(0..3)] = card,
        Move::SecondRail { link: l, beer, .. } => {
            if which == 0 {
                *l = link
            } else {
                *beer = slot
            }
        }
        Move::SellAnother { slot: s, .. } => *s = slot,
        Move::DevelopSecond { industry: i, .. } | Move::BonusDevelop { industry: i } => {
            *i = industry
        }
        Move::ShortfallRemove { slot: s } => *s = slot,
        Move::EndAction => {}
    }
    m
}

fn ensure(cond: bool, msg: impl FnOnce() -> String) -> Result<(), String> {
    if cond { Ok(()) } else { Err(msg()) }
}

/// VP only ever drops through a shortfall: the player had negative income and the move ended
/// a round (or removed a shortfall tile).
fn check_vp(prev: &GameState, next: &GameState, m: &Move) -> Result<(), String> {
    let round_closed = (prev.era, prev.round) != (next.era, next.round)
        || matches!(prev.phase, Phase::Shortfall { .. })
        || matches!(next.phase, Phase::Shortfall { .. });
    for p in 0..4 {
        let (before, after) = (prev.players[p].vp, next.players[p].vp);
        if after < before {
            ensure(
                round_closed && income_level(prev.players[p].income_space) < 0,
                || {
                    format!(
                        "player {p} VP fell {before} -> {after} on {m:?}; before {:?} {} {:?} \
                         {:?}; after {:?} {} {:?} {:?}",
                        prev.era,
                        prev.round,
                        prev.phase,
                        prev.players
                            .iter()
                            .map(|q| (q.vp, q.income_space, q.money))
                            .collect::<Vec<_>>(),
                        next.era,
                        next.round,
                        next.phase,
                        next.players
                            .iter()
                            .map(|q| (q.vp, q.income_space, q.money))
                            .collect::<Vec<_>>(),
                    )
                },
            )?;
        }
    }
    Ok(())
}

/// Tiles leave the mat (build, develop) but never come back.
fn check_mat(prev: &GameState, next: &GameState, m: &Move) -> Result<(), String> {
    for p in 0..4 {
        let (a, b) = (
            &prev.players[p].mat_remaining,
            &next.players[p].mat_remaining,
        );
        let ok = a
            .iter()
            .flatten()
            .zip(b.iter().flatten())
            .all(|(x, y)| y <= x);
        ensure(ok, || format!("player {p} mat gained a tile on {m:?}"))?;
    }
    Ok(())
}

/// Tiles `p` still owns: on the mat plus on the board.
fn tiles_owned(g: &GameState, p: usize) -> i32 {
    let mat: i32 = g.players[p]
        .mat_remaining
        .iter()
        .flatten()
        .map(|&n| n as i32)
        .sum();
    let board = g
        .slots
        .iter()
        .flatten()
        .filter(|t| t.owner as usize == p)
        .count() as i32;
    mat + board
}

/// Exact tile accounting per player: the tiles a move removes from the game. Develop (all
/// three kinds) takes one off the actor's mat for good; a Build moves a tile from mat to board
/// (net 0) but an overbuild removes the replaced tile (its owner loses one); a shortfall
/// removal loses one of the payer's tiles; and the Canal-to-Rail transition removes every
/// level-1 tile left on the board (after the move itself). Everything else removes nothing.
fn check_tile_totals(prev: &GameState, next: &GameState, m: &Move) -> Result<(), String> {
    let actor = prev.current_player().expect("game not over") as usize;
    let mut removed = [0i32; 4];
    // Level-1 tiles on the board once the move (but not the era change) has been applied.
    let mut level1 = [0i32; 4];
    for t in prev.slots.iter().flatten().filter(|t| t.level == 1) {
        level1[t.owner as usize] += 1;
    }
    let mut take = |owner: PlayerId, level: u8| {
        removed[owner as usize] += 1;
        if level == 1 {
            level1[owner as usize] -= 1;
        }
    };
    match m {
        // The developed tile never reaches the board, so it is not in `level1`.
        Move::Develop { .. } | Move::DevelopSecond { .. } | Move::BonusDevelop { .. } => {
            removed[actor] += 1;
        }
        Move::Build { industry, slot, .. } => {
            if let Some(old) = &prev.slots[*slot as usize] {
                take(old.owner, old.level);
            }
            if prev.players[actor].lowest_level(*industry) == Some(1) {
                level1[actor] += 1;
            }
        }
        Move::ShortfallRemove { slot } => {
            let old = prev.slots[*slot as usize].as_ref().expect("tile to remove");
            take(old.owner, old.level);
        }
        _ => {}
    }
    if (prev.era, next.era) == (Era::Canal, Era::Rail) {
        for p in 0..4 {
            removed[p] += level1[p];
        }
    }
    for (p, &gone) in removed.iter().enumerate() {
        let (before, after) = (tiles_owned(prev, p), tiles_owned(next, p));
        ensure(before - after == gone, || {
            format!("player {p} tiles {before} -> {after} on {m:?}, expected to lose {gone}")
        })?;
    }
    Ok(())
}

fn check_serde(g: &GameState) -> Result<(), String> {
    let json = serde_json::to_string(g).map_err(|e| format!("serialize: {e}"))?;
    let back: GameState = serde_json::from_str(&json).map_err(|e| format!("deserialize: {e}"))?;
    ensure(back == *g, || "serde round trip changed the state".into())
}

/// Play one random game, checking every invariant after every move. Returns the move count.
fn play_checked(setup_seed: u64, choice_seed: u64) -> Result<usize, String> {
    let mut g = new_game(setup_seed);
    let mut rng = Pcg64::seed_from_u64(choice_seed);
    check_invariants(&g)?;
    check_serde(&g)?;
    let mut prev_legal: Vec<Move> = Vec::new();
    let mut moves = 0;
    while !g.is_over() {
        let ctx = |e: String| format!("seed {setup_seed}/{choice_seed}, move {moves}: {e}");
        let legal = g.legal_moves();
        ensure(!legal.is_empty(), || "no legal move".into()).map_err(ctx)?;
        ensure(moves < MAX_MOVES, || "game too long".into()).map_err(ctx)?;

        // Every legal move applies without error (sampled).
        for _ in 0..SAMPLED_LEGAL.min(legal.len()) {
            let m = &legal[rng.random_range(0..legal.len())];
            let mut c = g.clone();
            c.apply(m)
                .map_err(|e| format!("legal {m:?} rejected: {e}"))
                .and_then(|()| check_invariants(&c))
                .map_err(ctx)?;
        }
        // Moves legal one step earlier but not now are rejected and change nothing.
        let stale: Vec<&Move> = prev_legal.iter().filter(|m| !legal.contains(m)).collect();
        for _ in 0..SAMPLED_STALE.min(stale.len()) {
            let m = stale[rng.random_range(0..stale.len())];
            let mut c = g.clone();
            let r = c.apply(m);
            ensure(r == Err(RuleError::IllegalMove) && c == g, || {
                format!("illegal {m:?} not rejected cleanly: {r:?}")
            })
            .map_err(ctx)?;
        }

        // Altered moves: `apply` accepts exactly those in `legal_moves()`.
        for _ in 0..SAMPLED_MUTANTS {
            let m = mutate(&legal[rng.random_range(0..legal.len())], &mut rng);
            let mut c = g.clone();
            let r = c.apply(&m);
            let ok = if legal.contains(&m) {
                r.is_ok()
            } else {
                r == Err(RuleError::IllegalMove) && c == g
            };
            ensure(ok, || format!("altered {m:?}: apply gave {r:?}")).map_err(ctx)?;
        }

        let m = &legal[rng.random_range(0..legal.len())];
        let prev = g.clone();
        let mut unchecked = g.clone();
        unchecked.apply_unchecked_for_testing(m);
        g.apply(m)
            .map_err(|e| format!("chosen legal {m:?} rejected: {e}"))
            .map_err(ctx)?;
        ensure(unchecked == g, || {
            format!("apply_unchecked differs on {m:?}")
        })
        .map_err(ctx)?;
        check_vp(&prev, &g, m).map_err(ctx)?;
        check_mat(&prev, &g, m).map_err(ctx)?;
        check_tile_totals(&prev, &g, m).map_err(ctx)?;
        check_invariants(&g)
            .map_err(|e| format!("after {m:?}: {e}"))
            .map_err(ctx)?;
        check_serde(&g).map_err(ctx)?;
        prev_legal = legal;
        moves += 1;
    }
    ensure(g.legal_moves().is_empty(), || {
        "moves listed after game end".into()
    })?;
    let m = Move::Pass {
        card: Card::WildLocation,
    };
    ensure(g.clone().apply(&m) == Err(RuleError::GameOver), || {
        "apply after game end".into()
    })?;
    Ok(moves)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn invariants_hold_in_random_games(setup_seed in any::<u64>(), choice_seed in any::<u64>()) {
        let r = play_checked(setup_seed, choice_seed);
        prop_assert!(r.is_ok(), "{}", r.unwrap_err());
    }
}

/// `BRASS_FUZZ_GAMES` overrides the game count (default 100,000); games run on every core.
#[test]
#[ignore = "long: run with --release -- --ignored"]
fn invariants_hold_100k_games() {
    let games: u64 = std::env::var("BRASS_FUZZ_GAMES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(100_000);
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()) as u64;
    let start = std::time::Instant::now();
    let total: usize = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                s.spawn(move || {
                    (t..games)
                        .step_by(threads as usize)
                        .map(|i| play_checked(i, i ^ 0xF00D).unwrap_or_else(|e| panic!("{e}")))
                        .sum::<usize>()
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).sum()
    });
    eprintln!(
        "{games} games, {total} moves, {:.1} s",
        start.elapsed().as_secs_f64()
    );
}
