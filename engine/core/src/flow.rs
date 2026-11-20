//! Turn, round and era flow; era-end scoring; game end and ranking.
//!
//! Rules: docs/rules-reference.md section 6 (turns, hand refill), 6.1 (end of round: turn
//! order, income, shortfall), 6.2 (era-end scoring), 6.3 (Canal-to-Rail transition), 6.4
//! (game end); RULES-QUESTIONS.md RQ-004, RQ-011, RQ-026, RQ-043.
//!
//! Every action ends with [`end_action`]. When the last turn of a round ends, the round is
//! closed: turn order, then income seat by seat. A seat that cannot pay negative income and
//! still has tiles worth money pauses the flow in `Phase::Shortfall` (resolved by
//! `actions::shortfall`, which resumes here via [`resume_income`]); then the era may end.

use rand::seq::SliceRandom;

use crate::actions::{actor, shortfall};
use crate::data::*;
use crate::moves::Phase;
use crate::network::link_locs;
use crate::state::GameState;

/// Actions per turn: 1 in the first Canal round, 2 otherwise (section 6).
fn actions_per_turn(g: &GameState) -> u8 {
    if g.era == Era::Canal && g.round == 1 {
        1
    } else {
        2
    }
}

/// The era ends after the round in which the deck and every hand run out (section 6.2).
fn era_exhausted(g: &GameState) -> bool {
    g.deck.is_empty() && g.players.iter().all(|ps| ps.hand.is_empty())
}

/// End-of-action bookkeeping: use one action; after the turn's last action (or once the hand
/// is empty) refill the hand to 8 from the deck and pass to the next seat in turn order,
/// closing the round after the fourth turn.
pub(crate) fn end_action(g: &mut GameState) {
    g.actions_left -= 1;
    let p = actor(g);
    if g.actions_left > 0 && !g.players[p as usize].hand.is_empty() {
        return;
    }
    let ps = &mut g.players[p as usize];
    let draw = HAND_SIZE.saturating_sub(ps.hand.len()).min(g.deck.len());
    ps.hand.extend(g.deck.drain(g.deck.len() - draw..));
    g.turn_idx += 1;
    if (g.turn_idx as usize) < g.turn_order.len() {
        g.actions_left = actions_per_turn(g);
        debug_assert_turn_has_cards(g);
    } else {
        end_round(g);
    }
}

/// RQ-026: a turn never starts with an empty hand (checked at every turn start, including
/// the first turn of a round or era).
fn debug_assert_turn_has_cards(g: &GameState) {
    debug_assert!(
        !g.players[actor(g) as usize].hand.is_empty(),
        "turn starts with an empty hand (RQ-026)"
    );
}

/// Section 6.1: new turn order by money spent (stable), reset spent, then income — except
/// after the final round of the game.
fn end_round(g: &mut GameState) {
    g.turn_idx = 0;
    let spent: [u16; 4] = std::array::from_fn(|p| g.players[p].spent);
    // `sort_by_key` is stable: equal spenders keep their relative order.
    g.turn_order.sort_by_key(|&p| spent[p as usize]);
    for ps in &mut g.players {
        ps.spent = 0;
    }
    if g.era == Era::Rail && era_exhausted(g) {
        finish_round(g);
    } else {
        pay_income_from(g, 0);
    }
}

/// Pay income for seats `from..4`, pausing in `Phase::Shortfall` for a seat that cannot pay
/// and has tiles to remove; then finish the round.
fn pay_income_from(g: &mut GameState, from: PlayerId) {
    for p in from..4 {
        let level = income_level(g.players[p as usize].income_space);
        let ps = &mut g.players[p as usize];
        if level >= 0 {
            ps.money += level as u16;
            continue;
        }
        let due = level.unsigned_abs() as u16;
        let paid = due.min(ps.money);
        ps.money -= paid;
        let owed = due - paid;
        if owed == 0 {
            continue;
        }
        if shortfall::removable(g, p).is_empty() {
            shortfall::lose_vp(g, p, owed);
        } else {
            g.phase = Phase::Shortfall { player: p, owed };
            return;
        }
    }
    finish_round(g);
}

/// Continue the round's income after `player`'s shortfall is settled.
pub(crate) fn resume_income(g: &mut GameState, player: PlayerId) {
    g.phase = Phase::ChooseAction;
    pay_income_from(g, player + 1);
}

/// After income: end the era if it is exhausted, otherwise start the next round.
fn finish_round(g: &mut GameState) {
    g.phase = Phase::ChooseAction;
    g.turn_idx = 0;
    if era_exhausted(g) {
        score_era(g);
        if g.era == Era::Rail {
            g.phase = Phase::GameOver;
            g.game_over = true;
            return;
        }
        start_rail_era(g);
    } else {
        g.round += 1;
    }
    g.actions_left = actions_per_turn(g);
    debug_assert_turn_has_cards(g);
}

/// Link icons at `loc`: flipped tiles of any player plus a merchant space's printed icons.
fn loc_link_vp(g: &GameState, loc: LocId) -> u16 {
    let tiles: u16 = SLOTS
        .iter()
        .zip(&g.slots)
        .filter(|(s, _)| s.loc == loc)
        .filter_map(|(_, t)| t.as_ref().filter(|t| t.flipped))
        .map(|t| tile(t.industry, t.level).link_vp as u16)
        .sum();
    let merchant: u16 = MERCHANT_SPACES
        .iter()
        .filter(|m| m.loc == loc)
        .map(|m| m.link_vp as u16)
        .sum();
    tiles + merchant
}

/// VP each player would gain if the current era were scored now (links, then flipped tiles;
/// rules-reference section 6.2). `score_era` adds exactly this.
pub fn era_end_vp(g: &GameState) -> [u16; 4] {
    let mut vp = [0u16; 4];
    for (def, owner) in LINKS.iter().zip(&g.links) {
        if let Some(owner) = owner {
            vp[*owner as usize] += link_locs(def).map(|l| loc_link_vp(g, l)).sum::<u16>();
        }
    }
    for t in g.slots.iter().flatten().filter(|t| t.flipped) {
        vp[t.owner as usize] += tile(t.industry, t.level).vp as u16;
    }
    vp
}

/// Section 6.2: score every link and flipped tile ([`era_end_vp`]), then remove every link
/// (it returns to its owner, RQ-043).
fn score_era(g: &mut GameState) {
    let vp = era_end_vp(g);
    for (ps, v) in g.players.iter_mut().zip(vp) {
        ps.vp += v;
    }
    g.links.iter_mut().for_each(|l| *l = None);
}

/// Section 6.3: remove level-1 tiles, refill merchant beer, shuffle all discards into a new
/// deck and deal 8 to each player. Turn order stays as set by the last Canal round.
fn start_rail_era(g: &mut GameState) {
    for s in &mut g.slots {
        if s.as_ref().is_some_and(|t| t.level == 1) {
            *s = None;
        }
    }
    for (beer, &t) in g.merchant_beer.iter_mut().zip(&g.merchant_tiles) {
        *beer = t != MerchantTile::Blank;
    }
    let mut deck = std::mem::take(&mut g.deck);
    for ps in &mut g.players {
        deck.append(&mut ps.discard);
    }
    deck.shuffle(&mut g.reshuffle_rng());
    g.reshuffles += 1;
    for ps in &mut g.players {
        let n = HAND_SIZE.min(deck.len());
        ps.hand = deck.split_off(deck.len() - n);
    }
    g.deck = deck;
    g.era = Era::Rail;
    g.round = 1;
}

/// Comparison key for the final standings: VP, then income level, then money (section 6.4,
/// RQ-011).
fn standing(g: &GameState, p: usize) -> (u16, i8, u16) {
    let ps = &g.players[p];
    (ps.vp, income_level(ps.income_space), ps.money)
}

/// Rank of each seat (0 = first). Fully tied players share a rank; the next rank skips
/// accordingly (e.g. `[0, 0, 2, 3]`).
pub fn ranking(g: &GameState) -> [u8; 4] {
    std::array::from_fn(|p| {
        let me = standing(g, p);
        (0..4).filter(|&q| standing(g, q) > me).count() as u8
    })
}

/// The seats ranked first (several on a full tie), ascending.
pub fn winners(g: &GameState) -> Vec<PlayerId> {
    let r = ranking(g);
    (0..4).filter(|&p| r[p as usize] == 0).collect()
}
