//! One module per action. Each exposes `legal(g)` (its moves for the acting player, in a
//! deterministic order) and `apply(g, ..)` (assumes the move is legal). `legal.rs` and
//! `apply.rs` only dispatch.

pub mod build;
pub mod develop;
pub mod loan;
pub mod network;
pub mod pass;
pub mod scout;
pub mod sell;
pub mod shortfall;

use crate::data::{Card, PlayerId};
use crate::state::GameState;

/// The player whose turn it is (`turn_order[turn_idx]`). Not meaningful during
/// `Phase::Shortfall` (whose player is in the phase) or once the game is over.
pub(crate) fn actor(g: &GameState) -> PlayerId {
    g.turn_order[g.turn_idx as usize]
}

/// The distinct card values in `p`'s hand, ascending. Moves are generated per value so
/// identical cards never produce duplicate moves.
pub(crate) fn distinct_cards(g: &GameState, p: PlayerId) -> Vec<Card> {
    let mut cards = g.players[p as usize].hand.clone();
    cards.sort_unstable();
    cards.dedup();
    cards
}

/// Discard one `card` from `p`'s hand: face up onto the discard pile, or back to its own
/// supply if it is a Wild card (rules-reference sections 3.2 and 6).
///
/// Panics if `p` does not hold `card`.
pub(crate) fn discard(g: &mut GameState, p: PlayerId, card: Card) {
    let ps = &mut g.players[p as usize];
    let i = ps
        .hand
        .iter()
        .position(|&c| c == card)
        .unwrap_or_else(|| panic!("player {p} does not hold {card:?}"));
    ps.hand.remove(i);
    match card {
        Card::WildLocation => g.wild_location_supply += 1,
        Card::WildIndustry => g.wild_industry_supply += 1,
        _ => ps.discard.push(card),
    }
}

/// Deduct `cost` from `p`'s money and record it as spent this round (rules-reference
/// section 6). Panics if `p` cannot pay; legality checks affordability first.
pub(crate) fn pay(g: &mut GameState, p: PlayerId, cost: u16) {
    let ps = &mut g.players[p as usize];
    ps.money = ps
        .money
        .checked_sub(cost)
        .unwrap_or_else(|| panic!("player {p} cannot pay £{cost}"));
    ps.spent += cost;
}
