//! Loan (rules-reference section 9.5, RQ-002).

use crate::actions::{actor, discard};
use crate::data::{Card, LOAN_AMOUNT, loan_space};
use crate::flow;
use crate::moves::Move;
use crate::state::GameState;

/// One `Loan` per distinct card, unless the loan would take income below level −10. `cards`
/// is `distinct_cards` of the actor.
pub(crate) fn legal(g: &GameState, cards: &[Card]) -> Vec<Move> {
    let p = actor(g);
    if loan_space(g.players[p as usize].income_space).is_none() {
        return Vec::new();
    }
    cards.iter().map(|&card| Move::Loan { card }).collect()
}

/// Discard `card`, take £30 (income, not spending), drop 3 income levels.
pub(crate) fn apply(g: &mut GameState, card: Card) {
    let p = actor(g);
    discard(g, p, card);
    let ps = &mut g.players[p as usize];
    ps.income_space = loan_space(ps.income_space).expect("loan legality checked");
    ps.money += LOAN_AMOUNT;
    flow::end_action(g);
}
