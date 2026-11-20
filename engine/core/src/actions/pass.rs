//! Pass (rules-reference section 6): discard a card and do nothing else.

use crate::actions::{actor, discard};
use crate::data::Card;
use crate::flow;
use crate::moves::Move;
use crate::state::GameState;

/// One `Pass` per distinct card. `cards` is `distinct_cards` of the actor.
pub(crate) fn legal(cards: &[Card]) -> Vec<Move> {
    cards.iter().map(|&card| Move::Pass { card }).collect()
}

pub(crate) fn apply(g: &mut GameState, card: Card) {
    let p = actor(g);
    discard(g, p, card);
    flow::end_action(g);
}
