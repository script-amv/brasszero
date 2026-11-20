//! Scout (rules-reference section 9.6, RQ-003, RQ-010, RQ-026).

use crate::actions::{actor, discard};
use crate::data::Card;
use crate::flow;
use crate::moves::Move;
use crate::state::GameState;

fn is_wild(c: Card) -> bool {
    matches!(c, Card::WildLocation | Card::WildIndustry)
}

/// One `Scout` per distinct sorted 3-card multiset of the hand. None if the hand holds any
/// Wild card or fewer than 3 cards.
pub(crate) fn legal(g: &GameState) -> Vec<Move> {
    let mut hand = g.players[actor(g) as usize].hand.clone();
    if hand.len() < 3 || hand.iter().any(|&c| is_wild(c)) {
        return Vec::new();
    }
    hand.sort_unstable();
    let mut out = Vec::new();
    for i in 0..hand.len() {
        if i > 0 && hand[i] == hand[i - 1] {
            continue;
        }
        for j in i + 1..hand.len() {
            if j > i + 1 && hand[j] == hand[j - 1] {
                continue;
            }
            for k in j + 1..hand.len() {
                if k > j + 1 && hand[k] == hand[k - 1] {
                    continue;
                }
                out.push(Move::Scout {
                    cards: [hand[i], hand[j], hand[k]],
                });
            }
        }
    }
    out
}

/// Discard the 3 cards, take 1 Wild Location and 1 Wild Industry.
pub(crate) fn apply(g: &mut GameState, cards: &[Card; 3]) {
    let p = actor(g);
    for &c in cards {
        discard(g, p, c);
    }
    // RQ-010: at 4 players a legal Scout always finds both supplies non-empty.
    debug_assert!(
        g.wild_location_supply > 0 && g.wild_industry_supply > 0,
        "wild supply empty on a legal Scout"
    );
    g.wild_location_supply -= 1;
    g.wild_industry_supply -= 1;
    let hand = &mut g.players[p as usize].hand;
    hand.push(Card::WildLocation);
    hand.push(Card::WildIndustry);
    flow::end_action(g);
}
