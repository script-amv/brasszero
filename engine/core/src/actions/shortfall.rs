//! Shortfall (rules-reference section 6.1 step 3, RQ-004): a player who cannot pay negative
//! income removes own Industry tiles, one at a time and of their choice, for half their cost
//! rounded down, until the debt is covered (keeping any excess). With no tile left that
//! raises money, they lose 1 VP per £1 still short, never below 0.

use crate::data::*;
use crate::flow;
use crate::moves::{Move, Phase};
use crate::state::GameState;

/// Money raised by removing the tile in `slot`: half its cost, rounded down.
fn value(g: &GameState, slot: usize) -> u16 {
    let t = g.slots[slot].as_ref().expect("tile in slot");
    tile(t.industry, t.level).cost_money as u16 / 2
}

/// `p`'s tiles (flipped or not) that raise money when removed, ascending. £0 tiles (Pottery
/// II and IV) are never required, so they are not offered.
pub(crate) fn removable(g: &GameState, p: PlayerId) -> Vec<SlotId> {
    (0..g.slots.len())
        .filter(|&s| g.slots[s].as_ref().is_some_and(|t| t.owner == p) && value(g, s) > 0)
        .map(|s| s as SlotId)
        .collect()
}

/// Lose 1 VP per £1 still owed, never below 0.
pub(crate) fn lose_vp(g: &mut GameState, p: PlayerId, owed: u16) {
    let ps = &mut g.players[p as usize];
    ps.vp = ps.vp.saturating_sub(owed);
}

fn phase(g: &GameState) -> (PlayerId, u16) {
    match g.phase {
        Phase::Shortfall { player, owed } => (player, owed),
        _ => unreachable!("not in Phase::Shortfall"),
    }
}

/// `Phase::Shortfall`: one `ShortfallRemove` per removable tile.
pub(crate) fn legal(g: &GameState) -> Vec<Move> {
    let (p, _) = phase(g);
    removable(g, p)
        .into_iter()
        .map(|slot| Move::ShortfallRemove { slot })
        .collect()
}

/// Remove the tile (its cubes go with it; income is unchanged). Once the debt is covered,
/// or no removable tile is left, settle and resume the round's income.
pub(crate) fn apply(g: &mut GameState, slot: SlotId) {
    let (p, owed) = phase(g);
    let raised = value(g, slot as usize);
    g.slots[slot as usize] = None;
    if raised >= owed {
        g.players[p as usize].money += raised - owed;
    } else if removable(g, p).is_empty() {
        lose_vp(g, p, owed - raised);
    } else {
        g.phase = Phase::Shortfall {
            player: p,
            owed: owed - raised,
        };
        return;
    }
    flow::resume_income(g, p);
}
