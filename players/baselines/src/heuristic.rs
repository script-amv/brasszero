//! One-ply position value used by the greedy bot: VP banked, VP the era end would add now,
//! and rough worth of future income and cash.

use brass_engine::data::{Era, PlayerId, income_level};
use brass_engine::flow::era_end_vp;
use brass_engine::state::GameState;

/// VP per income level per remaining payment.
pub const INCOME_WEIGHT: f64 = 0.5;
/// VP per £.
pub const MONEY_WEIGHT: f64 = 0.1;

/// Income payments still to come, counting the current round's: Canal 16 - round, Rail 8 - round.
pub fn payments_left(g: &GameState) -> u8 {
    match g.era {
        Era::Canal => 16u8.saturating_sub(g.round),
        Era::Rail => 8u8.saturating_sub(g.round),
    }
}

/// vp + era_end_vp + INCOME_WEIGHT * income_level * payments_left + MONEY_WEIGHT * money
pub fn value(g: &GameState, p: PlayerId) -> f64 {
    let ps = &g.players[p as usize];
    let era_end = era_end_vp(g)[p as usize];
    ps.vp as f64
        + era_end as f64
        + INCOME_WEIGHT * income_level(ps.income_space) as f64 * payments_left(g) as f64
        + MONEY_WEIGHT * ps.money as f64
}
