//! Develop (rules-reference section 9.3): remove 1 or 2 tiles from the mat, 1 iron each.
//!
//! The first tile is `Move::Develop` (discards the card, enters `Phase::DevelopMore`); the
//! optional second is `Move::DevelopSecond`, or `Move::EndAction` to stop after one.

use crate::actions::{actor, discard, pay};
use crate::data::{Card, Industry, PlayerId, tile};
use crate::flow;
use crate::moves::{Move, Phase};
use crate::network::iron_buy_cost;
use crate::resources::{ResourcePlan, consume, iron_plans};
use crate::state::GameState;

/// Industries the actor may develop now: the lowest remaining tile exists and is not a
/// lightbulb. A lightbulb at the bottom blocks the whole industry, since Develop always
/// takes the lowest-level tile.
pub(crate) fn developable(g: &GameState) -> Vec<Industry> {
    let ps = &g.players[actor(g) as usize];
    Industry::ALL
        .into_iter()
        .filter(|&ind| {
            ps.lowest_level(ind)
                .is_some_and(|lvl| tile(ind, lvl).developable)
        })
        .collect()
}

/// Ways to source the 1 iron that the actor can pay for right now.
fn affordable_iron(g: &GameState) -> Vec<ResourcePlan> {
    let money = g.players[actor(g) as usize].money;
    iron_plans(g, 1)
        .into_iter()
        .filter(|plan| iron_buy_cost(g.iron_market, plan.from_market).0 <= money)
        .collect()
}

/// `(industry, iron)` pairs for one tile, industry-major.
fn options(g: &GameState) -> Vec<(Industry, ResourcePlan)> {
    let inds = developable(g);
    if inds.is_empty() {
        return Vec::new();
    }
    let irons = affordable_iron(g);
    inds.into_iter()
        .flat_map(|ind| irons.iter().map(move |iron| (ind, iron.clone())))
        .collect()
}

/// `Phase::ChooseAction`: card × industry × iron plan. `cards` is `distinct_cards` of the actor.
pub(crate) fn legal(g: &GameState, cards: &[Card]) -> Vec<Move> {
    let opts = options(g);
    if opts.is_empty() {
        return Vec::new();
    }
    cards
        .iter()
        .flat_map(|&card| {
            opts.iter().map(move |(industry, iron)| Move::Develop {
                card,
                industry: *industry,
                iron: iron.clone(),
            })
        })
        .collect()
}

/// `Phase::DevelopMore`: the optional second tile (`EndAction` is added by `legal.rs`).
pub(crate) fn legal_second(g: &GameState) -> Vec<Move> {
    options(g)
        .into_iter()
        .map(|(industry, iron)| Move::DevelopSecond { industry, iron })
        .collect()
}

/// Remove the lowest `industry` tile from `p`'s mat. Shared with the Gloucester bonus.
pub(crate) fn remove_lowest(g: &mut GameState, p: PlayerId, industry: Industry) {
    let ps = &mut g.players[p as usize];
    let lvl = ps.lowest_level(industry).expect("develop legality checked");
    ps.mat_remaining[industry as usize][lvl as usize - 1] -= 1;
}

/// Remove the lowest `industry` tile from the actor's mat, consuming and paying for 1 iron.
fn develop_one(g: &mut GameState, industry: Industry, iron: &ResourcePlan) {
    let p = actor(g);
    let cost = consume(g, iron, Industry::Iron);
    pay(g, p, cost);
    remove_lowest(g, p, industry);
}

pub(crate) fn apply(g: &mut GameState, card: Card, industry: Industry, iron: &ResourcePlan) {
    let p = actor(g);
    discard(g, p, card);
    develop_one(g, industry, iron);
    g.phase = Phase::DevelopMore;
}

pub(crate) fn apply_second(g: &mut GameState, industry: Industry, iron: &ResourcePlan) {
    develop_one(g, industry, iron);
    g.phase = Phase::ChooseAction;
    flow::end_action(g);
}
