//! Network (rules-reference section 9.2): place a canal link (Canal Era), or one or two rail
//! links (Rail Era).
//!
//! `Move::Canal` ends the action. `Move::Rail` places the first rail link and enters
//! `Phase::SecondRail`, where `Move::SecondRail` places the optional second link, or
//! `Move::EndAction` stops after one.

use crate::actions::{actor, discard, pay};
use crate::data::*;
use crate::flow;
use crate::moves::{Move, Phase};
use crate::network::{
    Adjacency, LocSet, coal_buy_cost, has_board_presence, link_locs, player_network,
};
use crate::resources::{
    BeerSource, CoalTarget, ResourcePlan, beer_plans_in, coal_plans_in, consume, consume_beer,
};
use crate::state::GameState;

/// Unbuilt links `p` may place now: of the era's kind, adjacent to a location in `p`'s network,
/// or any link when `p` has nothing on the board (section 9.2, RQ-043).
fn placeable(g: &GameState, p: PlayerId) -> Vec<LinkId> {
    let network: LocSet = player_network(g, p);
    let presence = has_board_presence(g, p);
    (0..LINKS.len() as LinkId)
        .filter(|&id| {
            let l = &LINKS[id as usize];
            let era_ok = match g.era {
                Era::Canal => l.canal,
                Era::Rail => l.rail,
            };
            era_ok
                && g.links[id as usize].is_none()
                && (!presence || link_locs(l).any(|loc| network & (1 << loc) != 0))
        })
        .collect()
}

/// Link tiles `p` still has (RQ-043): 14 minus the links on the board. Links return to their
/// owner at era end: `flow` clears `g.links` after scoring them (section 6.2 step 1).
fn tiles_left(g: &GameState, p: PlayerId) -> usize {
    LINK_TILES - g.links.iter().filter(|o| **o == Some(p)).count()
}

/// Coal plans for one rail link placed by the actor, whose total cost (`base` plus the market
/// coal) is within the actor's money. Coal is sourced after placement, so `link` counts as
/// built (`adj` is the current board's adjacency).
fn affordable_coal(g: &GameState, adj: &Adjacency, link: LinkId, base: u16) -> Vec<ResourcePlan> {
    let money = g.players[actor(g) as usize].money;
    coal_plans_in(g, &adj.with_link(link), CoalTarget::Link(link), 1)
        .into_iter()
        .filter(|c| base + coal_buy_cost(g.coal_market, c.from_market).0 <= money)
        .collect()
}

/// `Phase::ChooseAction`: Canal (Canal Era) or Rail (Rail Era), per distinct card and link.
/// `cards` is `distinct_cards` of the actor.
pub(crate) fn legal(g: &GameState, cards: &[Card]) -> Vec<Move> {
    moves(g, cards, None)
}

/// The Canal / Rail moves of `legal` that place `link` (for `apply`'s check).
pub(crate) fn legal_for(g: &GameState, cards: &[Card], link: LinkId) -> Vec<Move> {
    moves(g, cards, Some(link))
}

fn moves(g: &GameState, cards: &[Card], only: Option<LinkId>) -> Vec<Move> {
    let p = actor(g);
    if tiles_left(g, p) == 0 {
        return Vec::new();
    }
    let mut links = placeable(g, p);
    if let Some(only) = only {
        links.retain(|&l| l == only);
    }
    if links.is_empty() {
        return Vec::new();
    }
    match g.era {
        Era::Canal => {
            if g.players[p as usize].money < CANAL_COST {
                return Vec::new();
            }
            cards
                .iter()
                .flat_map(|&card| links.iter().map(move |&link| Move::Canal { card, link }))
                .collect()
        }
        Era::Rail => {
            let adj = Adjacency::of(g);
            let mut out = Vec::new();
            for &link in &links {
                let plans = affordable_coal(g, &adj, link, RAIL_COST);
                for &card in cards {
                    out.extend(plans.iter().map(|coal| Move::Rail {
                        card,
                        link,
                        coal: coal.clone(),
                    }));
                }
            }
            out
        }
    }
}

/// `Phase::SecondRail`: the optional second rail link (`EndAction` is added by `legal.rs`).
/// The first link is already on the board, so it extends the network and connects the second
/// link's coal and beer; the second link never supplies the first one's coal (already taken).
pub(crate) fn legal_second(g: &GameState) -> Vec<Move> {
    let p = actor(g);
    if tiles_left(g, p) == 0 {
        return Vec::new();
    }
    let adj = Adjacency::of(g);
    let mut out = Vec::new();
    for link in placeable(g, p) {
        let coals = affordable_coal(g, &adj, link, DOUBLE_RAIL_COST - RAIL_COST);
        if coals.is_empty() {
            continue;
        }
        // Brewery beer only, never merchant beer. Both ends and any farm are joined by the
        // placed link, so any one endpoint tells which opponent breweries are connected.
        let beers = beer_plans_in(g, &adj.with_link(link), p, LINKS[link as usize].a, 1, None);
        for coal in &coals {
            for plan in &beers {
                let BeerSource::Brewery(beer) = plan[0] else {
                    unreachable!("no merchant beer was requested");
                };
                out.push(Move::SecondRail {
                    link,
                    coal: coal.clone(),
                    beer,
                });
            }
        }
    }
    out
}

pub(crate) fn apply_canal(g: &mut GameState, card: Card, link: LinkId) {
    let p = actor(g);
    discard(g, p, card);
    g.links[link as usize] = Some(p);
    pay(g, p, CANAL_COST);
    flow::end_action(g);
}

/// Place `link` first, then take its coal (section 9.2), and pay `base` plus any market coal.
fn place_rail(g: &mut GameState, link: LinkId, coal: &ResourcePlan, base: u16) {
    let p = actor(g);
    g.links[link as usize] = Some(p);
    let cost = base + consume(g, coal, Industry::Coal);
    pay(g, p, cost);
}

pub(crate) fn apply_rail(g: &mut GameState, card: Card, link: LinkId, coal: &ResourcePlan) {
    discard(g, actor(g), card);
    place_rail(g, link, coal, RAIL_COST);
    g.phase = Phase::SecondRail;
}

pub(crate) fn apply_second(g: &mut GameState, link: LinkId, coal: &ResourcePlan, beer: SlotId) {
    place_rail(g, link, coal, DOUBLE_RAIL_COST - RAIL_COST);
    consume_beer(g, &[BeerSource::Brewery(beer)]);
    g.phase = Phase::ChooseAction;
    flow::end_action(g);
}
