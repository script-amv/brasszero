// SPDX-License-Identifier: Apache-2.0
//! Resource sourcing and consumption: coal, iron, beer, tile flips.
//! Rules: docs/rules-reference.md section 8 (8.1 coal, 8.2 iron, 8.3 beer) and section 4.7
//! (flipping and income); RULES-QUESTIONS.md RQ-012, RQ-013, RQ-014, RQ-040.

mod common;

use brass_engine::data::*;
use brass_engine::resources::*;
use brass_engine::state::GameState;
use common::*;

use Industry::{Brewery, Coal, Iron};

fn plan(from_tiles: &[(SlotId, u8)], from_market: u8) -> ResourcePlan {
    let mut from_tiles = from_tiles.to_vec();
    from_tiles.sort();
    ResourcePlan {
        from_tiles,
        from_market,
    }
}

fn coal_at(g: &GameState, l: LocId, n: u8) -> Vec<ResourcePlan> {
    coal_plans(g, CoalTarget::Loc(l), n)
}

/// Sorted copy, so expectations do not depend on the enumeration order.
fn sorted<T: Ord + Clone>(v: &[T]) -> Vec<T> {
    let mut v = v.to_vec();
    v.sort();
    v
}

// ---------------------------------------------------------------- coal (section 8.1)

// 8.1 step 1: take from the closest connected unflipped mine, whoever owns it. A farther own
// mine and an unconnected mine with more coal are not offered.
#[test]
fn coal_takes_closest_connected_mine_even_if_opponents() {
    let mut g = empty_game();
    link(&mut g, "birmingham", "dudley", 2);
    link(&mut g, "dudley", "wolverhampton", 3);
    let opp = place(&mut g, "dudley", 0, 1, Coal, 2, 2); // distance 1, opponent's
    place(&mut g, "wolverhampton", 1, 0, Coal, 2, 3); // distance 2, own
    place(&mut g, "leek", 1, 0, Coal, 4, 5); // not connected
    assert_eq!(coal_at(&g, loc::BIRMINGHAM, 1), vec![plan(&[(opp, 1)], 0)]);
    assert_eq!(coal_at(&g, loc::BIRMINGHAM, 2), vec![plan(&[(opp, 2)], 0)]);
}

// 8.1 step 1, RQ-013, Review Focus 1: equally close mines are the player's choice, but all the
// coal comes from the chosen one until it is empty; splitting across tied mines is illegal.
#[test]
fn coal_equidistant_mines_offer_choice() {
    let mut g = empty_game();
    link(&mut g, "birmingham", "dudley", 1);
    link(&mut g, "birmingham", "coventry", 2);
    let dud = place(&mut g, "dudley", 0, 1, Coal, 2, 2);
    let cov = place(&mut g, "coventry", 1, 2, Coal, 2, 3);

    let one = coal_at(&g, loc::BIRMINGHAM, 1);
    assert_eq!(one.len(), 2);
    assert_eq!(
        sorted(&one),
        sorted(&[plan(&[(dud, 1)], 0), plan(&[(cov, 1)], 0)])
    );

    // Each mine can supply 2 alone: exactly 2 plans, no {dud 1, cov 1} split.
    let two = coal_at(&g, loc::BIRMINGHAM, 2);
    assert_eq!(two.len(), 2);
    assert_eq!(
        sorted(&two),
        sorted(&[plan(&[(dud, 2)], 0), plan(&[(cov, 2)], 0)])
    );

    // Need 3: drain Dudley (2) and take the rest from the other tied mine, or Coventry alone.
    let three = coal_at(&g, loc::BIRMINGHAM, 3);
    assert_eq!(
        sorted(&three),
        sorted(&[plan(&[(dud, 2), (cov, 1)], 0), plan(&[(cov, 3)], 0)])
    );

    // A unique closest mine: no choice.
    g.slots[cov as usize].as_mut().unwrap().resources = 0;
    g.slots[cov as usize].as_mut().unwrap().flipped = true;
    assert_eq!(coal_at(&g, loc::BIRMINGHAM, 1), vec![plan(&[(dud, 1)], 0)]);
}

// RQ-014: a mine in the same location is distance 0 and must be used first, any owner, even
// with no links at all.
#[test]
fn coal_mine_in_same_location_is_distance_zero() {
    let mut g = empty_game();
    let here = place(&mut g, "cannock", 1, 3, Coal, 2, 1);
    assert_eq!(coal_at(&g, loc::CANNOCK, 1), vec![plan(&[(here, 1)], 0)]);

    link(&mut g, "cannock", "wolverhampton", 0);
    let near = place(&mut g, "wolverhampton", 1, 0, Coal, 2, 3);
    assert_eq!(coal_at(&g, loc::CANNOCK, 1), vec![plan(&[(here, 1)], 0)]);
    assert_eq!(
        coal_at(&g, loc::CANNOCK, 2),
        vec![plan(&[(here, 1), (near, 1)], 0)]
    );
}

// RQ-014 for a rail link (8.1: "connected to the coal source after it is placed"): with the
// link built, a mine at either end is distance 0; distance is the minimum over all locations
// the link touches, including the south farm brewery's `extra`.
#[test]
fn coal_link_target_measures_from_either_end() {
    let mut g = empty_game();
    let target = link_id("cannock", "walsall");
    link(&mut g, "cannock", "walsall", 0); // placed before sourcing (Task 7)
    link(&mut g, "walsall", "wolverhampton", 1);
    let far = place(&mut g, "wolverhampton", 1, 1, Coal, 2, 3); // distance 1 via Walsall
    assert_eq!(
        coal_plans(&g, CoalTarget::Link(target), 1),
        vec![plan(&[(far, 1)], 0)]
    );
    let end = place(&mut g, "cannock", 1, 2, Coal, 2, 1); // distance 0
    assert_eq!(
        coal_plans(&g, CoalTarget::Link(target), 2),
        vec![plan(&[(end, 1), (far, 1)], 0)]
    );

    // Kidderminster–Worcester (+ farm-south): a mine at Kidderminster is at distance 0.
    let kw = link_id("kidderminster", "worcester");
    link(&mut g, "kidderminster", "worcester", 0);
    let kid = place(&mut g, "kidderminster", 0, 3, Coal, 2, 2);
    assert_eq!(
        coal_plans(&g, CoalTarget::Link(kw), 1),
        vec![plan(&[(kid, 1)], 0)]
    );
}

// 8.1 step 2: the market only once no connected mine has coal; unconnected mines do not count;
// mine and market coal combine for one need.
#[test]
fn coal_market_only_when_no_connected_mine_has_coal() {
    let mut g = empty_game();
    link(&mut g, "birmingham", "oxford", 0);
    link(&mut g, "birmingham", "dudley", 1);
    let dud = place(&mut g, "dudley", 0, 1, Coal, 2, 1);
    place(&mut g, "leek", 1, 0, Coal, 4, 5); // not connected: never blocks the market
    assert_eq!(coal_at(&g, loc::BIRMINGHAM, 1), vec![plan(&[(dud, 1)], 0)]);
    assert_eq!(coal_at(&g, loc::BIRMINGHAM, 2), vec![plan(&[(dud, 1)], 1)]);

    let t = g.slots[dud as usize].as_mut().unwrap();
    t.resources = 0;
    t.flipped = true;
    assert_eq!(coal_at(&g, loc::BIRMINGHAM, 2), vec![plan(&[], 2)]);
}

// 8.1 step 2-3: market coal requires the consuming location to be connected to a merchant;
// otherwise there is no legal plan.
#[test]
fn coal_market_requires_merchant_connection() {
    let mut g = empty_game();
    assert!(coal_at(&g, loc::BIRMINGHAM, 1).is_empty());
    // Mine coal still runs short without a merchant: no plan at all.
    link(&mut g, "birmingham", "dudley", 1);
    place(&mut g, "dudley", 0, 1, Coal, 2, 1);
    assert!(coal_at(&g, loc::BIRMINGHAM, 2).is_empty());
    // Connection through an opponent's link to a merchant location opens the market.
    link(&mut g, "birmingham", "oxford", 3);
    assert_eq!(coal_at(&g, loc::BIRMINGHAM, 2).len(), 1);
    assert_eq!(coal_at(&g, loc::BIRMINGHAM, 2)[0].from_market, 1);
}

// 8.1 step 1: move outward ring by ring once the closest mine runs out, choosing again among
// ties in the next ring.
#[test]
fn coal_spans_rings_when_closest_runs_out() {
    let mut g = empty_game();
    link(&mut g, "birmingham", "dudley", 0);
    link(&mut g, "dudley", "wolverhampton", 0);
    let dud = place(&mut g, "dudley", 0, 1, Coal, 2, 1);
    let wol = place(&mut g, "wolverhampton", 1, 2, Coal, 2, 1);
    assert_eq!(
        coal_at(&g, loc::BIRMINGHAM, 2),
        vec![plan(&[(dud, 1), (wol, 1)], 0)]
    );

    link(&mut g, "dudley", "kidderminster", 0);
    let kid = place(&mut g, "kidderminster", 0, 3, Coal, 2, 2);
    assert_eq!(
        sorted(&coal_at(&g, loc::BIRMINGHAM, 2)),
        sorted(&[
            plan(&[(dud, 1), (wol, 1)], 0),
            plan(&[(dud, 1), (kid, 1)], 0)
        ])
    );
}

// 8.1 step 2: "Only if no connected mine has coal" means no connected mine at ANY distance.
// With the market reachable, a farther ring that still has coal blocks it; once every
// connected mine is drained, the market supplies exactly the shortfall.
#[test]
fn coal_no_market_while_farther_connected_mine_has_coal() {
    let mut g = empty_game();
    link(&mut g, "birmingham", "oxford", 1); // market reachable
    link(&mut g, "birmingham", "dudley", 0);
    link(&mut g, "dudley", "wolverhampton", 0);
    link(&mut g, "dudley", "kidderminster", 0);
    let dud = place(&mut g, "dudley", 0, 1, Coal, 2, 1); // ring 1
    let wol = place(&mut g, "wolverhampton", 1, 2, Coal, 2, 1); // ring 2
    let kid = place(&mut g, "kidderminster", 0, 3, Coal, 2, 2); // ring 2
    for n in 1..=4 {
        let plans = coal_at(&g, loc::BIRMINGHAM, n);
        assert!(!plans.is_empty(), "need {n}");
        assert!(
            plans.iter().all(|p| p.from_market == 0),
            "need {n}: market offered while ring-2 coal suffices: {plans:?}"
        );
    }
    // Rings 1 and 2 hold 4 coal; need 5: drain every mine, buy exactly 1.
    let plans = coal_at(&g, loc::BIRMINGHAM, 5);
    assert_eq!(plans, vec![plan(&[(dud, 1), (wol, 1), (kid, 2)], 1)]);
    assert!(
        plans
            .iter()
            .all(|p| p.from_market == 1 && p.from_tiles.len() == 3)
    );
}

#[test]
fn zero_needed_is_one_empty_plan() {
    let g = empty_game();
    assert_eq!(coal_at(&g, loc::BIRMINGHAM, 0), vec![plan(&[], 0)]);
    assert_eq!(iron_plans(&g, 0), vec![plan(&[], 0)]);
}

// ---------------------------------------------------------------- iron (section 8.2)

// 8.2: no connection is ever needed, for works or market.
#[test]
fn iron_needs_no_connection() {
    let mut g = empty_game();
    assert_eq!(iron_plans(&g, 2), vec![plan(&[], 2)]);
    let w = place(&mut g, "dudley", 1, 1, Iron, 1, 4);
    assert_eq!(iron_plans(&g, 1), vec![plan(&[(w, 1)], 0)]);
}

// 8.2: any works, any split; the market only once every works is empty.
#[test]
fn iron_split_across_works() {
    let mut g = empty_game();
    let a = place(&mut g, "dudley", 1, 1, Iron, 1, 1);
    let b = place(&mut g, "coalbrookdale", 1, 2, Iron, 2, 2);
    assert_eq!(
        sorted(&iron_plans(&g, 1)),
        sorted(&[plan(&[(a, 1)], 0), plan(&[(b, 1)], 0)])
    );
    assert_eq!(
        sorted(&iron_plans(&g, 2)),
        sorted(&[plan(&[(a, 1), (b, 1)], 0), plan(&[(b, 2)], 0)])
    );
    assert_eq!(iron_plans(&g, 3), vec![plan(&[(a, 1), (b, 2)], 0)]);
    assert_eq!(iron_plans(&g, 4), vec![plan(&[(a, 1), (b, 2)], 1)]);
}

// Plans are deduplicated and in a deterministic order.
#[test]
fn plans_are_sorted_and_unique() {
    let mut g = empty_game();
    link(&mut g, "birmingham", "dudley", 1);
    link(&mut g, "birmingham", "coventry", 2);
    link(&mut g, "birmingham", "oxford", 2);
    place(&mut g, "dudley", 0, 1, Coal, 2, 1);
    place(&mut g, "coventry", 1, 2, Coal, 2, 1);
    place(&mut g, "tamworth", 0, 2, Coal, 2, 1);
    link(&mut g, "birmingham", "tamworth", 2);
    place(&mut g, "dudley", 1, 1, Iron, 1, 2);
    place(&mut g, "coalbrookdale", 1, 2, Iron, 2, 2);
    for v in [coal_at(&g, loc::BIRMINGHAM, 2), iron_plans(&g, 2)] {
        let mut s = v.clone();
        s.sort();
        s.dedup();
        assert_eq!(v, s);
    }
    // Three tied mines with 1 coal each, need 2: every pair, each listed once.
    assert_eq!(coal_at(&g, loc::BIRMINGHAM, 2).len(), 3);
}

// ---------------------------------------------------------------- consumption and markets

// 8.1 step 2 and 2.4: market coal is bought cheapest first, £8 each once the market is empty.
// `consume` returns the cost and does not touch the player's money.
#[test]
fn consume_market_coal_costs_cheapest_first() {
    let mut g = empty_game();
    assert_eq!(g.coal_market, 13); // one £1 space empty
    let money = g.players.clone();
    assert_eq!(consume(&mut g, &plan(&[], 2), Coal), 1 + 2);
    assert_eq!(g.coal_market, 11);
    assert_eq!(g.players, money);
    g.coal_market = 1;
    assert_eq!(consume(&mut g, &plan(&[], 3), Coal), 7 + 8 + 8);
    assert_eq!(g.coal_market, 0);
}

// 8.2 and 2.4: iron market cheapest first, £6 each once empty.
#[test]
fn consume_market_iron_costs_cheapest_first() {
    let mut g = empty_game();
    assert_eq!(g.iron_market, 8);
    assert_eq!(consume(&mut g, &plan(&[], 1), Iron), 2);
    g.iron_market = 0;
    assert_eq!(consume(&mut g, &plan(&[], 2), Iron), 12);
}

// 4.7: a mine or works flips only when its last cube leaves.
#[test]
fn consume_flips_only_when_emptied() {
    let mut g = empty_game();
    let w = place(&mut g, "dudley", 1, 1, Iron, 1, 4);
    assert_eq!(consume(&mut g, &plan(&[(w, 3)], 0), Iron), 0);
    let t = g.slots[w as usize].as_ref().unwrap();
    assert_eq!((t.resources, t.flipped), (1, false));
    assert_eq!(g.players[1].income_space, START_INCOME_SPACE);
    consume(&mut g, &plan(&[(w, 1)], 0), Iron);
    let t = g.slots[w as usize].as_ref().unwrap();
    assert_eq!((t.resources, t.flipped), (0, true));
    // Iron Works I: income +3 spaces.
    assert_eq!(g.players[1].income_space, START_INCOME_SPACE + 3);
}

// 4.7, 8.1 "the owner gains the flip", Review Focus 5: emptying an opponent's tile flips it and
// the income goes to the tile's owner, not the player consuming. Same for beer.
#[test]
fn flip_on_opponent_consumption_pays_owner() {
    let mut g = empty_game();
    link(&mut g, "birmingham", "dudley", 0);
    let mine = place(&mut g, "dudley", 0, 1, Coal, 2, 1); // Coal Mine II: income +7
    let before = g.players.clone();
    let plans = coal_at(&g, loc::BIRMINGHAM, 1);
    assert_eq!(consume(&mut g, &plans[0], Coal), 0);
    let t = g.slots[mine as usize].as_ref().unwrap();
    assert!(t.flipped);
    assert_eq!(t.resources, 0);
    assert_eq!(g.players[1].income_space, before[1].income_space + 7);
    assert_eq!(g.players[0], before[0]);
    assert_eq!(g.players[1].money, before[1].money);

    link(&mut g, "birmingham", "walsall", 0);
    let brew = place(&mut g, "walsall", 1, 2, Brewery, 1, 1); // Brewery I: income +4
    let srcs = beer_plans(&g, 0, loc::BIRMINGHAM, 1, None);
    assert_eq!(srcs, vec![vec![BeerSource::Brewery(brew)]]);
    consume_beer(&mut g, &srcs[0]);
    assert!(g.slots[brew as usize].as_ref().unwrap().flipped);
    assert_eq!(g.players[2].income_space, before[2].income_space + 4);
    assert_eq!(g.players[0], before[0]);
}

// 4.7, RQ-040: income gained on a flip is capped at space 99 (level 30).
#[test]
fn flip_income_capped_at_space_99() {
    let mut g = empty_game();
    let mine = place(&mut g, "dudley", 0, 1, Coal, 2, 0); // income +7
    g.players[1].income_space = 95;
    flip(&mut g, mine);
    assert!(g.slots[mine as usize].as_ref().unwrap().flipped);
    assert_eq!(g.players[1].income_space, MAX_INCOME_SPACE);
}

// ---------------------------------------------------------------- beer (section 8.3)

// 8.3 source 2: an opponent's brewery only if connected to the location needing the beer.
#[test]
fn beer_opponent_brewery_requires_connection() {
    let mut g = empty_game();
    let b = place(&mut g, "walsall", 1, 1, Brewery, 1, 1);
    assert!(beer_plans(&g, 0, loc::BIRMINGHAM, 1, None).is_empty());
    link(&mut g, "birmingham", "walsall", 2); // anyone's link connects
    assert_eq!(
        beer_plans(&g, 0, loc::BIRMINGHAM, 1, None),
        vec![vec![BeerSource::Brewery(b)]]
    );
}

// 8.3 source 1: own breweries anywhere, no connection; a 2-barrel brewery can supply both.
#[test]
fn beer_own_brewery_anywhere() {
    let mut g = empty_game();
    let b = place(&mut g, "uttoxeter", 0, 0, Brewery, 2, 2);
    assert_eq!(
        beer_plans(&g, 0, loc::BIRMINGHAM, 1, None),
        vec![vec![BeerSource::Brewery(b)]]
    );
    assert_eq!(
        beer_plans(&g, 0, loc::BIRMINGHAM, 2, None),
        vec![vec![BeerSource::Brewery(b), BeerSource::Brewery(b)]]
    );
    assert!(beer_plans(&g, 0, loc::BIRMINGHAM, 3, None).is_empty());
    // Not usable by another player without a connection.
    assert!(beer_plans(&g, 1, loc::BIRMINGHAM, 1, None).is_empty());
}

// 8.3: sources combine in any way, no priority order (own is not forced first).
#[test]
fn beer_has_no_priority_order() {
    let mut g = empty_game();
    let own = place(&mut g, "stafford", 0, 0, Brewery, 1, 1);
    link(&mut g, "birmingham", "walsall", 3);
    let opp = place(&mut g, "walsall", 1, 1, Brewery, 1, 1);
    use BeerSource::Brewery as B;
    assert_eq!(
        sorted(&beer_plans(&g, 0, loc::BIRMINGHAM, 1, None)),
        sorted(&[vec![B(own)], vec![B(opp)]])
    );
    assert_eq!(
        beer_plans(&g, 0, loc::BIRMINGHAM, 2, None),
        vec![sorted(&[B(own), B(opp)])]
    );
}

// 8.3 source 3, 9.4: the merchant barrel only when that merchant slot is given and still has
// beer, and at most once per sale.
#[test]
fn beer_merchant_barrel_only_when_given_and_present() {
    let mut g = empty_game();
    let own = place(&mut g, "stafford", 0, 0, Brewery, 1, 1);
    use BeerSource::{Brewery as B, Merchant as M};
    g.merchant_beer[2] = true;
    g.merchant_beer[3] = false;
    assert_eq!(
        sorted(&beer_plans(&g, 0, loc::BIRMINGHAM, 1, Some(2))),
        sorted(&[vec![B(own)], vec![M(2)]])
    );
    assert_eq!(
        beer_plans(&g, 0, loc::BIRMINGHAM, 2, Some(2)),
        vec![sorted(&[B(own), M(2)])]
    );
    assert_eq!(
        beer_plans(&g, 0, loc::BIRMINGHAM, 1, Some(3)),
        vec![vec![B(own)]]
    );
    assert_eq!(
        beer_plans(&g, 0, loc::BIRMINGHAM, 1, None),
        vec![vec![B(own)]]
    );

    // Only the merchant barrel, need 2: never used twice.
    g.slots[own as usize] = None;
    assert!(beer_plans(&g, 0, loc::BIRMINGHAM, 2, Some(2)).is_empty());

    consume_beer(&mut g, &[M(2)]);
    assert!(!g.merchant_beer[2]);
}

// 8.3 source 2 and 7 ("connected"): an opponent's brewery in the location needing the beer is
// connected to it (distance 0), even with no links; one elsewhere and unlinked is not.
#[test]
fn opponent_brewery_in_same_location_is_connected() {
    let mut g = empty_game();
    let here = place(&mut g, "walsall", 1, 1, Brewery, 1, 1);
    place(&mut g, "stafford", 0, 2, Brewery, 1, 1); // opponent's, not connected
    assert_eq!(
        beer_plans(&g, 0, loc::WALSALL, 1, None),
        vec![vec![BeerSource::Brewery(here)]]
    );
}

// RQ-012: a tile that needs no beer gets exactly the empty plan, never the merchant barrel.
#[test]
fn beer_zero_needed_offers_no_merchant_beer() {
    let mut g = empty_game();
    g.merchant_beer[0] = true;
    assert_eq!(
        beer_plans(&g, 0, loc::BIRMINGHAM, 0, Some(0)),
        vec![Vec::<BeerSource>::new()]
    );
}
