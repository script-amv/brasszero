// SPDX-License-Identifier: Apache-2.0
//! The Build action.
//! Rules: docs/rules-reference.md section 9.1 (cards, slot choice, era limits, no-tiles
//! exception, overbuilding, selling cubes to the markets), section 3.2 (wild cards), section 4
//! (tiles), section 7 (network); RULES-QUESTIONS.md RQ-006, RQ-007, RQ-017, RQ-018, RQ-024,
//! RQ-025, RQ-033, RQ-034, RQ-038, RQ-042.

mod common;

use brass_engine::data::*;
use brass_engine::moves::{Move, RuleError};
use brass_engine::resources::ResourcePlan;
use brass_engine::state::GameState;
use common::*;

use Industry::{Brewery, Coal, Cotton, Iron, Manufacturer, Pottery};

const BIRM: Card = Card::Location(loc::BIRMINGHAM);
const DUDLEY: Card = Card::Location(loc::DUDLEY);
const COAL_CARD: Card = Card::Industry(IndustryCard::Coal);
const BREWERY_CARD: Card = Card::Industry(IndustryCard::Brewery);

/// A fresh game whose acting player holds `cards`, has `money` and 2 actions left.
fn setup(cards: &[Card], money: u16) -> (GameState, PlayerId) {
    let mut g = empty_game();
    g.actions_left = 2;
    let p = g.current_player().expect("someone is to act");
    g.players[p as usize].hand = cards.to_vec();
    g.players[p as usize].money = money;
    (g, p)
}

/// An opponent of `p`.
fn opp(p: PlayerId) -> PlayerId {
    (p + 1) % 4
}

/// The `i`-th slot (printed order) of the location with kebab-case id `l`.
fn sl(l: &str, i: usize) -> SlotId {
    let l = loc_id(l).expect("location");
    SLOTS
        .iter()
        .enumerate()
        .filter(|(_, s)| s.loc == l)
        .nth(i)
        .expect("slot")
        .0 as SlotId
}

/// Remove every `ind` tile below `level` from `p`'s mat (as if developed away).
fn develop_below(g: &mut GameState, p: PlayerId, ind: Industry, level: u8) {
    for l in 1..level {
        g.players[p as usize].mat_remaining[ind as usize][l as usize - 1] = 0;
    }
}

/// The distinct `(card, industry, slot)` triples of the legal Build moves.
fn targets(g: &GameState) -> Vec<(Card, Industry, SlotId)> {
    let mut v: Vec<_> = g
        .legal_moves()
        .into_iter()
        .filter_map(|m| match m {
            Move::Build {
                card,
                industry,
                slot,
                ..
            } => Some((card, industry, slot)),
            _ => None,
        })
        .collect();
    v.dedup();
    v
}

fn offered(g: &GameState, card: Card, ind: Industry, slot: SlotId) -> bool {
    targets(g).contains(&(card, ind, slot))
}

/// Slots offered for `card` and `ind`, ascending.
fn slots_for(g: &GameState, card: Card, ind: Industry) -> Vec<SlotId> {
    targets(g)
        .into_iter()
        .filter(|&(c, i, _)| c == card && i == ind)
        .map(|(_, _, s)| s)
        .collect()
}

/// The first legal Build of `ind` in `slot` with `card`.
fn build_move(g: &GameState, card: Card, ind: Industry, slot: SlotId) -> Move {
    g.legal_moves()
        .into_iter()
        .find(|m| {
            matches!(m, Move::Build { card: c, industry: i, slot: s, .. }
                if *c == card && *i == ind && *s == slot)
        })
        .unwrap_or_else(|| panic!("no Build {card:?} {ind:?} in slot {slot}"))
}

fn market(n: u8) -> ResourcePlan {
    ResourcePlan {
        from_tiles: vec![],
        from_market: n,
    }
}

// ---------------------------------------------------------------- cards (9.1 step 1)

// 9.1 step 1: a Location card builds at that location even outside the network. Step 3: coal
// still needs a connection from the build location, so the Manufacturer I (1 coal) is not
// offered at an unconnected Birmingham, while Cotton I (no coal) is.
#[test]
fn build_location_card_outside_network() {
    let (mut g, p) = setup(&[BIRM], 50);
    place(&mut g, "leek", 0, p, Cotton, 1, 0); // presence far from Birmingham
    let cotton = sl("birmingham", 0); // the only slot showing cotton ([Ct, Mf])
    assert_eq!(slots_for(&g, BIRM, Cotton), vec![cotton]);
    assert!(slots_for(&g, BIRM, Manufacturer).is_empty());
    g.apply(&build_move(&g, BIRM, Cotton, cotton)).unwrap();
    let t = g.slots[cotton as usize].as_ref().expect("tile built");
    assert_eq!((t.owner, t.industry, t.level), (p, Cotton, 1));
}

// 9.1 step 1, section 7: an Industry card builds only in the player's network (here Birmingham
// and Dudley through the player's own link). An opponent's link does not extend it.
#[test]
fn build_industry_card_requires_network() {
    let (mut g, p) = setup(&[COAL_CARD], 50);
    link(&mut g, "birmingham", "dudley", p);
    link(&mut g, "cannock", "walsall", opp(p));
    let slots = slots_for(&g, COAL_CARD, Coal);
    assert_eq!(slots, vec![sl("dudley", 0)]);
    for (_, _, s) in targets(&g) {
        let l = SLOTS[s as usize].loc;
        assert!(l == loc::BIRMINGHAM || l == loc::DUDLEY, "slot {s}");
    }
}

// 9.1 "No tiles on the board": with no tiles or links, an Industry card builds at any location
// with a free slot showing its icon (slot preference still applies: Cannock's single [Co]
// slot, not its [Mf, Co] slot). Opponents' pieces do not matter.
#[test]
fn build_no_presence_exception_anywhere() {
    let (mut g, p) = setup(&[COAL_CARD], 50);
    place(&mut g, "birmingham", 1, opp(p), Manufacturer, 1, 0);
    link(&mut g, "birmingham", "coventry", opp(p));
    let slots = slots_for(&g, COAL_CARD, Coal);
    assert!(slots.contains(&sl("cannock", 1)));
    assert!(!slots.contains(&sl("cannock", 0)));
    assert!(slots.contains(&sl("dudley", 0)));
    assert!(slots.contains(&sl("leek", 1)));
    // Once the player has anything on the board the exception ends.
    link(&mut g, "birmingham", "dudley", p);
    assert_eq!(slots_for(&g, COAL_CARD, Coal), vec![sl("dudley", 0)]);
}

// 9.1 "No tiles on the board" applies again at the start of the Rail Era (no tiles or links
// left). Coal connection rules still apply: Manufacturer II (1 iron) is fine anywhere, but a
// Cotton II (1 coal) at unconnected Leek is not.
#[test]
fn build_no_presence_exception_rail_era_start() {
    let cotton_manu = Card::Industry(IndustryCard::CottonOrManufacturer);
    let (mut g, p) = setup(&[COAL_CARD, cotton_manu], 80);
    g.era = Era::Rail;
    develop_below(&mut g, p, Coal, 2);
    develop_below(&mut g, p, Cotton, 2);
    develop_below(&mut g, p, Manufacturer, 2);
    link(&mut g, "birmingham", "coventry", opp(p));
    let coal = slots_for(&g, COAL_CARD, Coal);
    assert!(coal.contains(&sl("cannock", 1)));
    assert!(coal.contains(&sl("dudley", 0)));
    assert!(slots_for(&g, cotton_manu, Manufacturer).contains(&sl("leek", 0)));
    assert!(!slots_for(&g, cotton_manu, Cotton).contains(&sl("leek", 0)));
    g.apply(&build_move(&g, COAL_CARD, Coal, sl("cannock", 1)))
        .unwrap();
    assert_eq!(
        g.slots[sl("cannock", 1) as usize].as_ref().unwrap().level,
        2
    );
}

// RQ-042: the no-tiles exception only opens free slots; it does not let an Industry card
// overbuild an opponent's mine outside the network. A Location card can.
#[test]
fn build_no_presence_exception_excludes_overbuild() {
    let (mut g, p) = setup(&[COAL_CARD, DUDLEY], 50);
    develop_below(&mut g, p, Coal, 2);
    let mine = place(&mut g, "dudley", 0, opp(p), Coal, 1, 0);
    g.slots[mine as usize].as_mut().unwrap().flipped = true;
    g.coal_market = 0;
    assert!(!offered(&g, COAL_CARD, Coal, mine));
    assert!(offered(&g, COAL_CARD, Coal, sl("cannock", 1)));
    assert!(offered(&g, DUDLEY, Coal, mine));
}

// Section 3.2, 9.1 step 1, RQ-007: Wild Location acts as any Location card, but never on a
// farm brewery, even with nothing on the board.
#[test]
fn build_wild_location_excludes_farm_breweries() {
    let wl = Card::WildLocation;
    let (g, _) = setup(&[wl], 50);
    let slots = slots_for(&g, wl, Brewery);
    assert!(slots.contains(&sl("burton-on-trent", 1)));
    assert!(slots.contains(&sl("coalbrookdale", 0)));
    assert!(!slots.contains(&sl("farm-north", 0)));
    assert!(!slots.contains(&sl("farm-south", 0)));
    // Any industry at any town, e.g. a Coal Mine at Dudley.
    assert!(offered(&g, wl, Coal, sl("dudley", 0)));
}

// RQ-007: a farm brewery takes a Brewery card or Wild Industry, and only when the farm is in
// the player's network through their own link (or own tile there). A tile in Cannock alone does
// not put the north farm in the network.
#[test]
fn build_farm_brewery_needs_brewery_card_and_network() {
    let wi = Card::WildIndustry;
    let farm_n = sl("farm-north", 0);
    let farm_s = sl("farm-south", 0);
    let (mut g, p) = setup(&[BREWERY_CARD, wi, COAL_CARD], 50);
    place(&mut g, "cannock", 1, p, Coal, 1, 0);
    assert!(!offered(&g, BREWERY_CARD, Brewery, farm_n));
    // An opponent's link to the farm does not help.
    link(&mut g, "cannock", "farm-north", opp(p));
    assert!(!offered(&g, BREWERY_CARD, Brewery, farm_n));

    // Own link to the farm (south farm: the Kidderminster–Worcester link's third endpoint).
    link(&mut g, "kidderminster", "worcester", p);
    assert!(offered(&g, BREWERY_CARD, Brewery, farm_s));
    assert!(offered(&g, wi, Brewery, farm_s));
    assert!(!offered(&g, BREWERY_CARD, Brewery, farm_n));

    // With nothing on the board, a Brewery card may build on a farm (RQ-007).
    let (g2, _) = setup(&[BREWERY_CARD], 50);
    assert!(offered(&g2, BREWERY_CARD, Brewery, farm_n));
    assert!(offered(&g2, BREWERY_CARD, Brewery, farm_s));
}

// ---------------------------------------------------------------- slots (9.1 step 2)

// 9.1 step 2: a free slot showing only the industry's icon must be used; a two-icon slot only
// when no such slot is free. Birmingham: [Ct, Mf], [Mf], [Ir], [Mf].
#[test]
fn build_single_icon_slot_preferred() {
    let (mut g, p) = setup(&[BIRM], 50);
    link(&mut g, "birmingham", "oxford", opp(p)); // market coal for Manufacturer I
    let (two, one_a, one_b) = (
        sl("birmingham", 0),
        sl("birmingham", 1),
        sl("birmingham", 3),
    );
    assert_eq!(slots_for(&g, BIRM, Manufacturer), vec![one_a, one_b]);
    place(&mut g, "birmingham", 1, opp(p), Manufacturer, 2, 0);
    assert_eq!(slots_for(&g, BIRM, Manufacturer), vec![one_b]);
    place(&mut g, "birmingham", 3, opp(p), Manufacturer, 2, 0);
    assert_eq!(slots_for(&g, BIRM, Manufacturer), vec![two]);
}

// ---------------------------------------------------------------- eras (9.1 era restrictions)

// 9.1 era restrictions, RQ-038: in the Canal Era at most one of the player's own tiles per
// location, flipped or not; only overbuilding that tile stays possible. The Rail Era lifts it.
#[test]
fn build_canal_one_own_tile_per_location() {
    let (mut g, p) = setup(&[BIRM], 80);
    link(&mut g, "birmingham", "oxford", opp(p));
    let own = place(&mut g, "birmingham", 1, p, Manufacturer, 1, 0);
    g.slots[own as usize].as_mut().unwrap().flipped = true;
    develop_below(&mut g, p, Manufacturer, 2);
    // Only the overbuild of the own tile; no Cotton, Iron or second Manufacturer.
    assert_eq!(
        targets(&g),
        vec![(BIRM, Manufacturer, own)],
        "canal: one own tile per location"
    );
    g.era = Era::Rail;
    develop_below(&mut g, p, Cotton, 2);
    develop_below(&mut g, p, Iron, 2);
    assert_eq!(
        slots_for(&g, BIRM, Manufacturer),
        vec![own, sl("birmingham", 3)]
    );
    assert!(offered(&g, BIRM, Iron, sl("birmingham", 2)));
}

// 9.1 era restrictions: light half-moon (level I) tiles cannot be built in the Rail Era; the
// player must develop them away first. Pottery I is the exception (RQ-018).
#[test]
fn build_canal_only_tile_rejected_in_rail() {
    let (mut g, p) = setup(&[DUDLEY, Card::Location(loc::COVENTRY)], 80);
    g.era = Era::Rail;
    assert!(slots_for(&g, DUDLEY, Coal).is_empty());
    assert!(slots_for(&g, DUDLEY, Iron).is_empty());
    assert!(offered(
        &g,
        Card::Location(loc::COVENTRY),
        Pottery,
        sl("coventry", 0)
    ));
    develop_below(&mut g, p, Coal, 2);
    assert_eq!(slots_for(&g, DUDLEY, Coal), vec![sl("dudley", 0)]);
}

// 9.1 era restrictions, RQ-017: dark half-moon tiles (Pottery V, Brewery IV) cannot be built in
// the Canal Era.
#[test]
fn build_rail_only_tile_rejected_in_canal() {
    let cov = Card::Location(loc::COVENTRY);
    let burton = Card::Location(loc::BURTON_ON_TRENT);
    let (mut g, p) = setup(&[cov, burton], 80);
    link(&mut g, "birmingham", "coventry", opp(p));
    link(&mut g, "birmingham", "oxford", opp(p)); // market coal for Pottery V
    develop_below(&mut g, p, Pottery, 5);
    develop_below(&mut g, p, Brewery, 4);
    assert!(slots_for(&g, cov, Pottery).is_empty());
    assert!(slots_for(&g, burton, Brewery).is_empty());
    g.era = Era::Rail;
    assert_eq!(slots_for(&g, cov, Pottery), vec![sl("coventry", 0)]);
    assert_eq!(
        slots_for(&g, burton, Brewery),
        vec![sl("burton-on-trent", 1)]
    );
}

// RQ-006: in the Canal Era, overbuilding an opponent's Coal Mine is illegal where the player
// already has a tile; elsewhere (or in the Rail Era) it is allowed.
#[test]
fn build_canal_opponent_overbuild_blocked_by_own_tile() {
    let (mut g, p) = setup(&[DUDLEY], 50);
    develop_below(&mut g, p, Coal, 2);
    g.coal_market = 0;
    let mine = place(&mut g, "dudley", 0, opp(p), Coal, 1, 0);
    g.slots[mine as usize].as_mut().unwrap().flipped = true;
    assert!(offered(&g, DUDLEY, Coal, mine));
    place(&mut g, "dudley", 1, p, Iron, 1, 0);
    assert!(!offered(&g, DUDLEY, Coal, mine));
    g.era = Era::Rail;
    assert!(offered(&g, DUDLEY, Coal, mine));
}

// ---------------------------------------------------------------- overbuilding (9.1)

// 9.1 Overbuilding, RQ-024: an own tile may be replaced by a strictly higher tile of the same
// industry (the lowest on the mat), even with a free slot available. Cubes on the old tile go
// back to the supply; the new tile gets its full contents.
#[test]
fn build_overbuild_own_higher_level() {
    let cannock = Card::Location(loc::CANNOCK);
    let (mut g, p) = setup(&[cannock], 50);
    let own = place(&mut g, "cannock", 1, p, Coal, 1, 1);
    // Same level on the mat: no overbuild.
    assert!(!offered(&g, cannock, Coal, own));
    develop_below(&mut g, p, Coal, 2);
    assert!(offered(&g, cannock, Coal, own));
    // Different industry: never (Cannock [Mf, Co] is still free but blocked by the canal limit).
    assert_eq!(targets(&g), vec![(cannock, Coal, own)]);
    let money = g.players[p as usize].money;
    g.apply(&build_move(&g, cannock, Coal, own)).unwrap();
    let t = g.slots[own as usize].as_ref().unwrap();
    assert_eq!((t.owner, t.level, t.resources, t.flipped), (p, 2, 3, false));
    assert_eq!(g.players[p as usize].money, money - 7);
    assert_eq!(g.players[p as usize].mat_remaining[Coal as usize][1], 1);

    // Any industry: an own flipped Cotton Mill in the Rail Era.
    let (mut g, p) = setup(&[Card::Location(loc::WORCESTER)], 50);
    g.era = Era::Rail;
    let mill = place(&mut g, "worcester", 0, p, Cotton, 1, 0);
    g.slots[mill as usize].as_mut().unwrap().flipped = true;
    develop_below(&mut g, p, Cotton, 2);
    link(&mut g, "worcester", "gloucester", opp(p)); // market coal
    assert!(offered(&g, Card::Location(loc::WORCESTER), Cotton, mill));
}

// 9.1 Overbuilding, RQ-025: an opponent's Coal Mine (or Iron Works) only when no cube of that
// resource is left on the board or in its market. The removed tile's owner keeps the income and
// VP already earned.
#[test]
fn build_overbuild_opponent_coal_only_when_none_left() {
    let (mut g, p) = setup(&[DUDLEY], 50);
    let o = opp(p);
    develop_below(&mut g, p, Coal, 2);
    develop_below(&mut g, p, Iron, 2);
    let mine = place(&mut g, "dudley", 0, o, Coal, 1, 0);
    g.slots[mine as usize].as_mut().unwrap().flipped = true;
    let works = place(&mut g, "dudley", 1, o, Iron, 1, 0);
    g.slots[works as usize].as_mut().unwrap().flipped = true;
    // Coal left in the market.
    g.coal_market = 1;
    assert!(!offered(&g, DUDLEY, Coal, mine));
    // Coal left on a mine elsewhere.
    g.coal_market = 0;
    let other = place(&mut g, "leek", 1, o, Coal, 2, 1);
    assert!(!offered(&g, DUDLEY, Coal, mine));
    g.slots[other as usize] = None;
    assert!(offered(&g, DUDLEY, Coal, mine));
    // Iron (Iron II needs 1 coal: market coal through Birmingham to Oxford).
    let mut gi = g.clone();
    link(&mut gi, "birmingham", "dudley", o);
    link(&mut gi, "birmingham", "oxford", o);
    assert!(!offered(&gi, DUDLEY, Iron, works)); // the market still holds iron
    gi.iron_market = 0;
    assert!(offered(&gi, DUDLEY, Iron, works));

    let before = g.players[o as usize].clone();
    g.apply(&build_move(&g, DUDLEY, Coal, mine)).unwrap();
    let t = g.slots[mine as usize].as_ref().unwrap();
    assert_eq!(
        (t.owner, t.industry, t.level, t.flipped),
        (p, Coal, 2, false)
    );
    assert_eq!(g.players[o as usize].income_space, before.income_space);
    assert_eq!(g.players[o as usize].vp, before.vp);
}

// 9.1 Overbuilding, RQ-025: an opponent's Cotton Mill (or Brewery) can never be overbuilt.
#[test]
fn build_overbuild_opponent_cotton_never() {
    let worc = Card::Location(loc::WORCESTER);
    let burton = Card::Location(loc::BURTON_ON_TRENT);
    let (mut g, p) = setup(&[worc, burton], 50);
    g.era = Era::Rail;
    link(&mut g, "worcester", "gloucester", opp(p));
    develop_below(&mut g, p, Cotton, 2);
    develop_below(&mut g, p, Brewery, 2);
    place(&mut g, "worcester", 0, opp(p), Cotton, 1, 0);
    place(&mut g, "worcester", 1, opp(p), Cotton, 1, 0);
    place(&mut g, "burton-on-trent", 1, opp(p), Brewery, 1, 0);
    assert!(slots_for(&g, worc, Cotton).is_empty());
    assert!(slots_for(&g, burton, Brewery).is_empty());

    // Positive control: the same builds are offered once those tiles are the player's own
    // (overbuild of an own lower tile) or the slots are free, so the refusals above come from
    // the overbuild rule, not from cost or coal.
    let (w0, w1, b1) = (
        sl("worcester", 0),
        sl("worcester", 1),
        sl("burton-on-trent", 1),
    );
    let mut own = g.clone();
    for s in [w0, w1, b1] {
        own.slots[s as usize].as_mut().unwrap().owner = p;
    }
    assert_eq!(slots_for(&own, worc, Cotton), vec![w0, w1]);
    assert_eq!(slots_for(&own, burton, Brewery), vec![b1]);
    let mut free = g.clone();
    for s in [w0, w1, b1] {
        free.slots[s as usize] = None;
    }
    assert_eq!(slots_for(&free, worc, Cotton), vec![w0, w1]);
    assert_eq!(slots_for(&free, burton, Brewery), vec![b1]);
}

// 9.1 Overbuilding, RQ-033: a Rail Era Brewery built over an own Brewery gets 2 barrels, and
// the old tile's cubes are replaced rather than kept.
#[test]
fn build_overbuild_rail_era_brewery_gets_two_beer() {
    let burton = Card::Location(loc::BURTON_ON_TRENT);
    let (mut g, p) = setup(&[burton], 50);
    g.era = Era::Rail;
    develop_below(&mut g, p, Brewery, 2);
    let slot = place(&mut g, "burton-on-trent", 1, p, Brewery, 1, 1);
    assert_eq!(slots_for(&g, burton, Brewery), vec![slot]);
    g.apply(&build_move(&g, burton, Brewery, slot)).unwrap();
    let t = g.slots[slot as usize].as_ref().unwrap();
    assert_eq!((t.owner, t.level, t.resources, t.flipped), (p, 2, 2, false));
}

fn coal_on_board(g: &GameState) -> u8 {
    g.slots
        .iter()
        .flatten()
        .filter(|t| t.industry == Coal)
        .map(|t| t.resources)
        .sum()
}

// 9.1 Overbuilding: the cubes on the overbuilt own tile leave the board (they are not kept).
#[test]
fn build_overbuild_removes_old_cubes_from_board() {
    let cannock = Card::Location(loc::CANNOCK);
    let (mut g, p) = setup(&[cannock], 50);
    let own = place(&mut g, "cannock", 1, p, Coal, 1, 1);
    develop_below(&mut g, p, Coal, 2);
    assert_eq!(coal_on_board(&g), 1);
    g.apply(&build_move(&g, cannock, Coal, own)).unwrap();
    // Old tile's 1 cube is gone; the Coal II tile holds its own 3 (no market link here).
    assert_eq!(coal_on_board(&g), 3);
}

// 3.2: a Wild Industry card builds a non-Brewery industry inside the network; with board
// presence it is refused outside the network.
#[test]
fn build_wild_industry_non_brewery_needs_network() {
    let wi = Card::WildIndustry;
    let (mut g, p) = setup(&[wi], 50);
    link(&mut g, "birmingham", "dudley", p);
    assert!(offered(&g, wi, Coal, sl("dudley", 0)));
    assert!(offered(&g, wi, Cotton, sl("birmingham", 0)));
    for (_, _, s) in targets(&g) {
        let l = SLOTS[s as usize].loc;
        assert!(l == loc::BIRMINGHAM || l == loc::DUDLEY, "slot {s}");
    }
    assert!(!offered(&g, wi, Coal, sl("cannock", 1)));
    assert!(!offered(&g, wi, Cotton, sl("worcester", 0)));
}

// ---------------------------------------------------------------- contents (9.1 steps 4–5)

// 9.1 step 5, RQ-034: a Coal Mine connected to any merchant location sells as many cubes as
// possible to the Coal Market, most expensive empty space first; if emptied it flips at once
// and the owner gains its income. An unconnected mine sells nothing.
#[test]
fn build_coal_mine_sells_to_market_when_connected() {
    let (mut g, p) = setup(&[DUDLEY, Card::Location(loc::CANNOCK)], 50);
    link(&mut g, "birmingham", "dudley", opp(p));
    link(&mut g, "birmingham", "oxford", opp(p));
    g.coal_market = 11; // empty spaces £1, £1, £2
    let income = g.players[p as usize].income_space;
    let slot = sl("dudley", 0);
    g.apply(&build_move(&g, DUDLEY, Coal, slot)).unwrap();
    // Coal I: £5, 2 cubes sold for £2 + £1.
    assert_eq!(g.players[p as usize].money, 50 - 5 + 3);
    assert_eq!(g.players[p as usize].spent, 5);
    assert_eq!(g.coal_market, 13);
    let t = g.slots[slot as usize].as_ref().unwrap();
    assert_eq!((t.resources, t.flipped), (0, true));
    assert_eq!(g.players[p as usize].income_space, income + 4);

    // Unconnected Cannock: nothing sold.
    let (mut g, p) = setup(&[Card::Location(loc::CANNOCK)], 50);
    g.coal_market = 11;
    let slot = sl("cannock", 1);
    g.apply(&build_move(&g, Card::Location(loc::CANNOCK), Coal, slot))
        .unwrap();
    let t = g.slots[slot as usize].as_ref().unwrap();
    assert_eq!((t.resources, t.flipped), (2, false));
    assert_eq!(g.coal_market, 11);
    assert_eq!(g.players[p as usize].money, 45);
}

// 9.1 step 5, RQ-034: an Iron Works always sells to the Iron Market, no connection needed; it
// flips if every cube is sold.
#[test]
fn build_iron_works_sells_and_may_flip() {
    let (mut g, p) = setup(&[DUDLEY], 50);
    place(&mut g, "dudley", 0, opp(p), Coal, 2, 3); // coal at distance 0
    g.iron_market = 8; // empty spaces £1, £1
    let slot = sl("dudley", 1);
    g.apply(&build_move(&g, DUDLEY, Iron, slot)).unwrap();
    // Iron I: £5 + 1 mine coal; 2 of 4 cubes sold for £1 + £1.
    assert_eq!(g.players[p as usize].money, 50 - 5 + 2);
    assert_eq!(g.iron_market, 10);
    let t = g.slots[slot as usize].as_ref().unwrap();
    assert_eq!((t.resources, t.flipped), (2, false));

    // Empty market: all 4 cubes sell into the most expensive spaces (£5 + £5 + £4 + £4) and
    // the works flips.
    let (mut g, p) = setup(&[DUDLEY], 50);
    place(&mut g, "dudley", 0, opp(p), Coal, 2, 3);
    g.iron_market = 0;
    let income = g.players[p as usize].income_space;
    g.apply(&build_move(&g, DUDLEY, Iron, slot)).unwrap();
    assert_eq!(g.players[p as usize].money, 50 - 5 + 18);
    assert_eq!(g.iron_market, 4);
    let t = g.slots[slot as usize].as_ref().unwrap();
    assert_eq!((t.resources, t.flipped), (0, true));
    assert_eq!(g.players[p as usize].income_space, income + 3);
}

// 9.1 step 4, RQ-033: a Brewery gets 1 barrel in the Canal Era and 2 in the Rail Era.
#[test]
fn build_brewery_beer_by_era() {
    let burton = Card::Location(loc::BURTON_ON_TRENT);
    let slot = sl("burton-on-trent", 1);
    let (mut g, _) = setup(&[burton], 50);
    g.apply(&build_move(&g, burton, Brewery, slot)).unwrap();
    assert_eq!(g.slots[slot as usize].as_ref().unwrap().resources, 1);

    let (mut g, p) = setup(&[burton], 50);
    g.era = Era::Rail;
    develop_below(&mut g, p, Brewery, 2);
    g.apply(&build_move(&g, burton, Brewery, slot)).unwrap();
    let t = g.slots[slot as usize].as_ref().unwrap();
    assert_eq!((t.level, t.resources), (2, 2));
}

// 9.1 steps 1–3, section 6: the card is discarded, the lowest mat tile is used, and the mat
// cost plus market coal/iron is paid and added to `spent`. Only affordable builds are offered,
// and apply() rejects anything else.
#[test]
fn build_cost_paid_and_added_to_spent() {
    let (mut g, p) = setup(&[BIRM, DUDLEY], 9);
    link(&mut g, "birmingham", "oxford", opp(p));
    let slot = sl("birmingham", 1);
    // Manufacturer I: £8 + 1 market coal at £1.
    let m = Move::Build {
        card: BIRM,
        industry: Manufacturer,
        slot,
        coal: market(1),
        iron: market(0),
    };
    assert!(g.legal_moves().contains(&m));
    // A wrong plan or a two-icon slot is rejected.
    let before = g.clone();
    for bad in [
        Move::Build {
            card: BIRM,
            industry: Manufacturer,
            slot,
            coal: market(2),
            iron: market(0),
        },
        Move::Build {
            card: BIRM,
            industry: Manufacturer,
            slot: sl("birmingham", 0),
            coal: market(1),
            iron: market(0),
        },
    ] {
        assert_eq!(g.apply(&bad), Err(RuleError::IllegalMove), "{bad:?}");
        assert_eq!(g, before);
    }
    g.apply(&m).unwrap();
    let ps = &g.players[p as usize];
    assert_eq!(ps.money, 0);
    assert_eq!(ps.spent, 9);
    assert_eq!(ps.hand, vec![DUDLEY]);
    assert_eq!(ps.discard.last(), Some(&BIRM));
    assert_eq!(ps.mat_remaining[Manufacturer as usize][0], 0);
    assert_eq!(g.coal_market, COAL_MARKET_START - 1);
    assert_eq!(g.actions_left, 1);

    // £8 is not enough for Manufacturer I (£8 + £1 coal) or Cotton I (£12), but pays for
    // Iron I (£5 + £1 coal).
    let (mut g, p) = setup(&[BIRM], 8);
    link(&mut g, "birmingham", "oxford", opp(p));
    assert!(slots_for(&g, BIRM, Manufacturer).is_empty());
    assert!(slots_for(&g, BIRM, Cotton).is_empty());
    assert_eq!(slots_for(&g, BIRM, Iron), vec![sl("birmingham", 2)]);
}
