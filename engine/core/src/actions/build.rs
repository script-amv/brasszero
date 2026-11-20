//! Build (rules-reference section 9.1): discard a card, place the lowest mat tile of an
//! industry on a slot (or overbuild), pay its cost, then stock and possibly sell its cubes.

use crate::actions::{actor, discard, pay};
use crate::data::*;
use crate::flow;
use crate::moves::Move;
use crate::network::{
    Adjacency, LocSet, coal_buy_cost, coal_sell_to_market, connected_to_any_merchant,
    has_board_presence, iron_buy_cost, iron_sell_to_market, player_network,
};
use crate::resources::{CoalSources, CoalTarget, ResourcePlan, consume, flip, iron_plans};
use crate::state::{GameState, PlacedTile};

/// One buildable `(industry, slot)` with every affordable `(coal, iron)` sourcing.
struct Target {
    industry: Industry,
    slot: SlotId,
    /// Whether an Industry card may pay for it: the location is in the actor's network, or
    /// the "no tiles on the board" exception applies (free slots only, RQ-042).
    industry_card_ok: bool,
    /// Range of [`Targets::plans`] (shared by every slot of one industry at one location).
    plans: std::ops::Range<usize>,
}

/// Every buildable target, with the plan lists they share.
struct Targets {
    targets: Vec<Target>,
    plans: Vec<(ResourcePlan, ResourcePlan)>,
}

/// The lowest mat tile of `ind` if it may be built in the current era.
fn buildable_level(g: &GameState, p: PlayerId, ind: Industry) -> Option<u8> {
    let lvl = g.players[p as usize].lowest_level(ind)?;
    let t = tile(ind, lvl);
    let era_ok = match g.era {
        Era::Canal => t.canal_ok,
        Era::Rail => t.rail_ok,
    };
    era_ok.then_some(lvl)
}

/// No cube of `ind`'s resource on any board tile nor in its market (RQ-025).
fn resource_exhausted(g: &GameState, ind: Industry) -> bool {
    let market = match ind {
        Industry::Coal => g.coal_market,
        Industry::Iron => g.iron_market,
        _ => return false,
    };
    market == 0
        && g.slots
            .iter()
            .flatten()
            .all(|t| t.industry != ind || t.resources == 0)
}

/// Whether `p` may overbuild `old` with a level `lvl` tile of `ind` (section 9.1
/// Overbuilding, RQ-024, RQ-025; the Canal Era limit RQ-006 is checked by the caller).
fn may_overbuild(g: &GameState, p: PlayerId, old: &PlacedTile, ind: Industry, lvl: u8) -> bool {
    old.industry == ind && old.level < lvl && (old.owner == p || resource_exhausted(g, ind))
}

/// `LOC_SLOTS[l]..LOC_SLOTS[l + 1]` are the `SLOTS` indices at location `l` (`SLOTS` is
/// flattened in location order; checked at compile time).
const LOC_SLOTS: [usize; LOCATIONS.len() + 1] = {
    let mut i = 1;
    while i < SLOTS.len() {
        assert!(
            SLOTS[i - 1].loc <= SLOTS[i].loc,
            "SLOTS not in location order"
        );
        i += 1;
    }
    // start[l] = number of slots at locations before `l`.
    let mut start = [0; LOCATIONS.len() + 1];
    let mut l = 0;
    while l <= LOCATIONS.len() {
        let mut i = 0;
        while i < SLOTS.len() {
            if (SLOTS[i].loc as usize) < l {
                start[l] += 1;
            }
            i += 1;
        }
        l += 1;
    }
    start
};

// The per-location coal plan cache in `targets` is indexed by coal need, at most 2.
const _: () = {
    let mut i = 0;
    while i < TILES.len() {
        assert!(
            TILES[i].cost_coal <= 2,
            "coal cache in `targets` holds needs 0..=2"
        );
        i += 1;
    }
};

/// The empty plan: the only way to source 0 cubes.
static NO_RESOURCE: ResourcePlan = ResourcePlan {
    from_tiles: Vec::new(),
    from_market: 0,
};

/// Every location.
const ALL_LOCS: LocSet = (1 << LOCATIONS.len()) - 1;

/// The towns: the locations a Wild Location card may name (section 3.2, RQ-007).
const TOWNS: LocSet = {
    let mut set = 0;
    let mut l = 0;
    while l < LOCATIONS.len() {
        if matches!(LOCATIONS[l].kind, LocKind::Town) {
            set |= 1 << l;
        }
        l += 1;
    }
    set
};

/// Per industry, the locations with a slot accepting it.
const IND_LOCS: [LocSet; 6] = {
    let mut sets = [0; 6];
    let mut s = 0;
    while s < SLOTS.len() {
        let inds = SLOTS[s].industries;
        let mut k = 0;
        while k < inds.len() {
            sets[inds[k] as usize] |= 1 << SLOTS[s].loc;
            k += 1;
        }
        s += 1;
    }
    sets
};

/// The `SLOTS` indices at `loc`.
fn loc_slots(loc: LocId) -> std::ops::Range<usize> {
    LOC_SLOTS[loc as usize]..LOC_SLOTS[loc as usize + 1]
}

/// Slots at `loc` where `p` may place `ind` at level `lvl`, ascending, each flagged
/// `true` if it is a free slot (false: an overbuild), written to `out` (cleared first).
/// Section 9.1 step 2 and era limits. `own_locs`: locations holding a tile of `p`.
fn slots_at(
    g: &GameState,
    p: PlayerId,
    own_locs: LocSet,
    loc: LocId,
    ind: Industry,
    lvl: u8,
    out: &mut Vec<(SlotId, bool)>,
) {
    out.clear();
    let here = loc_slots(loc).filter(|&s| SLOTS[s].industries.contains(&ind));
    // Canal Era: one own tile per location, flipped or not (RQ-038). Only overbuilding that
    // tile stays possible (RQ-006).
    let own_here = g.era == Era::Canal && own_locs & (1 << loc) != 0;
    // Section 9.1 step 2: a free single-icon slot, if any, must be used.
    let single = here
        .clone()
        .any(|s| g.slots[s].is_none() && SLOTS[s].industries.len() == 1);
    for s in here {
        let ok = match &g.slots[s] {
            None => !own_here && (!single || SLOTS[s].industries.len() == 1),
            Some(old) => may_overbuild(g, p, old, ind, lvl) && (!own_here || old.owner == p),
        };
        if ok {
            out.push((s as SlotId, g.slots[s].is_none()));
        }
    }
}

/// Every `(coal, iron)` plan pair for a level `lvl` `ind` tile at `loc` whose total cost
/// (mat cost plus market purchases) is at most `money`. Coal is sourced to `loc`.
/// `coals` and `irons` are the tile's coal plans at `loc` and its iron plans. Appended to
/// `out`.
fn affordable(
    g: &GameState,
    money: u16,
    t: &TileDef,
    coals: &[ResourcePlan],
    irons: &[ResourcePlan],
    out: &mut Vec<(ResourcePlan, ResourcePlan)>,
) {
    for c in coals {
        let base = t.cost_money as u16 + coal_buy_cost(g.coal_market, c.from_market).0;
        for i in irons {
            if base + iron_buy_cost(g.iron_market, i.from_market).0 <= money {
                out.push((c.clone(), i.clone()));
            }
        }
    }
}

/// Every buildable target for the actor, ignoring which card pays for it, ordered by
/// industry then slot (`SLOTS` is in location order).
/// With `only = Some((ind, loc))`, just the targets of that industry and location.
/// Only targets that some card in `cards` may pay for are listed (others yield no move).
fn targets(g: &GameState, p: PlayerId, cards: &[Card], only: Option<(Industry, LocId)>) -> Targets {
    let money = g.players[p as usize].money;
    let network: LocSet = player_network(g, p);
    let presence = has_board_presence(g, p);
    // Locations where an Industry card may pay (the exact per-slot rule is `card_allows`).
    let industry_card_locs = if presence { network } else { ALL_LOCS };
    // Locations some Location / Wild Location card names.
    let mut location_card_locs: LocSet = 0;
    for &c in cards {
        match c {
            Card::Location(l) => location_card_locs |= 1 << l,
            Card::WildLocation => location_card_locs |= TOWNS,
            _ => {}
        }
    }
    let mut out = Targets {
        targets: Vec::new(),
        plans: Vec::new(),
    };
    let mut coal = CoalSources::new(g, Adjacency::of(g));
    let own_locs = SLOTS
        .iter()
        .zip(&g.slots)
        .filter(|(_, t)| t.as_ref().is_some_and(|t| t.owner == p))
        .fold(0 as LocSet, |set, (d, _)| set | (1 << d.loc));
    let mut slots = Vec::new();
    // Coal plans depend only on (location, coal needed): cache them per location, indexed by
    // the need (0..=2, the most any tile needs).
    let mut coal_cache: [[Option<Vec<ResourcePlan>>; 3]; LOCATIONS.len()] =
        std::array::from_fn(|_| [None, None, None]);
    for ind in Industry::ALL {
        if only.is_some_and(|(i, _)| i != ind) {
            continue;
        }
        let Some(lvl) = buildable_level(g, p, ind) else {
            continue;
        };
        let t = tile(ind, lvl);
        let industry_card = cards.iter().any(|&c| match c {
            Card::Industry(ic) => industry_card_matches(ic, ind),
            Card::WildIndustry => true,
            _ => false,
        });
        // Cheap pre-filter, a superset of `card_allows` over every slot of a location: the
        // costly sourcing below is skipped where no held card could pay.
        let payable = location_card_locs | if industry_card { industry_card_locs } else { 0 };
        let mut locs = IND_LOCS[ind as usize] & payable;
        if let Some((_, l)) = only {
            locs &= 1 << l;
        }
        let mut irons = None;
        while locs != 0 {
            let loc = locs.trailing_zeros() as LocId;
            locs &= locs - 1;
            slots_at(g, p, own_locs, loc, ind, lvl, &mut slots);
            if slots.is_empty() {
                continue;
            }
            let irons = irons.get_or_insert_with(|| iron_plans(g, t.cost_iron));
            let coals: &[ResourcePlan] = if t.cost_coal == 0 {
                // The one plan for no coal (as `coal_plans` gives), without allocating.
                std::slice::from_ref(&NO_RESOURCE)
            } else {
                coal_cache[loc as usize][t.cost_coal as usize]
                    .get_or_insert_with(|| coal.plans(CoalTarget::Loc(loc), t.cost_coal))
            };
            let start = out.plans.len();
            affordable(g, money, t, coals, irons, &mut out.plans);
            let plans = start..out.plans.len();
            if plans.is_empty() {
                continue;
            }
            let in_network = network & (1 << loc) != 0;
            for &(slot, free) in &slots {
                out.targets.push(Target {
                    industry: ind,
                    slot,
                    industry_card_ok: in_network || (!presence && free),
                    plans: plans.clone(),
                });
            }
        }
    }
    out
}

/// Whether `card` may pay for target `t` (section 9.1 step 1, section 3.2).
/// Location cards name towns only, and Wild Location excludes the farm breweries, so a farm
/// is reachable only by an Industry card (RQ-007).
fn card_allows(card: Card, t: &Target) -> bool {
    let loc = SLOTS[t.slot as usize].loc;
    match card {
        Card::Location(l) => l == loc,
        Card::WildLocation => LOCATIONS[loc as usize].kind == LocKind::Town,
        Card::Industry(ic) => t.industry_card_ok && industry_card_matches(ic, t.industry),
        Card::WildIndustry => t.industry_card_ok,
    }
}

fn industry_card_matches(ic: IndustryCard, ind: Industry) -> bool {
    match ic {
        IndustryCard::Iron => ind == Industry::Iron,
        IndustryCard::Coal => ind == Industry::Coal,
        IndustryCard::Pottery => ind == Industry::Pottery,
        IndustryCard::Brewery => ind == Industry::Brewery,
        IndustryCard::CottonOrManufacturer => {
            matches!(ind, Industry::Cotton | Industry::Manufacturer)
        }
    }
}

/// `Phase::ChooseAction`: card × (industry, slot) × coal plan × iron plan. `cards` is
/// `distinct_cards` of the actor.
pub(crate) fn legal(g: &GameState, cards: &[Card]) -> Vec<Move> {
    moves(g, cards, None)
}

/// The Build moves of `legal` for industry `ind` at location `loc` (for `apply`'s check).
pub(crate) fn legal_at(g: &GameState, cards: &[Card], ind: Industry, loc: LocId) -> Vec<Move> {
    moves(g, cards, Some((ind, loc)))
}

fn moves(g: &GameState, cards: &[Card], only: Option<(Industry, LocId)>) -> Vec<Move> {
    let p = actor(g);
    let Targets { targets, plans } = targets(g, p, cards, only);
    if targets.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    for &card in cards {
        for t in &targets {
            if !card_allows(card, t) {
                continue;
            }
            out.extend(
                plans[t.plans.clone()]
                    .iter()
                    .map(|(coal, iron)| Move::Build {
                        card,
                        industry: t.industry,
                        slot: t.slot,
                        coal: coal.clone(),
                        iron: iron.clone(),
                    }),
            );
        }
    }
    out
}

pub(crate) fn apply(
    g: &mut GameState,
    card: Card,
    industry: Industry,
    slot: SlotId,
    coal: &ResourcePlan,
    iron: &ResourcePlan,
) {
    let p = actor(g);
    discard(g, p, card);
    let ps = &mut g.players[p as usize];
    let level = ps.lowest_level(industry).expect("build legality checked");
    ps.mat_remaining[industry as usize][level as usize - 1] -= 1;
    let t = tile(industry, level);
    // Pay before any sale: sale money cannot pay for the tile (section 9.1 step 5).
    let cost =
        t.cost_money as u16 + consume(g, coal, Industry::Coal) + consume(g, iron, Industry::Iron);
    pay(g, p, cost);

    // An overbuilt tile is simply replaced: its cubes return to the supply and its owner keeps
    // the income and VP already earned.
    let resources = match industry {
        Industry::Brewery => match g.era {
            Era::Canal => BREWERY_BEER_CANAL,
            Era::Rail => BREWERY_BEER_RAIL,
        },
        _ => t.resources,
    };
    g.slots[slot as usize] = Some(PlacedTile {
        owner: p,
        industry,
        level,
        flipped: false,
        resources,
    });
    sell_to_market(g, p, slot, industry);
    flow::end_action(g);
}

/// Section 9.1 step 5, RQ-034: a new Coal Mine connected to any merchant location, or any new
/// Iron Works, sells as many cubes as its market takes; an emptied tile flips at once.
fn sell_to_market(g: &mut GameState, p: PlayerId, slot: SlotId, industry: Industry) {
    let loc = SLOTS[slot as usize].loc;
    let cubes = g.slots[slot as usize]
        .as_ref()
        .expect("just built")
        .resources;
    let (income, sold) = match industry {
        Industry::Coal if connected_to_any_merchant(g, loc) => {
            let (income, sold, market) = coal_sell_to_market(g.coal_market, cubes);
            g.coal_market = market;
            (income, sold)
        }
        Industry::Iron => {
            let (income, sold, market) = iron_sell_to_market(g.iron_market, cubes);
            g.iron_market = market;
            (income, sold)
        }
        _ => return,
    };
    g.players[p as usize].money += income;
    let t = g.slots[slot as usize].as_mut().expect("just built");
    t.resources -= sold;
    if t.resources == 0 {
        flip(g, slot);
    }
}
