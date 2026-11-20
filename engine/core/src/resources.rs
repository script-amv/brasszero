//! Resource sourcing and consumption: coal, iron, beer, and tile flips
//! (docs/rules-reference.md section 8 and section 4.7).
//!
//! The `*_plans` functions enumerate every legal way to source a resource need; the action
//! code (Tasks 6–8) picks one and applies it with [`consume`] / [`consume_beer`]. Plans are
//! canonical (tile entries sorted by slot, beer sources sorted), deduplicated, and returned
//! in ascending order, so the move list is deterministic. An empty vector means the need
//! cannot be met and the action is illegal. A need of 0 has exactly one plan: the empty one.

use serde::{Deserialize, Serialize};

use crate::data::*;
use crate::network::{Adjacency, LocSet, N_LOCS, coal_buy_cost, iron_buy_cost, link_locs};
use crate::state::{GameState, PlacedTile};

/// Where coal or iron comes from: cubes taken from tiles (`(slot, count)`, sorted by slot,
/// counts > 0) plus cubes bought from the market.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ResourcePlan {
    pub from_tiles: Vec<(SlotId, u8)>,
    pub from_market: u8,
}

/// One barrel of beer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum BeerSource {
    /// A barrel from the Brewery tile in this slot.
    Brewery(SlotId),
    /// The barrel beside the merchant tile at this flattened merchant tile slot index
    /// (index into `GameState::merchant_beer`).
    Merchant(u8),
}

/// What consumes the coal: an industry tile at a location, or a rail link.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CoalTarget {
    Loc(LocId),
    Link(LinkId),
}

/// Locations the coal target touches: the location, or every location of the link
/// (both ends plus the farm brewery `extra`).
fn target_locs(at: CoalTarget) -> LocSet {
    match at {
        CoalTarget::Loc(l) => 1 << l,
        CoalTarget::Link(id) => link_locs(&LINKS[id as usize]).fold(0, |s, l| s | (1 << l)),
    }
}

/// Unflipped tiles of `ind` that still hold cubes or barrels, in slot order.
fn stocked(g: &GameState, ind: Industry) -> impl Iterator<Item = (SlotId, &PlacedTile)> {
    g.slots.iter().enumerate().filter_map(move |(i, t)| {
        t.as_ref()
            .filter(|t| t.industry == ind && !t.flipped && t.resources > 0)
            .map(|t| (i as SlotId, t))
    })
}

fn canonical(taken: &[(SlotId, u8)], from_market: u8) -> ResourcePlan {
    let mut from_tiles = taken.to_vec();
    from_tiles.sort_unstable();
    ResourcePlan {
        from_tiles,
        from_market,
    }
}

fn sort_dedup<T: Ord>(mut v: Vec<T>) -> Vec<T> {
    v.sort();
    v.dedup();
    v
}

/// Every legal way to source `n` coal for `at` (rules-reference section 8.1).
///
/// 1. Mines: any owner's unflipped Coal Mine with coal, connected to the target by built
///    links. Distance is the number of links; a mine in the same location is distance 0
///    (RQ-014). The closest mine is used first. When several are equally close the player
///    picks ONE and takes all the coal it can give before picking again among the remaining
///    equally close mines; splitting a need across tied mines while the chosen one still has
///    coal is not a plan (RQ-013). Only when a whole distance ring is empty does sourcing move
///    to the next ring.
/// 2. Market: only for coal still needed once every connected mine is empty, and only if the
///    target is connected to a merchant location (bought cheapest first, £8 once empty;
///    see [`consume`]).
///
/// For `CoalTarget::Link`, distance is the minimum over all locations the link touches
/// (including `extra`), using the state as given. The rules measure *after* the link is
/// placed, so the caller (the Network action) must mark the link built in `g` before calling;
/// a mine at either end is then distance 0.
pub fn coal_plans(g: &GameState, at: CoalTarget, n: u8) -> Vec<ResourcePlan> {
    coal_plans_in(g, &Adjacency::of(g), at, n)
}

/// [`coal_plans`] with connections taken from `adj` (`Adjacency::of(g)`, possibly with a
/// link being placed added) instead of `g.links`.
pub(crate) fn coal_plans_in(
    g: &GameState,
    adj: &Adjacency,
    at: CoalTarget,
    n: u8,
) -> Vec<ResourcePlan> {
    CoalSources::new(g, *adj).plans(at, n)
}

/// [`coal_plans`] for many targets on one board: the stocked mines and the locations connected
/// to a merchant are found once and shared by every query.
pub(crate) struct CoalSources {
    adj: Adjacency,
    /// Unflipped Coal Mines with coal: `(slot, location, cubes)`, in slot order.
    mines: Vec<(SlotId, LocId, u8)>,
    /// Locations connected to a merchant location (computed on first use).
    merchant_reach: Option<LocSet>,
}

impl CoalSources {
    /// `adj`: `Adjacency::of(g)`, possibly with a link being placed added.
    pub(crate) fn new(g: &GameState, adj: Adjacency) -> Self {
        let mines = stocked(g, Industry::Coal)
            .map(|(s, t)| (s, SLOTS[s as usize].loc, t.resources))
            .collect();
        CoalSources {
            adj,
            mines,
            merchant_reach: None,
        }
    }

    /// Whether some location in `locs` is connected to a merchant location.
    fn reaches_merchant(&mut self, locs: LocSet) -> bool {
        let adj = &self.adj;
        let reach = *self.merchant_reach.get_or_insert_with(|| {
            let merchants = MERCHANT_SPACES.iter().fold(0, |s, m| s | (1 << m.loc));
            let d = adj.distances_from_set(merchants);
            (0..N_LOCS)
                .filter(|&l| d[l] != u8::MAX)
                .fold(0, |s, l| s | (1 << l))
        });
        reach & locs != 0
    }

    /// [`coal_plans`]`(g, at, n)` for the board given to [`CoalSources::new`].
    pub(crate) fn plans(&mut self, at: CoalTarget, n: u8) -> Vec<ResourcePlan> {
        if n == 0 {
            return vec![ResourcePlan::default()];
        }
        let locs = target_locs(at);
        if self.mines.is_empty() {
            // No mine anywhere: only the market, if the target reaches a merchant.
            return if self.reaches_merchant(locs) {
                vec![canonical(&[], n)]
            } else {
                Vec::new()
            };
        }
        // Distance to the nearest location of the target.
        let dist = self.adj.distances_from_set(locs);
        if n == 1 {
            // Fast path, same result as the general case: one plan per mine of the nearest
            // ring (in slot order), else the market.
            let mut nearest = u8::MAX;
            let mut out = Vec::new();
            for &(s, l, _) in &self.mines {
                let d = dist[l as usize];
                if d < nearest {
                    nearest = d;
                    out.clear();
                }
                if d == nearest && d != u8::MAX {
                    out.push(canonical(&[(s, 1)], 0));
                }
            }
            if out.is_empty() && self.reaches_merchant(locs) {
                out.push(canonical(&[], 1));
            }
            return out;
        }
        let mut mines: Vec<(u8, SlotId, u8)> = self
            .mines
            .iter()
            .map(|&(s, l, r)| (dist[l as usize], s, r))
            .filter(|&(d, _, _)| d != u8::MAX)
            .collect();
        mines.sort_unstable();
        let mut rings: Vec<Vec<(SlotId, u8)>> = Vec::new();
        let mut last = None;
        for (d, s, r) in mines {
            if last != Some(d) {
                rings.push(Vec::new());
                last = Some(d);
            }
            rings.last_mut().expect("just pushed").push((s, r));
        }
        let supply: u32 = rings.iter().flatten().map(|&(_, r)| r as u32).sum();
        let market_ok = supply < n as u32 && self.reaches_merchant(locs);

        let mut out = Vec::new();
        coal_rec(
            &rings,
            0,
            Vec::new(),
            n,
            &mut Vec::new(),
            market_ok,
            &mut out,
        );
        sort_dedup(out)
    }
}

/// `pool`: mines still holding coal in the ring being drained; `next`: the next ring to open.
fn coal_rec(
    rings: &[Vec<(SlotId, u8)>],
    next: usize,
    pool: Vec<(SlotId, u8)>,
    need: u8,
    taken: &mut Vec<(SlotId, u8)>,
    market_ok: bool,
    out: &mut Vec<ResourcePlan>,
) {
    if need == 0 {
        out.push(canonical(taken, 0));
        return;
    }
    if pool.is_empty() {
        match rings.get(next) {
            Some(ring) => coal_rec(rings, next + 1, ring.clone(), need, taken, market_ok, out),
            None if market_ok => out.push(canonical(taken, need)),
            None => {}
        }
        return;
    }
    // The player picks one mine of the ring and takes as much as it can give.
    for i in 0..pool.len() {
        let (s, r) = pool[i];
        let k = r.min(need);
        let mut rest = pool.clone();
        rest.remove(i);
        taken.push((s, k));
        coal_rec(rings, next, rest, need - k, taken, market_ok, out);
        taken.pop();
    }
}

/// Every legal way to source `n` iron (rules-reference section 8.2): any unflipped Iron Works
/// of any owner, no connection needed, in any split. The market is used only for iron still
/// needed once every works is empty, and is always available (£6 once empty).
pub fn iron_plans(g: &GameState, n: u8) -> Vec<ResourcePlan> {
    let works: Vec<(SlotId, u8)> = stocked(g, Industry::Iron)
        .map(|(s, t)| (s, t.resources))
        .collect();
    let supply: u32 = works.iter().map(|&(_, r)| r as u32).sum();
    if supply <= n as u32 {
        // Every works is drained; the rest comes from the market.
        return vec![canonical(&works, n - supply as u8)];
    }
    let mut out = Vec::new();
    iron_rec(&works, n, &mut Vec::new(), &mut out);
    sort_dedup(out)
}

fn iron_rec(
    works: &[(SlotId, u8)],
    need: u8,
    taken: &mut Vec<(SlotId, u8)>,
    out: &mut Vec<ResourcePlan>,
) {
    let Some((&(s, r), rest)) = works.split_first() else {
        if need == 0 {
            out.push(canonical(taken, 0));
        }
        return;
    };
    for k in 0..=r.min(need) {
        if k > 0 {
            taken.push((s, k));
        }
        iron_rec(rest, need - k, taken, out);
        if k > 0 {
            taken.pop();
        }
    }
}

/// Every distinct combination of `n` barrels for player `p` consuming beer at `at`
/// (rules-reference section 8.3), with no priority order:
/// - `p`'s own unflipped Breweries anywhere, no connection needed;
/// - opponents' unflipped Breweries connected to `at`;
/// - the barrel at merchant tile slot `merchant`, if given and still there, at most once.
///
/// The caller checks that `merchant` is the tile being sold to (Sell only; never for the
/// double rail). Each combination is sorted; a brewery appears once per barrel taken.
pub fn beer_plans(
    g: &GameState,
    p: PlayerId,
    at: LocId,
    n: u8,
    merchant: Option<u8>,
) -> Vec<Vec<BeerSource>> {
    beer_plans_in(g, &Adjacency::of(g), p, at, n, merchant)
}

/// [`beer_plans`] with connections taken from `adj` (`Adjacency::of(g)`, possibly with a
/// link being placed added) instead of `g.links`.
pub(crate) fn beer_plans_in(
    g: &GameState,
    adj: &Adjacency,
    p: PlayerId,
    at: LocId,
    n: u8,
    merchant: Option<u8>,
) -> Vec<Vec<BeerSource>> {
    let dist = adj.distances(at);
    let mut sources: Vec<(BeerSource, u8)> = stocked(g, Industry::Brewery)
        // Opponent's brewery: `connected(at, loc)`, with one BFS for all breweries. A brewery
        // in `at` itself is connected (distance 0).
        .filter(|(s, t)| t.owner == p || dist[SLOTS[*s as usize].loc as usize] != u8::MAX)
        .map(|(s, t)| (BeerSource::Brewery(s), t.resources))
        .collect();
    if let Some(m) = merchant.filter(|&m| g.merchant_beer[m as usize]) {
        sources.push((BeerSource::Merchant(m), 1));
    }
    let mut out = Vec::new();
    beer_rec(&sources, n, &mut Vec::new(), &mut out);
    sort_dedup(out)
}

fn beer_rec(
    sources: &[(BeerSource, u8)],
    need: u8,
    taken: &mut Vec<BeerSource>,
    out: &mut Vec<Vec<BeerSource>>,
) {
    let Some((&(src, cap), rest)) = sources.split_first() else {
        if need == 0 {
            let mut v = taken.clone();
            v.sort_unstable();
            out.push(v);
        }
        return;
    };
    for k in 0..=cap.min(need) {
        taken.extend(std::iter::repeat_n(src, k as usize));
        beer_rec(rest, need - k, taken, out);
        taken.truncate(taken.len() - k as usize);
    }
}

/// Apply a coal or iron plan: remove the cubes from the tiles (flipping any tile that is
/// emptied, see [`flip`]) and buy `from_market` cubes from the matching market, lowering it.
///
/// Returns the money cost of the market purchase. It does NOT deduct money from the player
/// or add to `spent`; the action code does that.
///
/// Panics if `kind` is not Coal or Iron, or the plan does not match the board.
pub fn consume(g: &mut GameState, plan: &ResourcePlan, kind: Industry) -> u16 {
    assert!(
        matches!(kind, Industry::Coal | Industry::Iron),
        "consume is for coal or iron, got {kind:?}"
    );
    for &(s, k) in &plan.from_tiles {
        take_from_tile(g, s, kind, k);
    }
    let (cost, market) = if kind == Industry::Coal {
        coal_buy_cost(g.coal_market, plan.from_market)
    } else {
        iron_buy_cost(g.iron_market, plan.from_market)
    };
    if kind == Industry::Coal {
        g.coal_market = market;
    } else {
        g.iron_market = market;
    }
    cost
}

/// Apply beer sources: take each barrel from its Brewery (flipping an emptied one) or remove
/// the merchant barrel. Merchant bonuses are the Sell action's job.
///
/// Panics if a source is not available.
pub fn consume_beer(g: &mut GameState, srcs: &[BeerSource]) {
    for &src in srcs {
        match src {
            BeerSource::Brewery(s) => take_from_tile(g, s, Industry::Brewery, 1),
            BeerSource::Merchant(m) => {
                let beer = &mut g.merchant_beer[m as usize];
                assert!(*beer, "no beer at merchant tile slot {m}");
                *beer = false;
            }
        }
    }
}

fn take_from_tile(g: &mut GameState, s: SlotId, kind: Industry, k: u8) {
    let t = g.slots[s as usize]
        .as_mut()
        .unwrap_or_else(|| panic!("no tile in slot {s}"));
    assert!(
        t.industry == kind && !t.flipped && t.resources >= k,
        "slot {s} cannot supply {k} {kind:?}: {t:?}"
    );
    t.resources -= k;
    if t.resources == 0 {
        flip(g, s);
    }
}

/// Flip the tile in `slot` and advance its *owner's* income marker by the tile's printed
/// income in spaces, capped at space 99 (rules-reference section 4.7, RQ-040). The owner
/// gains even when another player emptied the tile.
///
/// Panics if the slot is empty or the tile is already flipped.
pub fn flip(g: &mut GameState, slot: SlotId) {
    let t = g.slots[slot as usize]
        .as_mut()
        .unwrap_or_else(|| panic!("no tile in slot {slot}"));
    assert!(!t.flipped, "tile in slot {slot} is already flipped");
    t.flipped = true;
    let income = tile(t.industry, t.level).income;
    let owner = &mut g.players[t.owner as usize];
    owner.income_space = owner
        .income_space
        .saturating_add(income)
        .min(MAX_INCOME_SPACE);
}
