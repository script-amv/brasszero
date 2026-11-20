//! Network, connectivity and market pricing (docs/rules-reference.md sections 2.4 and 7).

use crate::data::*;
use crate::state::GameState;

/// Number of board locations.
pub const N_LOCS: usize = LOCATIONS.len();
const _: () = assert!(N_LOCS <= 32, "LocSet is a u32 bitset");

/// Bitset of locations: bit `l` is set for `LocId` `l`.
pub type LocSet = u32;

/// All locations a link touches: both endpoints plus the farm brewery `extra`, if any.
pub(crate) fn link_locs(l: &LinkDef) -> impl Iterator<Item = LocId> {
    [Some(l.a), Some(l.b), l.extra].into_iter().flatten()
}

/// Locations in `p`'s network: those holding one of `p`'s tiles or touched by one of `p`'s
/// links (rules-reference section 7). Opponents' links never extend it.
pub fn player_network(g: &GameState, p: PlayerId) -> LocSet {
    let mut set: LocSet = 0;
    for (i, t) in g.slots.iter().enumerate() {
        if t.as_ref().is_some_and(|t| t.owner == p) {
            set |= 1 << SLOTS[i].loc;
        }
    }
    for (i, o) in g.links.iter().enumerate() {
        if *o == Some(p) {
            set = link_locs(&LINKS[i]).fold(set, |s, l| s | (1 << l));
        }
    }
    set
}

/// Whether `p` has any tile or link on the board (the "no tiles on the board" exception).
pub fn has_board_presence(g: &GameState, p: PlayerId) -> bool {
    g.slots.iter().flatten().any(|t| t.owner == p) || g.links.contains(&Some(p))
}

/// Adjacency over built links of any player, as one location bitset per location. A link
/// with `extra` joins all three locations. Computing it once and reusing it saves the
/// per-query scan of every link in move generation.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Adjacency([LocSet; N_LOCS]);

impl Adjacency {
    pub(crate) fn of(g: &GameState) -> Self {
        let mut adj = Adjacency([0; N_LOCS]);
        for (i, o) in g.links.iter().enumerate() {
            if o.is_some() {
                adj = adj.with_link(i as LinkId);
            }
        }
        adj
    }

    /// The adjacency with `link` also built (for sourcing coal and beer for a link that is
    /// being placed: the rules measure after placement).
    pub(crate) fn with_link(mut self, link: LinkId) -> Self {
        let l = &LINKS[link as usize];
        let set = link_locs(l).fold(0 as LocSet, |s, x| s | (1 << x));
        for x in link_locs(l) {
            self.0[x as usize] |= set & !(1 << x);
        }
        self
    }

    /// Link-count distance from `from` to every location; `u8::MAX` if unreachable.
    /// Breadth-first by whole distance rings, as bitsets.
    pub(crate) fn distances(&self, from: LocId) -> [u8; N_LOCS] {
        self.distances_from_set(1 << from)
    }

    /// Distance from the nearest location of `from` (the minimum over its members).
    pub(crate) fn distances_from_set(&self, from: LocSet) -> [u8; N_LOCS] {
        let mut dist = [u8::MAX; N_LOCS];
        let mut seen: LocSet = from;
        let mut ring: LocSet = from;
        let mut d = 0u8;
        while ring != 0 {
            let mut next: LocSet = 0;
            let mut r = ring;
            while r != 0 {
                let x = r.trailing_zeros() as usize;
                r &= r - 1;
                dist[x] = d;
                next |= self.0[x];
            }
            ring = next & !seen;
            seen |= ring;
            d += 1;
        }
        dist
    }

    /// Whether `loc` is a merchant location or connected to one.
    pub(crate) fn reaches_merchant(&self, loc: LocId) -> bool {
        let d = self.distances(loc);
        MERCHANT_SPACES.iter().any(|m| d[m.loc as usize] != u8::MAX)
    }
}

/// Link-count distance from `from` to every location over built links; `u8::MAX` if
/// unreachable.
pub fn link_distances(g: &GameState, from: LocId) -> [u8; N_LOCS] {
    Adjacency::of(g).distances(from)
}

/// Whether a route of built links (any player's) joins `a` and `b`.
pub fn connected(g: &GameState, a: LocId, b: LocId) -> bool {
    link_distances(g, a)[b as usize] != u8::MAX
}

/// Whether `loc` is a merchant location or connected to one (access to the markets).
pub fn connected_to_any_merchant(g: &GameState, loc: LocId) -> bool {
    Adjacency::of(g).reaches_merchant(loc)
}

/// Buy `n` cubes: the filled spaces are the most expensive `market` of `prices`; take the
/// cheapest filled first, then `fallback` once the market is empty.
fn buy_cost(prices: &[u8], fallback: u8, market: u8, n: u8) -> (u16, u8) {
    let first_filled = prices.len() - market as usize;
    let from_market = n.min(market);
    let mut cost: u16 = (0..from_market as usize)
        .map(|k| prices[first_filled + k] as u16)
        .sum();
    cost += (n - from_market) as u16 * fallback as u16;
    (cost, market - from_market)
}

/// Sell up to `n` cubes, filling the most expensive empty space first; stops when full.
fn sell_income(prices: &[u8], market: u8, n: u8) -> (u16, u8, u8) {
    let empty = prices.len() - market as usize;
    let sold = n.min(empty as u8);
    let income = (0..sold as usize)
        .map(|k| prices[empty - 1 - k] as u16)
        .sum();
    (income, sold, market + sold)
}

/// Cost of buying `n` coal from the market and the new market level.
pub fn coal_buy_cost(market: u8, n: u8) -> (u16, u8) {
    buy_cost(&COAL_MARKET_PRICES, COAL_FALLBACK_PRICE, market, n)
}

/// Cost of buying `n` iron from the market and the new market level.
pub fn iron_buy_cost(market: u8, n: u8) -> (u16, u8) {
    buy_cost(&IRON_MARKET_PRICES, IRON_FALLBACK_PRICE, market, n)
}

/// Income, cubes sold and new market level when selling `n` coal.
pub fn coal_sell_to_market(market: u8, n: u8) -> (u16, u8, u8) {
    sell_income(&COAL_MARKET_PRICES, market, n)
}

/// Income, cubes sold and new market level when selling `n` iron.
pub fn iron_sell_to_market(market: u8, n: u8) -> (u16, u8, u8) {
    sell_income(&IRON_MARKET_PRICES, market, n)
}
