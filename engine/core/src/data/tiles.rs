//! Industry tiles on each player mat (45 per player, 29 distinct industry/level pairs).

use super::Industry;

#[derive(Debug, PartialEq, Eq)]
pub struct TileDef {
    pub industry: Industry,
    pub level: u8,
    /// Tiles of this industry and level on one player mat.
    pub count: u8,
    pub cost_money: u8,
    pub cost_coal: u8,
    pub cost_iron: u8,
    /// Beer consumed to sell (flip) this tile; 0 for coal, iron and breweries.
    pub beer_to_sell: u8,
    /// VP scored at era end while flipped.
    pub vp: u8,
    /// Link icons on the flipped side (VP for each adjacent link at era end).
    pub link_vp: u8,
    /// Income spaces advanced when the tile flips.
    pub income: u8,
    /// Coal or iron cubes placed when built. Breweries get beer by era instead
    /// (`BREWERY_BEER_CANAL` / `BREWERY_BEER_RAIL`, RQ-033), so they are 0 here.
    pub resources: u8,
    /// May be built in the canal era (false for the rail-era-only tiles, RQ-017).
    pub canal_ok: bool,
    /// May be built in the rail era (false for level-1 tiles except pottery I).
    pub rail_ok: bool,
    /// May be removed by Develop (false for the lightbulb potteries I and III).
    pub developable: bool,
}

#[allow(clippy::too_many_arguments)]
const fn t(
    industry: Industry,
    level: u8,
    count: u8,
    (cost_money, cost_coal, cost_iron): (u8, u8, u8),
    beer_to_sell: u8,
    vp: u8,
    link_vp: u8,
    income: u8,
    resources: u8,
) -> TileDef {
    TileDef {
        industry,
        level,
        count,
        cost_money,
        cost_coal,
        cost_iron,
        beer_to_sell,
        vp,
        link_vp,
        income,
        resources,
        canal_ok: true,
        rail_ok: level != 1,
        developable: true,
    }
}

const fn rail_too(d: TileDef) -> TileDef {
    TileDef { rail_ok: true, ..d }
}
const fn rail_only(d: TileDef) -> TileDef {
    TileDef {
        canal_ok: false,
        ..d
    }
}
const fn lightbulb(d: TileDef) -> TileDef {
    TileDef {
        developable: false,
        ..d
    }
}

use Industry::*;

/// All tiles, grouped by industry in [`Industry`] order, levels ascending.
/// Columns: industry, level, count, (£, coal, iron), beer, vp, link vp, income, cubes.
// [C: mat][C: BGG 4258023][C: BGG 4598816] values; [R p.1] per-industry totals;
// per-level counts RQ-028; era ribbons [R p.10] (RQ-017, RQ-018); lightbulbs [R p.11].
// Source: docs/rules-reference.md §4.
pub const TILES: &[TileDef] = &[
    // §4.1 Cotton Mill
    t(Cotton, 1, 3, (12, 0, 0), 1, 5, 1, 5, 0),
    t(Cotton, 2, 2, (14, 1, 0), 1, 5, 2, 4, 0),
    t(Cotton, 3, 3, (16, 1, 1), 1, 9, 1, 3, 0),
    t(Cotton, 4, 3, (18, 1, 1), 1, 12, 1, 2, 0),
    // §4.2 Manufacturer (IV costs £8: RQ-027)
    t(Manufacturer, 1, 1, (8, 1, 0), 1, 3, 2, 5, 0),
    t(Manufacturer, 2, 2, (10, 0, 1), 1, 5, 1, 1, 0), // income +1: RQ-032
    t(Manufacturer, 3, 1, (12, 2, 0), 0, 4, 0, 4, 0),
    t(Manufacturer, 4, 1, (8, 0, 1), 1, 3, 1, 6, 0),
    t(Manufacturer, 5, 2, (16, 1, 0), 2, 8, 2, 2, 0),
    t(Manufacturer, 6, 1, (20, 0, 0), 1, 7, 1, 6, 0),
    t(Manufacturer, 7, 1, (16, 1, 1), 0, 9, 0, 4, 0),
    t(Manufacturer, 8, 2, (20, 0, 2), 1, 11, 1, 1, 0), // count 2: RQ-028
    // §4.3 Pottery (I is buildable in the Rail Era: RQ-018)
    lightbulb(rail_too(t(Pottery, 1, 1, (17, 0, 1), 1, 10, 1, 5, 0))),
    t(Pottery, 2, 1, (0, 1, 0), 1, 1, 1, 1, 0),
    lightbulb(t(Pottery, 3, 1, (22, 2, 0), 2, 11, 1, 5, 0)),
    t(Pottery, 4, 1, (0, 1, 0), 1, 1, 1, 1, 0),
    rail_only(t(Pottery, 5, 1, (24, 2, 0), 2, 20, 1, 5, 0)),
    // §4.4 Coal Mine
    t(Coal, 1, 1, (5, 0, 0), 0, 1, 2, 4, 2),
    t(Coal, 2, 2, (7, 0, 0), 0, 2, 1, 7, 3),
    t(Coal, 3, 2, (8, 0, 1), 0, 3, 1, 6, 4),
    t(Coal, 4, 2, (10, 0, 1), 0, 4, 1, 5, 5),
    // §4.5 Iron Works
    t(Iron, 1, 1, (5, 1, 0), 0, 3, 1, 3, 4),
    t(Iron, 2, 1, (7, 1, 0), 0, 5, 1, 3, 4),
    t(Iron, 3, 1, (9, 1, 0), 0, 7, 1, 2, 5),
    t(Iron, 4, 1, (12, 1, 0), 0, 9, 1, 1, 6),
    // §4.6 Brewery
    t(Brewery, 1, 2, (5, 0, 1), 0, 4, 2, 4, 0),
    t(Brewery, 2, 2, (7, 0, 1), 0, 5, 2, 5, 0),
    t(Brewery, 3, 2, (9, 0, 1), 0, 7, 2, 5, 0),
    rail_only(t(Brewery, 4, 1, (9, 0, 1), 0, 10, 2, 5, 0)),
];

/// The tile definition for `ind` at `level`. Panics if that level does not exist.
pub fn tile(ind: Industry, level: u8) -> &'static TileDef {
    TILES
        .iter()
        .find(|t| t.industry == ind && t.level == level)
        .unwrap_or_else(|| panic!("no {ind:?} tile at level {level}"))
}
