//! Board map: locations, industry slots, links and merchant spaces.

use serde::{Deserialize, Serialize};

use super::{Industry, LocId};
use Industry::{
    Brewery as Br, Coal as Co, Cotton as Ct, Iron as Ir, Manufacturer as Mf, Pottery as Po,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum LocKind {
    /// A named town with industry slots and location cards.
    Town,
    /// One of the two unnamed farm breweries (one brewery slot, no location card).
    FarmBrewery,
    /// A merchant space on the board edge (no industry slots).
    Merchant,
}

#[derive(Debug)]
pub struct LocationDef {
    /// Kebab-case id, matching the TS oracle.
    pub id: &'static str,
    pub kind: LocKind,
    /// Industry slots in printed order; each slot lists the industries it accepts.
    pub slots: &'static [&'static [Industry]],
}

#[derive(Debug)]
pub struct SlotDef {
    pub loc: LocId,
    pub industries: &'static [Industry],
}

#[derive(Debug)]
pub struct LinkDef {
    pub a: LocId,
    pub b: LocId,
    pub canal: bool,
    pub rail: bool,
    /// A third location the same link also connects: the southern farm brewery on the
    /// Kidderminster–Worcester link.
    pub extra: Option<LocId>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MerchantBonus {
    /// One free Develop (no iron needed).
    Develop,
    /// Advance the income marker this many spaces.
    Income(u8),
    /// Advance the VP marker this many spaces.
    Vp(u8),
    /// Take this many pounds.
    Money(u8),
}

#[derive(Debug)]
pub struct MerchantSpaceDef {
    pub loc: LocId,
    /// Merchant tile slots at this space (all are filled in a 4-player game).
    pub tile_slots: u8,
    /// Bonus for consuming this merchant's beer when selling to it.
    pub bonus: MerchantBonus,
    /// Printed link icons: VP scored by each adjacent link at era end.
    pub link_vp: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MerchantTile {
    /// Buys cotton, manufactured goods and pottery.
    Any,
    Cotton,
    Manufacturer,
    Pottery,
    /// Buys nothing and gets no beer.
    Blank,
}

/// Location ids as constants, in [`LOCATIONS`] order.
pub mod loc {
    use super::LocId;
    pub const BELPER: LocId = 0;
    pub const DERBY: LocId = 1;
    pub const LEEK: LocId = 2;
    pub const STOKE_ON_TRENT: LocId = 3;
    pub const STONE: LocId = 4;
    pub const UTTOXETER: LocId = 5;
    pub const STAFFORD: LocId = 6;
    pub const BURTON_ON_TRENT: LocId = 7;
    pub const CANNOCK: LocId = 8;
    pub const TAMWORTH: LocId = 9;
    pub const WALSALL: LocId = 10;
    pub const WOLVERHAMPTON: LocId = 11;
    pub const COALBROOKDALE: LocId = 12;
    pub const DUDLEY: LocId = 13;
    pub const KIDDERMINSTER: LocId = 14;
    pub const WORCESTER: LocId = 15;
    pub const BIRMINGHAM: LocId = 16;
    pub const COVENTRY: LocId = 17;
    pub const NUNEATON: LocId = 18;
    pub const REDDITCH: LocId = 19;
    pub const FARM_NORTH: LocId = 20;
    pub const FARM_SOUTH: LocId = 21;
    pub const SHREWSBURY: LocId = 22;
    pub const GLOUCESTER: LocId = 23;
    pub const OXFORD: LocId = 24;
    pub const WARRINGTON: LocId = 25;
    pub const NOTTINGHAM: LocId = 26;
}
use loc::*;

const fn town(id: &'static str, slots: &'static [&'static [Industry]]) -> LocationDef {
    LocationDef {
        id,
        kind: LocKind::Town,
        slots,
    }
}
const fn merchant(id: &'static str) -> LocationDef {
    LocationDef {
        id,
        kind: LocKind::Merchant,
        slots: &[],
    }
}
const FARM_SLOTS: &[&[Industry]] = &[&[Br]];

/// All 27 locations: 20 towns, 2 farm breweries, 5 merchant spaces (RQ-031).
// Slots as printed: [C: board][R p.4–5, p.9] (docs/rules-reference.md §2.1).
pub const LOCATIONS: &[LocationDef] = &[
    town("belper", &[&[Ct, Mf], &[Co], &[Po]]),
    town("derby", &[&[Ct, Br], &[Ct, Mf], &[Ir]]),
    town("leek", &[&[Ct, Mf], &[Ct, Co]]),
    town("stoke-on-trent", &[&[Ct, Mf], &[Po, Ir], &[Mf]]),
    town("stone", &[&[Ct, Br], &[Mf, Co]]),
    town("uttoxeter", &[&[Mf, Br], &[Ct, Br]]),
    town("stafford", &[&[Mf, Br], &[Po]]),
    town("burton-on-trent", &[&[Mf, Co], &[Br]]),
    town("cannock", &[&[Mf, Co], &[Co]]),
    town("tamworth", &[&[Ct, Co], &[Ct, Co]]),
    town("walsall", &[&[Ir, Mf], &[Mf, Br]]),
    town("wolverhampton", &[&[Mf], &[Mf, Co]]),
    town("coalbrookdale", &[&[Ir, Br], &[Ir], &[Co]]),
    town("dudley", &[&[Co], &[Ir]]),
    town("kidderminster", &[&[Ct, Co], &[Ct]]),
    town("worcester", &[&[Ct], &[Ct]]),
    town("birmingham", &[&[Ct, Mf], &[Mf], &[Ir], &[Mf]]),
    town("coventry", &[&[Po], &[Mf, Co], &[Ir, Mf]]),
    town("nuneaton", &[&[Mf, Br], &[Ct, Co]]),
    town("redditch", &[&[Mf, Co], &[Ir]]),
    LocationDef {
        id: "farm-north",
        kind: LocKind::FarmBrewery,
        slots: FARM_SLOTS,
    },
    LocationDef {
        id: "farm-south",
        kind: LocKind::FarmBrewery,
        slots: FARM_SLOTS,
    },
    merchant("shrewsbury"),
    merchant("gloucester"),
    merchant("oxford"),
    merchant("warrington"),
    merchant("nottingham"),
];

/// Looks up a location by its kebab-case id.
pub fn loc_id(id: &str) -> Option<LocId> {
    LOCATIONS
        .iter()
        .position(|l| l.id == id)
        .map(|i| i as LocId)
}

const fn slot(loc: LocId, industries: &'static [Industry]) -> SlotDef {
    SlotDef { loc, industries }
}

/// Every industry slot on the board (49), flattened in [`LOCATIONS`] order.
// [C: board] (§2.1); must equal LOCATIONS' slots (checked by tests).
pub const SLOTS: &[SlotDef] = &[
    slot(BELPER, &[Ct, Mf]),
    slot(BELPER, &[Co]),
    slot(BELPER, &[Po]),
    slot(DERBY, &[Ct, Br]),
    slot(DERBY, &[Ct, Mf]),
    slot(DERBY, &[Ir]),
    slot(LEEK, &[Ct, Mf]),
    slot(LEEK, &[Ct, Co]),
    slot(STOKE_ON_TRENT, &[Ct, Mf]),
    slot(STOKE_ON_TRENT, &[Po, Ir]),
    slot(STOKE_ON_TRENT, &[Mf]),
    slot(STONE, &[Ct, Br]),
    slot(STONE, &[Mf, Co]),
    slot(UTTOXETER, &[Mf, Br]),
    slot(UTTOXETER, &[Ct, Br]),
    slot(STAFFORD, &[Mf, Br]),
    slot(STAFFORD, &[Po]),
    slot(BURTON_ON_TRENT, &[Mf, Co]),
    slot(BURTON_ON_TRENT, &[Br]),
    slot(CANNOCK, &[Mf, Co]),
    slot(CANNOCK, &[Co]),
    slot(TAMWORTH, &[Ct, Co]),
    slot(TAMWORTH, &[Ct, Co]),
    slot(WALSALL, &[Ir, Mf]),
    slot(WALSALL, &[Mf, Br]),
    slot(WOLVERHAMPTON, &[Mf]),
    slot(WOLVERHAMPTON, &[Mf, Co]),
    slot(COALBROOKDALE, &[Ir, Br]),
    slot(COALBROOKDALE, &[Ir]),
    slot(COALBROOKDALE, &[Co]),
    slot(DUDLEY, &[Co]),
    slot(DUDLEY, &[Ir]),
    slot(KIDDERMINSTER, &[Ct, Co]),
    slot(KIDDERMINSTER, &[Ct]),
    slot(WORCESTER, &[Ct]),
    slot(WORCESTER, &[Ct]),
    slot(BIRMINGHAM, &[Ct, Mf]),
    slot(BIRMINGHAM, &[Mf]),
    slot(BIRMINGHAM, &[Ir]),
    slot(BIRMINGHAM, &[Mf]),
    slot(COVENTRY, &[Po]),
    slot(COVENTRY, &[Mf, Co]),
    slot(COVENTRY, &[Ir, Mf]),
    slot(NUNEATON, &[Mf, Br]),
    slot(NUNEATON, &[Ct, Co]),
    slot(REDDITCH, &[Mf, Co]),
    slot(REDDITCH, &[Ir]),
    slot(FARM_NORTH, &[Br]),
    slot(FARM_SOUTH, &[Br]),
];

const fn both(a: LocId, b: LocId) -> LinkDef {
    LinkDef {
        a,
        b,
        canal: true,
        rail: true,
        extra: None,
    }
}
const fn rail(a: LocId, b: LocId) -> LinkDef {
    LinkDef {
        a,
        b,
        canal: false,
        rail: true,
        extra: None,
    }
}
const fn canal(a: LocId, b: LocId) -> LinkDef {
    LinkDef {
        a,
        b,
        canal: true,
        rail: false,
        extra: None,
    }
}

/// All 39 links (30 canal+rail, 8 rail-only, 1 canal-only).
// [C: board][R p.9, p.11] (docs/rules-reference.md §2.2); farm links RQ-007.
pub const LINKS: &[LinkDef] = &[
    both(BELPER, DERBY),
    rail(BELPER, LEEK),
    both(BIRMINGHAM, COVENTRY),
    both(BIRMINGHAM, DUDLEY),
    rail(BIRMINGHAM, NUNEATON),
    both(BIRMINGHAM, OXFORD),
    rail(BIRMINGHAM, REDDITCH),
    both(BIRMINGHAM, TAMWORTH),
    both(BIRMINGHAM, WALSALL),
    both(BIRMINGHAM, WORCESTER),
    rail(BURTON_ON_TRENT, CANNOCK),
    both(BURTON_ON_TRENT, DERBY),
    both(BURTON_ON_TRENT, STONE),
    both(BURTON_ON_TRENT, TAMWORTH),
    canal(BURTON_ON_TRENT, WALSALL),
    both(CANNOCK, STAFFORD),
    both(CANNOCK, FARM_NORTH),
    both(CANNOCK, WALSALL),
    both(CANNOCK, WOLVERHAMPTON),
    both(COALBROOKDALE, KIDDERMINSTER),
    both(COALBROOKDALE, SHREWSBURY),
    both(COALBROOKDALE, WOLVERHAMPTON),
    rail(COVENTRY, NUNEATON),
    both(DERBY, NOTTINGHAM),
    rail(DERBY, UTTOXETER),
    both(DUDLEY, KIDDERMINSTER),
    both(DUDLEY, WOLVERHAMPTON),
    both(GLOUCESTER, REDDITCH),
    both(GLOUCESTER, WORCESTER),
    LinkDef {
        a: KIDDERMINSTER,
        b: WORCESTER,
        canal: true,
        rail: true,
        extra: Some(FARM_SOUTH),
    },
    both(LEEK, STOKE_ON_TRENT),
    both(NUNEATON, TAMWORTH),
    both(REDDITCH, OXFORD),
    both(STAFFORD, STONE),
    both(STOKE_ON_TRENT, STONE),
    both(STOKE_ON_TRENT, WARRINGTON),
    rail(STONE, UTTOXETER),
    rail(TAMWORTH, WALSALL),
    both(WALSALL, WOLVERHAMPTON),
];

/// The 5 merchant spaces. Each prints 2 link icons (scored even with no merchant tile).
// [R p.4, p.10][C: board] (§2.1); link icons RQ-022; Oxford moves spaces RQ-008;
// Gloucester develop RQ-009.
pub const MERCHANT_SPACES: &[MerchantSpaceDef] = &[
    MerchantSpaceDef {
        loc: SHREWSBURY,
        tile_slots: 1,
        bonus: MerchantBonus::Vp(4),
        link_vp: 2,
    },
    MerchantSpaceDef {
        loc: GLOUCESTER,
        tile_slots: 2,
        bonus: MerchantBonus::Develop,
        link_vp: 2,
    },
    MerchantSpaceDef {
        loc: OXFORD,
        tile_slots: 2,
        bonus: MerchantBonus::Income(2),
        link_vp: 2,
    },
    MerchantSpaceDef {
        loc: WARRINGTON,
        tile_slots: 2,
        bonus: MerchantBonus::Money(5),
        link_vp: 2,
    },
    MerchantSpaceDef {
        loc: NOTTINGHAM,
        tile_slots: 2,
        bonus: MerchantBonus::Vp(3),
        link_vp: 2,
    },
];

/// The 9 merchant tiles used in a 4-player game (shuffled onto the 9 tile slots at setup).
// [R p.1, p.4][C: BGG 8612590] (§5).
pub const MERCHANT_TILES_4P: [MerchantTile; 9] = [
    MerchantTile::Any,
    MerchantTile::Cotton,
    MerchantTile::Cotton,
    MerchantTile::Manufacturer,
    MerchantTile::Manufacturer,
    MerchantTile::Pottery,
    MerchantTile::Blank,
    MerchantTile::Blank,
    MerchantTile::Blank,
];
