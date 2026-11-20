//! Markets, the income track and other fixed numbers.

// Markets: [R p.4, p.8–9][C: board] (docs/rules-reference.md §2.4).

/// Coal market spaces, cheapest first.
pub const COAL_MARKET_PRICES: [u8; 14] = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7];
/// Coal cubes in the market at setup (one £1 space empty).
pub const COAL_MARKET_START: u8 = 13;
/// Price per coal when the market is empty.
pub const COAL_FALLBACK_PRICE: u8 = 8;
/// Iron market spaces, cheapest first.
pub const IRON_MARKET_PRICES: [u8; 10] = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5];
/// Iron cubes in the market at setup (both £1 spaces empty).
pub const IRON_MARKET_START: u8 = 8;
/// Price per iron when the market is empty.
pub const IRON_FALLBACK_PRICE: u8 = 6;

// Setup [R p.4–5] (§5); links [R p.11] (§9.2); loan [R p.11] (§9.5); beer [R p.9] (§4.6, RQ-033);
// wild cards [R p.1] (§1).
pub const START_MONEY: u16 = 17;
pub const HAND_SIZE: usize = 8;
pub const LOAN_AMOUNT: u16 = 30;
/// Link tiles each player owns in total (RQ-043).
pub const LINK_TILES: usize = 14;
pub const CANAL_COST: u16 = 3;
/// One rail link (plus 1 coal).
pub const RAIL_COST: u16 = 5;
/// Two rail links in one action (plus 1 coal each and 1 beer).
pub const DOUBLE_RAIL_COST: u16 = 15;
/// Income track space at setup (income level 0, RQ-001).
pub const START_INCOME_SPACE: u8 = 10;
/// Beer barrels placed on a brewery built in the canal era.
pub const BREWERY_BEER_CANAL: u8 = 1;
/// Beer barrels placed on a brewery built in the rail era.
pub const BREWERY_BEER_RAIL: u8 = 2;
pub const WILD_LOCATION_COUNT: u8 = 4;
pub const WILD_INDUSTRY_COUNT: u8 = 4;

// Progress track [C: board][R p.6, p.8] (§2.3); cap at level 30 (RQ-040).

/// Highest income track space.
pub const MAX_INCOME_SPACE: u8 = 99;
pub const MIN_INCOME_LEVEL: i8 = -10;
pub const MAX_INCOME_LEVEL: i8 = 30;

/// Income level shown at track `space` (0..=99).
///
/// Spaces 0–10 are levels −10…0 (1 space each), 11–30 are 1…10 (2 each),
/// 31–60 are 11…20 (3 each), 61–96 are 21…29 (4 each) and 97–99 are level 30.
pub fn income_level(space: u8) -> i8 {
    assert!(
        space <= MAX_INCOME_SPACE,
        "income space {space} out of range"
    );
    let s = space as i8;
    match space {
        0..=10 => s - 10,
        11..=30 => 1 + (s - 11) / 2,
        31..=60 => 11 + (s - 31) / 3,
        61..=96 => 21 + (s - 61) / 4,
        _ => 30,
    }
}

/// Highest track space of income `level` (−10..=30).
pub fn level_top_space(level: i8) -> u8 {
    assert!(
        (MIN_INCOME_LEVEL..=MAX_INCOME_LEVEL).contains(&level),
        "income level {level} out of range"
    );
    let l = level as i16;
    let top = match level {
        ..=0 => l + 10,
        1..=10 => 11 + (l - 1) * 2 + 1,
        11..=20 => 31 + (l - 11) * 3 + 2,
        21..=29 => 61 + (l - 21) * 4 + 3,
        _ => 99,
    };
    top as u8
}

/// Track space after taking a loan from `space`: 3 income levels down, on the top space
/// of the new level. `None` if that level would be below −10 (loan not allowed, RQ-002).
pub fn loan_space(space: u8) -> Option<u8> {
    let level = income_level(space) - 3;
    (level >= MIN_INCOME_LEVEL).then(|| level_top_space(level))
}
