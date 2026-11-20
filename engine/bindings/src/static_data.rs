//! Read-only component data derived exclusively from our engine tables.

use brass_engine::data::*;
use serde_json::{Value, json};

/// Versioned, deterministic component JSON. Array indices equal engine IDs.
pub fn render() -> String {
    let locations: Vec<Value> = LOCATIONS
        .iter()
        .enumerate()
        .map(|(i, l)| {
            let slots: Vec<usize> = SLOTS
                .iter()
                .enumerate()
                .filter(|(_, s)| s.loc as usize == i)
                .map(|(j, _)| j)
                .collect();
            json!({ "id": l.id, "kind": l.kind, "slots": slots })
        })
        .collect();

    let slots: Vec<Value> = SLOTS
        .iter()
        .map(|s| json!({ "loc": s.loc, "industries": s.industries }))
        .collect();

    let links: Vec<Value> = LINKS
        .iter()
        .map(|l| json!({ "a": l.a, "b": l.b, "canal": l.canal, "rail": l.rail, "extra": l.extra }))
        .collect();

    let mut first_tile = 0u32;
    let merchant_spaces: Vec<Value> = MERCHANT_SPACES
        .iter()
        .map(|m| {
            let v = json!({
                "loc": m.loc,
                "tile_slots": m.tile_slots,
                "bonus": m.bonus,
                "link_vp": m.link_vp,
                "first_tile": first_tile,
            });
            first_tile += u32::from(m.tile_slots);
            v
        })
        .collect();

    let tiles: Vec<Value> = TILES
        .iter()
        .map(|t| {
            json!({
                "industry": t.industry,
                "level": t.level,
                "count": t.count,
                "cost_money": t.cost_money,
                "cost_coal": t.cost_coal,
                "cost_iron": t.cost_iron,
                "beer_to_sell": t.beer_to_sell,
                "vp": t.vp,
                "link_vp": t.link_vp,
                "income": t.income,
                "resources": t.resources,
                "canal_ok": t.canal_ok,
                "rail_ok": t.rail_ok,
                "developable": t.developable,
            })
        })
        .collect();

    let income_levels: Vec<i8> = (0..=MAX_INCOME_SPACE).map(income_level).collect();

    let mut board = json!({
        "version": 1,
        "industries": Industry::ALL,
        "deck_4p": deck_4p(),
        "merchant_tiles_4p": MERCHANT_TILES_4P,
        "constants": {
            "COAL_MARKET_START": COAL_MARKET_START,
            "COAL_FALLBACK_PRICE": COAL_FALLBACK_PRICE,
            "IRON_MARKET_START": IRON_MARKET_START,
            "IRON_FALLBACK_PRICE": IRON_FALLBACK_PRICE,
            "START_MONEY": START_MONEY, "HAND_SIZE": HAND_SIZE,
            "LOAN_AMOUNT": LOAN_AMOUNT, "LINK_TILES": LINK_TILES,
            "CANAL_COST": CANAL_COST, "RAIL_COST": RAIL_COST,
            "DOUBLE_RAIL_COST": DOUBLE_RAIL_COST,
            "START_INCOME_SPACE": START_INCOME_SPACE,
            "BREWERY_BEER_CANAL": BREWERY_BEER_CANAL,
            "BREWERY_BEER_RAIL": BREWERY_BEER_RAIL,
            "WILD_LOCATION_COUNT": WILD_LOCATION_COUNT,
            "WILD_INDUSTRY_COUNT": WILD_INDUSTRY_COUNT,
            "MAX_INCOME_SPACE": MAX_INCOME_SPACE,
            "MIN_INCOME_LEVEL": MIN_INCOME_LEVEL, "MAX_INCOME_LEVEL": MAX_INCOME_LEVEL
        },
        "locations": locations,
        "slots": slots,
        "links": links,
        "merchant_spaces": merchant_spaces,
        "tiles": tiles,
        "coal_market_prices": COAL_MARKET_PRICES,
        "iron_market_prices": IRON_MARKET_PRICES,
        "income_levels": income_levels,
    });

    // FNV-1a is a non-security content fingerprint over compact JSON without this field.
    // Schema/artifact cryptographic hashes are owned by the training package.
    let payload = serde_json::to_string(&board).expect("component data serializes");
    let hash = payload.bytes().fold(0xcbf29ce484222325u64, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x100000001b3)
    });
    board["content_hash"] = json!(format!("fnv1a64:{hash:016x}"));
    serde_json::to_string(&board).expect("component data serializes")
}
