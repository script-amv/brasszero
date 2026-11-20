// SPDX-License-Identifier: Apache-2.0
//! Static component data: sanity checks plus a field-by-field cross-check against the
//! TS oracle's data dump (`node oracles/ts-adapter/dump-data.mjs`).

use std::collections::{BTreeMap, BTreeSet};

use brass_engine::data::*;

#[test]
fn deck_has_64_cards_with_expected_split() {
    let d = deck_4p();
    assert_eq!(d.len(), 64);
    assert_eq!(
        d.iter().filter(|c| matches!(c, Card::Location(_))).count(),
        41
    );
    assert_eq!(
        d.iter()
            .filter(|c| **c == Card::Industry(IndustryCard::CottonOrManufacturer))
            .count(),
        8
    );
    let count = |c: Card| d.iter().filter(|x| **x == c).count();
    assert_eq!(count(Card::Industry(IndustryCard::Iron)), 4);
    assert_eq!(count(Card::Industry(IndustryCard::Coal)), 3);
    assert_eq!(count(Card::Industry(IndustryCard::Pottery)), 3);
    assert_eq!(count(Card::Industry(IndustryCard::Brewery)), 5);
    assert_eq!(count(Card::Location(loc_id("birmingham").unwrap())), 3);
    assert_eq!(count(Card::Location(loc_id("redditch").unwrap())), 1);
    // Only towns have location cards (never farm breweries or merchants).
    for c in &d {
        if let Card::Location(l) = c {
            assert_eq!(LOCATIONS[*l as usize].kind, LocKind::Town);
        }
    }
}

#[test]
fn tile_counts_per_player_total_45() {
    let per = |i| {
        TILES
            .iter()
            .filter(|t| t.industry == i)
            .map(|t| t.count as u32)
            .sum::<u32>()
    };
    assert_eq!(
        (
            per(Industry::Manufacturer),
            per(Industry::Cotton),
            per(Industry::Brewery),
            per(Industry::Coal),
            per(Industry::Pottery),
            per(Industry::Iron)
        ),
        (11, 11, 7, 7, 5, 4)
    );
    assert_eq!(TILES.iter().map(|t| t.count as u32).sum::<u32>(), 45);
}

#[test]
fn tile_lookup_and_era_flags() {
    let t = tile(Industry::Manufacturer, 4);
    assert_eq!((t.cost_money, t.cost_iron, t.income), (8, 1, 6));
    // Level-1 tiles cannot be built in the rail era, except pottery I.
    for t in TILES.iter().filter(|t| t.level == 1) {
        assert!(t.canal_ok);
        assert_eq!(
            t.rail_ok,
            t.industry == Industry::Pottery,
            "{:?}",
            t.industry
        );
    }
    assert_eq!(tile(Industry::Manufacturer, 2).income, 1); // RQ-032
    assert_eq!(tile(Industry::Manufacturer, 8).count, 2); // RQ-028
    // Breweries carry no cubes; their beer is an era rule (RQ-033).
    assert!(
        TILES
            .iter()
            .filter(|t| t.industry == Industry::Brewery)
            .all(|t| t.resources == 0)
    );
    // Rail-era-only tiles (RQ-017).
    assert!(!tile(Industry::Pottery, 5).canal_ok);
    assert!(!tile(Industry::Brewery, 4).canal_ok);
    // Lightbulb potteries cannot be developed.
    assert!(!tile(Industry::Pottery, 1).developable);
    assert!(!tile(Industry::Pottery, 3).developable);
    // Levels are contiguous from 1 within each industry.
    for ind in Industry::ALL {
        let levels: Vec<u8> = TILES
            .iter()
            .filter(|t| t.industry == ind)
            .map(|t| t.level)
            .collect();
        assert_eq!(levels, (1..=levels.len() as u8).collect::<Vec<_>>());
    }
}

#[test]
fn income_track_levels() {
    assert_eq!(income_level(10), 0);
    assert_eq!(income_level(0), -10);
    assert_eq!(income_level(11), 1);
    assert_eq!(income_level(30), 10);
    assert_eq!(income_level(31), 11);
    assert_eq!(income_level(99), 30);
    assert_eq!(level_top_space(1), 12);
    assert_eq!(loan_space(10), Some(7)); // level 0 -> -3, top space 7
    assert_eq!(loan_space(2), None); // level -8 -> -11 illegal
    assert_eq!(loan_space(3), Some(0)); // level -7 -> -10
    assert_eq!(level_top_space(30), 99);
    for s in 0..=99u8 {
        assert!(level_top_space(income_level(s)) >= s);
    }
}

#[test]
fn board_has_20_towns_2_farms_5_merchants() {
    let n = |k| LOCATIONS.iter().filter(|l| l.kind == k).count();
    assert_eq!(n(LocKind::Town), 20);
    assert_eq!(n(LocKind::FarmBrewery), 2);
    assert_eq!(n(LocKind::Merchant), 5);
    let ids: BTreeSet<&str> = LOCATIONS.iter().map(|l| l.id).collect();
    assert_eq!(ids.len(), LOCATIONS.len(), "duplicate location id");
    for (i, l) in LOCATIONS.iter().enumerate() {
        assert_eq!(loc_id(l.id), Some(i as LocId));
        match l.kind {
            LocKind::Merchant => assert!(l.slots.is_empty()),
            LocKind::FarmBrewery => assert_eq!(l.slots, &[&[Industry::Brewery][..]]),
            LocKind::Town => assert!(!l.slots.is_empty()),
        }
    }
    // SLOTS is LOCATIONS' slots flattened in location order.
    let flat: Vec<(LocId, &[Industry])> = LOCATIONS
        .iter()
        .enumerate()
        .flat_map(|(i, l)| l.slots.iter().map(move |s| (i as LocId, *s)))
        .collect();
    let slots: Vec<(LocId, &[Industry])> = SLOTS.iter().map(|s| (s.loc, s.industries)).collect();
    assert_eq!(slots, flat);
    assert_eq!(SLOTS.len(), 49);
}

#[test]
fn every_link_endpoint_exists_and_no_duplicate_links() {
    assert_eq!(LINKS.len(), 39);
    let mut seen = BTreeSet::new();
    for l in LINKS {
        assert!((l.a as usize) < LOCATIONS.len() && (l.b as usize) < LOCATIONS.len());
        assert_ne!(l.a, l.b);
        assert!(l.canal || l.rail);
        assert!(seen.insert((l.a.min(l.b), l.a.max(l.b))), "duplicate link");
        if let Some(e) = l.extra {
            assert_eq!(LOCATIONS[e as usize].kind, LocKind::FarmBrewery);
        }
    }
    // Exactly one three-ended link: Kidderminster–Worcester through the southern farm.
    let extra: Vec<_> = LINKS.iter().filter(|l| l.extra.is_some()).collect();
    assert_eq!(extra.len(), 1);
    let names = |l: &LinkDef| {
        let mut v = vec![LOCATIONS[l.a as usize].id, LOCATIONS[l.b as usize].id];
        v.sort();
        v
    };
    assert_eq!(names(extra[0]), ["kidderminster", "worcester"]);
    assert_eq!(extra[0].extra, loc_id("farm-south"));
    // Every location is reachable by some link; merchants and farms only through links.
    for i in 0..LOCATIONS.len() as LocId {
        assert!(
            LINKS
                .iter()
                .any(|l| l.a == i || l.b == i || l.extra == Some(i)),
            "{} has no link",
            LOCATIONS[i as usize].id
        );
    }
}

#[test]
fn merchant_spaces_cover_the_5_merchants() {
    assert_eq!(MERCHANT_SPACES.len(), 5);
    let locs: BTreeSet<LocId> = MERCHANT_SPACES.iter().map(|m| m.loc).collect();
    let merchants: BTreeSet<LocId> = (0..LOCATIONS.len() as LocId)
        .filter(|&i| LOCATIONS[i as usize].kind == LocKind::Merchant)
        .collect();
    assert_eq!(locs, merchants);
    let slots: u32 = MERCHANT_SPACES.iter().map(|m| m.tile_slots as u32).sum();
    assert_eq!(slots as usize, MERCHANT_TILES_4P.len());
}

#[test]
fn merchant_tile_pool_4p() {
    use MerchantTile::*;
    let mut v = MERCHANT_TILES_4P.to_vec();
    v.sort();
    assert_eq!(
        v,
        [
            Any,
            Cotton,
            Cotton,
            Manufacturer,
            Manufacturer,
            Pottery,
            Blank,
            Blank,
            Blank
        ]
    );
}

#[test]
fn track_constants() {
    assert_eq!(
        COAL_MARKET_PRICES,
        [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]
    );
    assert_eq!(IRON_MARKET_PRICES, [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]);
    assert_eq!((COAL_MARKET_START, COAL_FALLBACK_PRICE), (13, 8));
    assert_eq!((IRON_MARKET_START, IRON_FALLBACK_PRICE), (8, 6));
    assert_eq!((START_MONEY, HAND_SIZE, START_INCOME_SPACE), (17, 8, 10));
    assert_eq!(
        (LOAN_AMOUNT, CANAL_COST, RAIL_COST, DOUBLE_RAIL_COST),
        (30, 3, 5, 15)
    );
    assert_eq!((BREWERY_BEER_CANAL, BREWERY_BEER_RAIL), (1, 2));
    assert_eq!((WILD_LOCATION_COUNT, WILD_INDUSTRY_COUNT), (4, 4));
}

#[test]
fn serde_names_are_kebab_case() {
    assert_eq!(
        serde_json::to_string(&Industry::Manufacturer).unwrap(),
        "\"manufacturer\""
    );
    assert_eq!(
        serde_json::to_string(&Card::Industry(IndustryCard::CottonOrManufacturer)).unwrap(),
        "{\"industry\":\"cotton-or-manufacturer\"}"
    );
    let c: Card = serde_json::from_str("\"wild-location\"").unwrap();
    assert_eq!(c, Card::WildLocation);
}

// ---------------------------------------------------------------------------
// Cross-check against the TS oracle dump.
// ---------------------------------------------------------------------------

fn ind_name(i: Industry) -> &'static str {
    match i {
        Industry::Cotton => "cotton",
        Industry::Manufacturer => "manufacturer",
        Industry::Pottery => "pottery",
        Industry::Coal => "coal",
        Industry::Iron => "iron",
        Industry::Brewery => "brewery",
    }
}

fn list<T: ToString>(v: &[T]) -> String {
    v.iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

/// Our data as the same flat `key -> value` map that `dump-data.mjs` writes.
fn our_flat_data() -> BTreeMap<String, String> {
    let mut m = BTreeMap::new();
    let mut put = |k: String, v: String| {
        assert!(m.insert(k.clone(), v).is_none(), "duplicate key {k}");
    };
    for l in LOCATIONS {
        let kind = match l.kind {
            LocKind::Town => "town",
            LocKind::FarmBrewery => "farm-brewery",
            LocKind::Merchant => "merchant",
        };
        put(format!("location.{}.kind", l.id), kind.into());
        if l.kind != LocKind::Merchant {
            let slots: Vec<String> = l
                .slots
                .iter()
                .map(|s| s.iter().map(|i| ind_name(*i)).collect::<Vec<_>>().join("+"))
                .collect();
            put(format!("location.{}.slots", l.id), slots.join("|"));
        }
    }
    for l in LINKS {
        let mut ends = vec![LOCATIONS[l.a as usize].id, LOCATIONS[l.b as usize].id];
        if let Some(e) = l.extra {
            ends.push(LOCATIONS[e as usize].id);
        }
        ends.sort();
        put(
            format!("link.{}", ends.join("~")),
            format!("canal={},rail={}", l.canal, l.rail),
        );
    }
    for ms in MERCHANT_SPACES {
        let id = LOCATIONS[ms.loc as usize].id;
        let bonus = match ms.bonus {
            MerchantBonus::Develop => "develop:1".to_string(),
            MerchantBonus::Income(n) => format!("income:{n}"),
            MerchantBonus::Vp(n) => format!("vp:{n}"),
            MerchantBonus::Money(n) => format!("money:{n}"),
        };
        put(
            format!("merchant.{id}.tile_slots"),
            ms.tile_slots.to_string(),
        );
        put(format!("merchant.{id}.bonus"), bonus);
        put(format!("merchant.{id}.link_vp"), ms.link_vp.to_string());
    }
    let mut pool: BTreeMap<&str, u32> = BTreeMap::new();
    for t in MERCHANT_TILES_4P {
        let name = match t {
            MerchantTile::Any => "any",
            MerchantTile::Cotton => "cotton",
            MerchantTile::Manufacturer => "manufacturer",
            MerchantTile::Pottery => "pottery",
            MerchantTile::Blank => "blank",
        };
        *pool.entry(name).or_default() += 1;
    }
    for (k, v) in pool {
        put(format!("merchant_tiles_4p.{k}"), v.to_string());
    }
    for t in TILES {
        let p = format!("tile.{}.{}", ind_name(t.industry), t.level);
        for (f, v) in [
            ("count", t.count.to_string()),
            ("cost_money", t.cost_money.to_string()),
            ("cost_coal", t.cost_coal.to_string()),
            ("cost_iron", t.cost_iron.to_string()),
            ("beer_to_sell", t.beer_to_sell.to_string()),
            ("vp", t.vp.to_string()),
            ("link_vp", t.link_vp.to_string()),
            ("income", t.income.to_string()),
            ("resources", t.resources.to_string()),
            ("canal_ok", t.canal_ok.to_string()),
            ("rail_ok", t.rail_ok.to_string()),
            ("developable", t.developable.to_string()),
        ] {
            put(format!("{p}.{f}"), v);
        }
    }
    put("market.coal_prices".into(), list(&COAL_MARKET_PRICES));
    put("market.coal_start".into(), COAL_MARKET_START.to_string());
    put(
        "market.coal_fallback".into(),
        COAL_FALLBACK_PRICE.to_string(),
    );
    put("market.iron_prices".into(), list(&IRON_MARKET_PRICES));
    put("market.iron_start".into(), IRON_MARKET_START.to_string());
    put(
        "market.iron_fallback".into(),
        IRON_FALLBACK_PRICE.to_string(),
    );
    put("brewery_beer.canal".into(), BREWERY_BEER_CANAL.to_string());
    put("brewery_beer.rail".into(), BREWERY_BEER_RAIL.to_string());
    for s in 0..=99u8 {
        put(format!("income.level.{s}"), income_level(s).to_string());
        put(
            format!("income.loan.{s}"),
            loan_space(s).map_or("none".into(), |x| x.to_string()),
        );
    }
    for l in -10..=30i8 {
        put(
            format!("income.top_space.{l}"),
            level_top_space(l).to_string(),
        );
    }
    put("setup.start_money".into(), START_MONEY.to_string());
    put("setup.hand_size".into(), HAND_SIZE.to_string());
    put(
        "setup.start_income_space".into(),
        START_INCOME_SPACE.to_string(),
    );
    put("wild.location".into(), WILD_LOCATION_COUNT.to_string());
    put("wild.industry".into(), WILD_INDUSTRY_COUNT.to_string());
    let mut deck: BTreeMap<String, u32> = BTreeMap::new();
    for c in deck_4p() {
        let k = match c {
            Card::Location(l) => format!("location.{}", LOCATIONS[l as usize].id),
            Card::Industry(i) => format!(
                "industry.{}",
                serde_json::to_value(i).unwrap().as_str().unwrap()
            ),
            Card::WildLocation | Card::WildIndustry => unreachable!("wilds are not in the deck"),
        };
        *deck.entry(k).or_default() += 1;
    }
    for (k, v) in deck {
        put(format!("deck.{k}"), v.to_string());
    }
    m
}

#[test]
fn data_matches_oracle_dumps() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let dump = root.join("oracles/cases/ts-data.json");
    let Ok(text) = std::fs::read_to_string(&dump) else {
        // Written to the stderr handle, not with `eprintln!`, so the test harness does not
        // capture it: the skip shows in every `cargo test` run, even when the test passes.
        use std::io::Write;
        let _ = writeln!(
            std::io::stderr(),
            "[SKIPPED] data_matches_oracle_dumps: {} missing (needs the vendored TS oracle, \
             see oracles/PINNED.md; then run `node oracles/ts-adapter/dump-data.mjs`)",
            dump.display()
        );
        return;
    };
    let theirs: BTreeMap<String, String> = serde_json::from_str(&text).expect("ts-data.json");
    let known_text = std::fs::read_to_string(root.join("oracles/known_differences.toml"))
        .expect("oracles/known_differences.toml");
    let known: toml::Table = toml::from_str(&known_text).expect("known_differences.toml");
    let known: BTreeMap<String, String> = known["data"]
        .as_table()
        .expect("[data] table")
        .iter()
        .map(|(k, v)| (k.clone(), v.as_str().expect("string value").to_string()))
        .collect();
    for (k, why) in &known {
        assert!(
            cites_rq(why),
            "known difference {k} must cite an RQ-NNN: {why:?}"
        );
    }

    let ours = our_flat_data();
    let keys: BTreeSet<&String> = ours.keys().chain(theirs.keys()).collect();
    let mut unexplained = Vec::new();
    let mut differing = BTreeSet::new();
    for k in keys {
        let (a, b) = (ours.get(k), theirs.get(k));
        if a != b {
            differing.insert(k.clone());
            if !known.contains_key(k) {
                unexplained.push(format!("{k}: ours={a:?} ts={b:?}"));
            }
        }
    }
    assert!(
        unexplained.is_empty(),
        "unexplained differences:\n{}",
        unexplained.join("\n")
    );
    let stale: Vec<_> = known.keys().filter(|k| !differing.contains(*k)).collect();
    assert!(
        stale.is_empty(),
        "known_differences entries that no longer differ: {stale:?}"
    );
}

/// True if `s` contains `RQ-` followed by three digits.
fn cites_rq(s: &str) -> bool {
    s.as_bytes()
        .windows(6)
        .any(|w| w.starts_with(b"RQ-") && w[3..].iter().all(u8::is_ascii_digit))
}
