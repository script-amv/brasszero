// SPDX-License-Identifier: Apache-2.0
//! Network, connectivity and market pricing.
//! Rules: docs/rules-reference.md section 7 (network/connected), section 2.4 (markets),
//! section 2.1 (merchants as ordinary nodes, farm breweries).

mod common;

use brass_engine::data::*;
use brass_engine::network::*;
use common::*;

fn has(s: LocSet, l: LocId) -> bool {
    s & (1 << l) != 0
}

// Section 7: a location is in your network if it holds your tile or touches your link;
// opponents' links never extend it.
#[test]
fn network_includes_link_endpoints_and_tile_locations() {
    let mut g = empty_game();
    assert!(!has_board_presence(&g, 0));
    place(&mut g, "derby", 0, 0, Industry::Cotton, 1, 0);
    link(&mut g, "birmingham", "coventry", 0);
    link(&mut g, "oxford", "redditch", 1);
    let n = player_network(&g, 0);
    assert!(has(n, loc::DERBY));
    assert!(has(n, loc::BIRMINGHAM) && has(n, loc::COVENTRY));
    assert!(!has(n, loc::OXFORD) && !has(n, loc::REDDITCH));
    assert!(has_board_presence(&g, 0));
    assert!(has_board_presence(&g, 1));
    assert!(!has_board_presence(&g, 2));
}

// Section 7: connection uses links of any player; merchants are ordinary nodes.
#[test]
fn connected_via_opponent_links() {
    let mut g = empty_game();
    link(&mut g, "birmingham", "oxford", 1);
    link(&mut g, "birmingham", "coventry", 2);
    assert!(connected(&g, loc::COVENTRY, loc::OXFORD));
    assert!(!connected(&g, loc::COVENTRY, loc::DERBY));
    assert!(connected_to_any_merchant(&g, loc::COVENTRY));
    assert!(!connected_to_any_merchant(&g, loc::DERBY));
    // A merchant location itself counts as connected to a merchant.
    assert!(connected_to_any_merchant(&g, loc::OXFORD));
}

#[test]
fn distances_count_links() {
    let mut g = empty_game();
    link(&mut g, "birmingham", "coventry", 0);
    link(&mut g, "birmingham", "oxford", 1);
    link(&mut g, "coventry", "nuneaton", 2);
    let d = link_distances(&g, loc::OXFORD);
    assert_eq!(d[loc::OXFORD as usize], 0);
    assert_eq!(d[loc::BIRMINGHAM as usize], 1);
    assert_eq!(d[loc::COVENTRY as usize], 2);
    assert_eq!(d[loc::NUNEATON as usize], 3);
    assert_eq!(d[loc::DERBY as usize], u8::MAX);
}

// Section 2.4: empty coal market costs 8 each (fallback), market stays at 0.
#[test]
fn coal_buy_empty_market_costs_8_each() {
    assert_eq!(coal_buy_cost(0, 3), (24, 0));
    assert_eq!(iron_buy_cost(0, 2), (12, 0));
}

// Section 2.4: setup coal market has 13 cubes, one 1-pound space empty: buy 1 then 2.
#[test]
fn coal_buy_from_start_market() {
    assert_eq!(coal_buy_cost(13, 2), (3, 11));
    // Buying past the end of the market falls back to 8.
    assert_eq!(coal_buy_cost(1, 2), (7 + 8, 0));
    assert_eq!(iron_buy_cost(8, 1), (2, 7));
    assert_eq!(coal_buy_cost(5, 0), (0, 5));
}

// Section 2.4: selling fills the most expensive empty space first and stops when full.
#[test]
fn iron_sell_fills_most_expensive_first() {
    // Start iron market 8: empty spaces are the two 1-pound ones.
    assert_eq!(iron_sell_to_market(8, 1), (1, 1, 9));
    assert_eq!(iron_sell_to_market(8, 5), (2, 2, 10));
    assert_eq!(iron_sell_to_market(0, 2), (10, 2, 2));
    assert_eq!(coal_sell_to_market(13, 3), (1, 1, 14));
    assert_eq!(coal_sell_to_market(12, 2), (2, 2, 14));
}

// Section 7 / RQ-007: the Kidderminster-Worcester link also touches the south farm brewery.
#[test]
fn farm_brewery_extra_endpoint_joins_network() {
    let mut g = empty_game();
    place(&mut g, "kidderminster", 0, 0, Industry::Coal, 1, 1);
    assert!(!has(player_network(&g, 0), loc::FARM_SOUTH));
    link(&mut g, "kidderminster", "worcester", 0);
    let n = player_network(&g, 0);
    assert!(has(n, loc::FARM_SOUTH) && has(n, loc::WORCESTER));
    assert!(connected(&g, loc::FARM_SOUTH, loc::WORCESTER));
    assert_eq!(
        link_distances(&g, loc::FARM_SOUTH)[loc::KIDDERMINSTER as usize],
        1
    );
}

// Section 7 / RQ-015: merchant locations are ordinary nodes; two of them are connected only
// through links, never through an implicit edge between them.
#[test]
fn merchant_locations_connect_only_through_links() {
    let mut g = empty_game();
    assert!(!connected(&g, loc::OXFORD, loc::WARRINGTON));
    assert!(!connected(&g, loc::SHREWSBURY, loc::GLOUCESTER));
    // Separate links to different merchants still do not join them.
    link(&mut g, "birmingham", "oxford", 0);
    link(&mut g, "stoke-on-trent", "warrington", 0);
    assert!(!connected(&g, loc::OXFORD, loc::WARRINGTON));
    assert!(!connected(&g, loc::BIRMINGHAM, loc::WARRINGTON));
}
