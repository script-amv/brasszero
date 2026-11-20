// SPDX-License-Identifier: Apache-2.0
//! Shared test helpers. Later tasks extend this module.
#![allow(dead_code)]

use brass_engine::data::*;
use brass_engine::state::{GameState, PlacedTile, new_game};

/// A fresh game (`new_game(0)`): empty board, no tiles or links.
pub fn empty_game() -> GameState {
    new_game(0)
}

/// Place an unflipped tile of `owner` in the `slot_in_loc`-th slot (printed order, 0-based)
/// of the location with kebab-case id `loc`. Panics if the location or slot does not exist.
/// Does not check industry legality or touch the player's mat. Returns the slot id.
pub fn place(
    g: &mut GameState,
    loc: &str,
    slot_in_loc: usize,
    owner: PlayerId,
    ind: Industry,
    level: u8,
    resources: u8,
) -> SlotId {
    let l = loc_id(loc).unwrap_or_else(|| panic!("no location {loc}"));
    let slot_id = SLOTS
        .iter()
        .enumerate()
        .filter(|(_, s)| s.loc == l)
        .nth(slot_in_loc)
        .unwrap_or_else(|| panic!("no slot {slot_in_loc} at {loc}"))
        .0;
    g.slots[slot_id] = Some(PlacedTile {
        owner,
        industry: ind,
        level,
        flipped: false,
        resources,
    });
    slot_id as SlotId
}

/// Build a link of `owner` between the locations with ids `a` and `b` (either order).
/// Panics if no such link exists on the board.
pub fn link(g: &mut GameState, a: &str, b: &str, owner: PlayerId) {
    let id = link_id(a, b);
    g.links[id as usize] = Some(owner);
}

/// Id of the link between the locations with ids `a` and `b` (either order).
/// Panics if no such link exists on the board.
pub fn link_id(a: &str, b: &str) -> LinkId {
    let (a, b) = (loc_id(a).expect("loc a"), loc_id(b).expect("loc b"));
    LINKS
        .iter()
        .position(|l| (l.a == a && l.b == b) || (l.a == b && l.b == a))
        .unwrap_or_else(|| panic!("no link between {a} and {b}")) as LinkId
}
