//! Human-readable text for moves and observations (terminal play). Everything here reads only
//! a [`Move`] or an [`Observation`], so rendering can never leak hidden information.

use std::fmt::Write;

use crate::data::{
    Card, Era, IndustryCard, LOCATIONS, MERCHANT_SPACES, MerchantTile, SLOTS, income_level,
};
use crate::moves::{Move, Phase};
use crate::observe::Observation;
use crate::record::{industry_name, link_name};
use crate::resources::{BeerSource, ResourcePlan};

/// Location id, industry-card kebab name, `wild-location` or `wild-industry`.
pub fn card_name(c: Card) -> String {
    match c {
        Card::Location(l) => LOCATIONS[l as usize].id.to_string(),
        Card::Industry(i) => match i {
            IndustryCard::Iron => "iron",
            IndustryCard::Coal => "coal",
            IndustryCard::Pottery => "pottery",
            IndustryCard::Brewery => "brewery",
            IndustryCard::CottonOrManufacturer => "cotton-or-manufacturer",
        }
        .to_string(),
        Card::WildLocation => "wild-location".to_string(),
        Card::WildIndustry => "wild-industry".to_string(),
    }
}

fn loc_name(slot: u8) -> &'static str {
    LOCATIONS[SLOTS[slot as usize].loc as usize].id
}

fn plan_text(p: &ResourcePlan) -> String {
    let mut parts: Vec<String> = p
        .from_tiles
        .iter()
        .map(|(s, n)| format!("slot {s} x {n}"))
        .collect();
    if p.from_market > 0 {
        parts.push(format!("market x {}", p.from_market));
    }
    parts.join(", ")
}

fn beer_text(beer: &[BeerSource]) -> String {
    beer.iter()
        .map(|b| match b {
            BeerSource::Brewery(s) => format!("slot {s}"),
            BeerSource::Merchant(m) => format!("merchant {m}"),
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// `<clause>[ (discard <cards>)][; coal: ..][; iron: ..][; beer: ..]`.
fn compose(
    clause: String,
    discard: Option<String>,
    coal: &ResourcePlan,
    iron: &ResourcePlan,
    beer: String,
) -> String {
    let mut s = clause;
    if let Some(d) = discard {
        let _ = write!(s, " (discard {d})");
    }
    for (label, text) in [
        ("coal", plan_text(coal)),
        ("iron", plan_text(iron)),
        ("beer", beer),
    ] {
        if !text.is_empty() {
            let _ = write!(s, "; {label}: {text}");
        }
    }
    s
}

/// One-line description of a move, distinct for distinct legal moves.
pub fn describe_move(m: &Move) -> String {
    let none = ResourcePlan::default();
    let card = |c: &Card| Some(card_name(*c));
    match m {
        Move::Build {
            card: c,
            industry,
            slot,
            coal,
            iron,
        } => compose(
            format!(
                "build {} at {} (slot {slot})",
                industry_name(*industry),
                loc_name(*slot)
            ),
            card(c),
            coal,
            iron,
            String::new(),
        ),
        Move::Canal { card: c, link } => compose(
            format!("canal {}", link_name(*link)),
            card(c),
            &none,
            &none,
            String::new(),
        ),
        Move::Rail {
            card: c,
            link,
            coal,
        } => compose(
            format!("rail {}", link_name(*link)),
            card(c),
            coal,
            &none,
            String::new(),
        ),
        Move::SecondRail { link, coal, beer } => compose(
            format!("second rail {}", link_name(*link)),
            None,
            coal,
            &none,
            beer_text(&[BeerSource::Brewery(*beer)]),
        ),
        Move::Develop {
            card: c,
            industry,
            iron,
        } => compose(
            format!("develop {}", industry_name(*industry)),
            card(c),
            &none,
            iron,
            String::new(),
        ),
        Move::DevelopSecond { industry, iron } => compose(
            format!("develop {} again", industry_name(*industry)),
            None,
            &none,
            iron,
            String::new(),
        ),
        Move::Sell {
            card: c,
            slot,
            merchant,
            beer,
        } => compose(
            format!(
                "sell from slot {slot} at {} to merchant {merchant}",
                loc_name(*slot)
            ),
            card(c),
            &none,
            &none,
            beer_text(beer),
        ),
        Move::SellAnother {
            slot,
            merchant,
            beer,
        } => compose(
            format!(
                "sell from slot {slot} at {} to merchant {merchant}",
                loc_name(*slot)
            ),
            None,
            &none,
            &none,
            beer_text(beer),
        ),
        Move::Loan { card: c } => compose("loan".into(), card(c), &none, &none, String::new()),
        Move::Scout { cards } => {
            let names: Vec<String> = cards.iter().map(|c| card_name(*c)).collect();
            compose(
                "scout".into(),
                Some(names.join(", ")),
                &none,
                &none,
                String::new(),
            )
        }
        Move::Pass { card: c } => compose("pass".into(), card(c), &none, &none, String::new()),
        Move::BonusDevelop { industry } => compose(
            format!("bonus develop {}", industry_name(*industry)),
            None,
            &none,
            &none,
            String::new(),
        ),
        Move::ShortfallRemove { slot } => compose(
            format!("remove tile in slot {slot} at {}", loc_name(*slot)),
            None,
            &none,
            &none,
            String::new(),
        ),
        Move::EndAction => "end action".to_string(),
    }
}

fn merchant_tile_name(t: MerchantTile) -> &'static str {
    match t {
        MerchantTile::Any => "any",
        MerchantTile::Cotton => "cotton",
        MerchantTile::Manufacturer => "manufacturer",
        MerchantTile::Pottery => "pottery",
        MerchantTile::Blank => "blank",
    }
}

/// Multi-line board and hand text for the observing player.
pub fn render_observation(o: &Observation) -> String {
    let p = &o.public;
    let mut s = String::new();
    let era = match p.era {
        Era::Canal => "Canal",
        Era::Rail => "Rail",
    };
    let to_move = match p.phase {
        Phase::Shortfall { player, owed } => {
            format!("P{player} must remove tiles (owes £{owed})")
        }
        Phase::GameOver => "game over".to_string(),
        _ => format!(
            "P{} to move, {} action(s) left, phase {:?}",
            p.turn_order[p.turn_idx as usize], p.actions_left, p.phase
        ),
    };
    let _ = writeln!(
        s,
        "== {era} era, round {}; you are P{}; {to_move} ==",
        p.round, o.me
    );
    for (i, pl) in p.players.iter().enumerate() {
        let mut wilds = Vec::new();
        if pl.wild_location_in_hand > 0 {
            wilds.push(format!("{} wild-location", pl.wild_location_in_hand));
        }
        if pl.wild_industry_in_hand > 0 {
            wilds.push(format!("{} wild-industry", pl.wild_industry_in_hand));
        }
        let wilds = if wilds.is_empty() {
            "no wilds".to_string()
        } else {
            wilds.join(", ")
        };
        let _ = writeln!(
            s,
            "P{i}{}: £{}, income level {}, {} VP, spent £{}, {} cards ({wilds})",
            if i == o.me as usize { " (you)" } else { "" },
            pl.money,
            income_level(pl.income_space),
            pl.vp,
            pl.spent,
            o.hand_sizes[i],
        );
    }
    let hand: Vec<String> = o.my_hand.iter().map(|c| card_name(*c)).collect();
    let _ = writeln!(s, "Your hand: {}", hand.join(", "));
    let _ = writeln!(s, "Deck: {} cards", o.deck_size);
    let _ = writeln!(s, "Tiles:");
    for (i, t) in p.slots.iter().enumerate() {
        if let Some(t) = t {
            let state = if t.flipped {
                "flipped".to_string()
            } else {
                format!("cubes {}", t.resources)
            };
            let _ = writeln!(
                s,
                "  {} slot {i}: P{} {} L{} {state}",
                loc_name(i as u8),
                t.owner,
                industry_name(t.industry),
                t.level
            );
        }
    }
    let _ = writeln!(s, "Links:");
    for (i, l) in p.links.iter().enumerate() {
        if let Some(owner) = l {
            let _ = writeln!(s, "  {}: P{owner}", link_name(i as u8));
        }
    }
    let _ = writeln!(
        s,
        "Markets: coal {} cubes, iron {} cubes",
        p.coal_market, p.iron_market
    );
    let _ = writeln!(s, "Merchants:");
    let mut idx = 0;
    for sp in MERCHANT_SPACES {
        for _ in 0..sp.tile_slots {
            let beer = if p.merchant_beer[idx] {
                "beer"
            } else {
                "no beer"
            };
            let _ = writeln!(
                s,
                "  merchant {idx} ({}): {}, {beer}",
                LOCATIONS[sp.loc as usize].id,
                merchant_tile_name(p.merchant_tiles[idx])
            );
            idx += 1;
        }
    }
    s
}
