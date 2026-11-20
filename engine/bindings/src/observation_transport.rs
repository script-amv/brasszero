//! Strict transport validation, using only the sanitized observation itself.
use brass_engine::data::*;
use brass_engine::moves::Phase;
use brass_engine::observe::Observation;
use brass_engine::state::check_invariants;
use serde_json::Value;

/// No Game or caller-provided seed enters this validator. The hypothetical state is discarded.
pub fn parse(s: &str) -> Result<Observation, String> {
    let input: Value = serde_json::from_str(s).map_err(|e| e.to_string())?;
    let obs: Observation = serde_json::from_str(s).map_err(|e| e.to_string())?;
    // Serde's default permissive unknown-field handling is inappropriate for this bridge.
    // Comparing the entire JSON tree rejects nested extras and private/full-state fields too.
    if serde_json::to_value(&obs).map_err(|e| e.to_string())? != input {
        return Err("unknown fields or noncanonical observation values".into());
    }
    validate(&obs)?;
    // Defense in depth: never surface a native panic as a Python PanicException.
    let checked = std::panic::catch_unwind(|| -> Result<(), String> {
        const VALIDATOR_SEED: u64 = 0x5452_414E_5350_4F52; // Fixed hypothetical transport seed.
        let state = obs.determinize(VALIDATOR_SEED);
        check_invariants(&state)?;
        if !state.is_over() && state.legal_moves().is_empty() {
            return Err("nonterminal state has no legal moves".into());
        }
        if state.observe(obs.me) != obs {
            return Err(
                "observation disagrees with reconstructed public state or legal moves".into(),
            );
        }
        Ok(())
    });
    match checked {
        Ok(result) => result?,
        Err(_) => return Err("invalid observation".into()),
    }
    Ok(obs)
}

fn validate(obs: &Observation) -> Result<(), String> {
    let p = &obs.public;
    let require = |ok: bool, why: &str| if ok { Ok(()) } else { Err(why.to_string()) };
    require(obs.me < 4, "viewer must be 0..3")?;
    require(obs.phase == p.phase, "phase mismatch")?;
    require(obs.deck_size == p.deck_size, "deck size mismatch")?;
    require(
        obs.my_hand.len() == p.players[obs.me as usize].hand_size as usize,
        "viewer hand size mismatch",
    )?;
    require(
        p.slots.len() == SLOTS.len() && p.links.len() == LINKS.len(),
        "board size mismatch",
    )?;
    require(
        p.merchant_tiles.len() == MERCHANT_TILES_4P.len()
            && p.merchant_beer.len() == MERCHANT_TILES_4P.len(),
        "merchant size mismatch",
    )?;
    let mut order = p.turn_order;
    order.sort_unstable();
    require(
        order == [0, 1, 2, 3] && p.turn_idx < 4,
        "invalid turn order or index",
    )?;
    require(
        p.round > 0 && p.round as usize <= deck_4p().len() / (4 * 2),
        "invalid round",
    )?;
    require(
        p.game_over == (p.phase == Phase::GameOver),
        "inconsistent terminal phase",
    )?;
    match p.phase {
        Phase::Shortfall { player, owed } => {
            require(player < 4 && owed > 0 && owed <= 10, "invalid shortfall")?;
            require(
                p.players[player as usize].money == 0 && p.turn_idx == 0,
                "inconsistent shortfall state",
            )?;
        }
        Phase::GameOver => {
            require(
                p.era == Era::Rail
                    && p.deck_size == 0
                    && p.players.iter().all(|ps| ps.hand_size == 0),
                "inconsistent game over",
            )?;
        }
        _ => {
            require((1..=2).contains(&p.actions_left), "invalid actions left")?;
            require(
                !(p.era == Era::Canal && p.round == 1) || p.actions_left == 1,
                "invalid initial round actions",
            )?;
            require(
                p.phase != Phase::SecondRail || p.era == Era::Rail,
                "canal second rail",
            )?;
        }
    }
    let mut merchants = p.merchant_tiles.clone();
    let mut expected_merchants = MERCHANT_TILES_4P.to_vec();
    merchants.sort_unstable();
    expected_merchants.sort_unstable();
    require(
        merchants == expected_merchants,
        "merchant tiles not conserved",
    )?;
    require(
        p.links.iter().flatten().all(|&owner| owner < 4),
        "invalid link owner",
    )?;
    for placed in p.slots.iter().flatten() {
        let def = TILES
            .iter()
            .find(|t| t.industry == placed.industry && t.level == placed.level)
            .ok_or("invalid tile level")?;
        require(
            if p.era == Era::Canal {
                def.canal_ok
            } else {
                def.rail_ok
            },
            "tile is not valid in this era",
        )?;
        let capacity = if placed.industry == Industry::Brewery {
            if p.era == Era::Canal {
                BREWERY_BEER_CANAL
            } else {
                BREWERY_BEER_RAIL
            }
        } else {
            def.resources
        };
        require(
            capacity == 0 || placed.flipped || placed.resources > 0,
            "empty resource tile must be flipped",
        )?;
        require(
            placed.owner < 4 && placed.resources <= capacity,
            "invalid tile owner/resources",
        )?;
    }
    let mut wild_loc = p.wild_location_supply as usize;
    let mut wild_ind = p.wild_industry_supply as usize;
    let mut hidden = p.deck_size as usize;
    for (seat, ps) in p.players.iter().enumerate() {
        require(
            ps.hand_size as usize <= HAND_SIZE && ps.hand_size == obs.hand_sizes[seat],
            "invalid hand size",
        )?;
        let wilds = ps.wild_location_in_hand as usize + ps.wild_industry_in_hand as usize;
        require(
            wilds <= ps.hand_size as usize,
            "wild counts exceed hand size",
        )?;
        wild_loc += ps.wild_location_in_hand as usize;
        wild_ind += ps.wild_industry_in_hand as usize;
        require(ps.income_space <= MAX_INCOME_SPACE, "invalid income space")?;
        require(
            ps.has_face_down_discard == (p.era == Era::Canal),
            "invalid face-down discard flag",
        )?;
        hidden += usize::from(ps.has_face_down_discard);
        if seat != obs.me as usize {
            hidden += ps.hand_size as usize - wilds;
        } else {
            require(
                obs.my_hand
                    .iter()
                    .filter(|&&c| c == Card::WildLocation)
                    .count()
                    == ps.wild_location_in_hand as usize,
                "viewer wild location mismatch",
            )?;
            require(
                obs.my_hand
                    .iter()
                    .filter(|&&c| c == Card::WildIndustry)
                    .count()
                    == ps.wild_industry_in_hand as usize,
                "viewer wild industry mismatch",
            )?;
        }
        for industry in Industry::ALL {
            for (level, &remaining) in ps.mat_remaining[industry as usize].iter().enumerate() {
                let count = TILES
                    .iter()
                    .find(|t| t.industry == industry && t.level as usize == level + 1)
                    .map_or(0, |t| t.count);
                require(remaining <= count, "invalid mat tile count")?;
            }
        }
    }
    require(
        wild_loc == WILD_LOCATION_COUNT as usize && wild_ind == WILD_INDUSTRY_COUNT as usize,
        "wild cards not conserved",
    )?;
    // Validate the exact ordinary card multiset before determinize's pool subtraction.
    let mut pool = deck_4p();
    let seen = obs
        .my_hand
        .iter()
        .filter(|c| !matches!(c, Card::WildLocation | Card::WildIndustry))
        .chain(p.players.iter().flat_map(|ps| &ps.discard));
    for card in seen {
        let index = pool
            .iter()
            .position(|c| c == card)
            .ok_or("invalid or excess seen card")?;
        pool.swap_remove(index);
    }
    require(
        pool.len() == hidden,
        "hidden card sizes do not conserve the deck",
    )?;
    Ok(())
}
