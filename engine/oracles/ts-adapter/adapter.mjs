// SPDX-License-Identifier: Apache-2.0
// JSON-lines adapter: drives the vendored TS oracle (oracles/vendor/BrassBirmingham,
// pinned in oracles/PINNED.md) for the differential harness `brass-diff`. One JSON request
// per stdin line, one JSON response per stdout line; see oracles/protocol.md.
//
// The oracle has no license: it is imported at runtime from the gitignored vendor tree and
// nothing of it is copied here. Usage: node oracles/ts-adapter/adapter.mjs
// (Node >= 22.18: built-in TypeScript type stripping; the oracle engine has no runtime
// dependencies, so no `npm install` is needed).

import { existsSync } from 'node:fs';
import { register } from 'node:module';
import { dirname, join } from 'node:path';
import { createInterface } from 'node:readline';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const src = join(here, '../vendor/BrassBirmingham/packages/engine/src');
if (!existsSync(src)) {
  console.error(`TS oracle not found at ${src}; see oracles/PINNED.md`);
  process.exit(1);
}

// The oracle's sources import each other as './x.js' while the files are './x.ts'.
register(
  'data:text/javascript,' +
    encodeURIComponent(`
      import { existsSync } from 'node:fs';
      import { fileURLToPath } from 'node:url';
      export async function resolve(spec, ctx, next) {
        if (spec.startsWith('.') && spec.endsWith('.js') && ctx.parentURL?.startsWith('file:')) {
          const url = new URL(spec, ctx.parentURL);
          if (!existsSync(fileURLToPath(url))) return next(url.href.slice(0, -3) + '.ts', ctx);
        }
        return next(spec, ctx);
      }`),
);

const E = await import(pathToFileURL(join(src, 'index.ts')).href);

// Our Industry enum order (indexes of `mat_remaining`).
const INDUSTRIES = ['cotton', 'manufacturer', 'pottery', 'coal', 'iron', 'brewery'];

// ---- Session state -------------------------------------------------------------------------

let T = null; // our static tables (sent with `init`), plus derived maps
let state = null; // the TS GameState
let discards = null; // per-player discard piles (our Card JSON), see `trackDiscards`
let railRoundBase = 0; // TS round number of Rail round 1 (TS numbers rounds across eras)
const idCounters = new Map(); // fresh TS card ids per card face

// ---- Cards ---------------------------------------------------------------------------------

function freshId(base) {
  const n = idCounters.get(base) ?? 0;
  idCounters.set(base, n + 1);
  return `${base}-x${n}`;
}

/** Our Card JSON -> a new TS card (fresh unique id). */
function toTsCard(c) {
  if (c === 'wild-location' || c === 'wild-industry') return { id: freshId(c), kind: c };
  if (c.location !== undefined) {
    const location = T.locations[c.location];
    return { id: freshId(`loc-${location}`), kind: 'location', location };
  }
  const industries = c.industry === 'cotton-or-manufacturer' ? ['cotton', 'manufacturer'] : [c.industry];
  return { id: freshId(`ind-${industries.join('-')}`), kind: 'industry', industries };
}

/** A TS card -> our Card JSON. */
function fromTsCard(c) {
  switch (c.kind) {
    case 'wild-location':
    case 'wild-industry':
      return c.kind;
    case 'location':
      return { location: T.locIndex.get(c.location) };
    case 'industry':
      return { industry: c.industries.length === 2 ? 'cotton-or-manufacturer' : c.industries[0] };
  }
  throw new Error(`unknown TS card ${JSON.stringify(c)}`);
}

const sameCard = (a, b) => JSON.stringify(a) === JSON.stringify(b);

/** Ids of `cards` (our JSON) in `player`'s TS hand, each hand card used at most once. */
function handIds(player, cards) {
  const hand = state.players[player].hand;
  const used = new Set();
  return cards.map((c) => {
    const i = hand.findIndex((h, k) => !used.has(k) && sameCard(fromTsCard(h), c));
    if (i < 0) throw new Error(`card ${JSON.stringify(c)} not in player ${player}'s TS hand`);
    used.add(i);
    return { index: i, id: hand[i].id };
  });
}

// ---- Board mapping -------------------------------------------------------------------------

function loadTables(tables) {
  const t = { ...tables };
  t.locIndex = new Map(tables.locations.map((id, i) => [id, i]));
  // Our link id -> TS LINKS index, matched by endpoint set (not by position).
  const key = (ends) => [...ends].sort().join('~');
  const tsIndex = new Map(
    E.LINKS.map((l, i) => [key([l.a, l.b, ...(E.LINK_EXTRA_ENDPOINTS[i] ?? [])]), i]),
  );
  t.linkToTs = tables.links.map((l) => {
    const i = tsIndex.get(key(l.filter((x) => x !== null)));
    if (i === undefined) throw new Error(`no TS link for ${JSON.stringify(l)}`);
    return i;
  });
  t.linkFromTs = new Map(t.linkToTs.map((ts, ours) => [ts, ours]));
  for (const [loc, idx] of tables.slots) {
    if (!E.LOCATIONS[loc] || idx >= E.LOCATIONS[loc].slots.length) {
      throw new Error(`TS has no slot ${idx} at ${loc}`);
    }
  }
  return t;
}

const slotRef = (s) => {
  const [location, slotIndex] = T.slots[s];
  return { location, slotIndex };
};
const sources = (plan) => plan.from_tiles.map(([s, count]) => ({ ...slotRef(s), count }));

// ---- Requests ------------------------------------------------------------------------------

function init(spec, tables) {
  T = loadTables(tables);
  idCounters.clear();
  const g = E.newGame(4, 0);
  const merchants = {};
  for (const id of Object.keys(g.merchants)) merchants[id] = { tiles: [], barrels: [] };
  spec.merchant_tiles.forEach((tile, i) => {
    const [m] = T.merchant_slots[i];
    merchants[m].tiles.push(tile);
    merchants[m].barrels.push(tile !== 'blank');
  });
  state = {
    ...g,
    players: g.players.map((p, i) => ({ ...p, hand: spec.hands[i].map(toTsCard) })),
    // TS draws from the front of its deck; ours draws from the end.
    deck: [...spec.deck].reverse().map(toTsCard),
    discard: spec.discard_bottoms.map(toTsCard),
    merchants,
    turnOrder: [...spec.turn_order],
  };
  discards = spec.discard_bottoms.map((c) => [c]);
  railRoundBase = 0;
}

/** Our ActionRecord -> one TS Action. */
function toTsAction(rec) {
  const p = rec.player;
  const moves = rec.moves.filter((m) => m === 'EndAction' || !('ShortfallRemove' in m));
  const [first, ...rest] = moves;
  if (first === undefined || first === 'EndAction') throw new Error('record has no action');
  const kind = Object.keys(first)[0];
  const m = first[kind];
  const card = (c) => handIds(p, [c])[0].id;
  const follow = (k) => rest.filter((x) => x !== 'EndAction' && k in x).map((x) => x[k]);
  switch (kind) {
    case 'Build':
      return {
        type: 'build',
        cardId: card(m.card),
        industry: m.industry,
        location: T.slots[m.slot][0],
        slotIndex: T.slots[m.slot][1],
        coalSources: sources(m.coal),
        ironSources: sources(m.iron),
      };
    case 'Canal':
      return { type: 'network', cardId: card(m.card), links: [T.linkToTs[m.link]] };
    case 'Rail': {
      const second = follow('SecondRail')[0];
      // One coal per link: a tile source, or null (= the oracle's own resolution, which buys
      // from the market when no mine is connected, the only case our plan uses the market).
      const coal = (plan) => (plan.from_tiles.length ? slotRef(plan.from_tiles[0][0]) : null);
      const a = {
        type: 'network',
        cardId: card(m.card),
        links: [T.linkToTs[m.link]],
        coalSources: [coal(m.coal)],
      };
      if (second) {
        a.links.push(T.linkToTs[second.link]);
        a.coalSources.push(coal(second.coal));
        a.beerSource = slotRef(second.beer);
      }
      return a;
    }
    case 'Develop': {
      const steps = [{ industry: m.industry, iron: m.iron }, ...follow('DevelopSecond')];
      // The oracle lists removals in industry order; the result does not depend on it.
      steps.sort((x, y) => INDUSTRIES.indexOf(x.industry) - INDUSTRIES.indexOf(y.industry));
      return {
        type: 'develop',
        cardId: card(m.card),
        removals: steps.map((s) => s.industry),
        ironSources: steps.flatMap((s) => sources(s.iron)),
      };
    }
    case 'Sell': {
      const sales = [m, ...follow('SellAnother')].map((s) => {
        const [merchant] = T.merchant_slots[s.merchant];
        const beerSources = s.beer.map((b) => {
          if ('Merchant' in b) return { kind: 'merchant', tileIndex: T.merchant_slots[b.Merchant][1] };
          return { kind: 'brewery', ...slotRef(b.Brewery) };
        });
        return {
          ...slotRef(s.slot),
          merchant,
          useMerchantBeer: s.beer.some((b) => 'Merchant' in b),
          beerSources,
        };
      });
      const a = { type: 'sell', cardId: card(m.card), sales };
      const bonus = follow('BonusDevelop');
      if (bonus.length > 0) a.bonusDevelop = bonus[0].industry;
      return a;
    }
    case 'Loan':
      return { type: 'loan', cardId: card(m.card) };
    case 'Pass':
      return { type: 'pass', cardId: card(m.card) };
    case 'Scout': {
      // The oracle wants the three ids in hand order.
      const ids = handIds(p, m.cards).sort((x, y) => x.index - y.index);
      return { type: 'scout', cardIds: ids.map((x) => x.id) };
    }
  }
  throw new Error(`cannot translate ${kind}`);
}

/** Link-count distance from `at` to every node over built links (any player). */
function distances(g, at) {
  const dist = new Map([[at, 0]]);
  let frontier = [at];
  while (frontier.length > 0) {
    const next = [];
    for (const x of frontier) {
      for (const bl of g.board.links) {
        const ends = [E.LINKS[bl.linkIndex].a, E.LINKS[bl.linkIndex].b, ...(E.LINK_EXTRA_ENDPOINTS[bl.linkIndex] ?? [])];
        if (!ends.includes(x)) continue;
        for (const y of ends) {
          if (!dist.has(y)) {
            dist.set(y, dist.get(x) + 1);
            next.push(y);
          }
        }
      }
    }
    frontier = next;
  }
  return dist;
}

/**
 * Explicit resource sources are accepted by the oracle as long as each source is legal on its
 * own (any connected mine, market iron with iron left on the board), so a sourcing rule our
 * engine broke would pass unnoticed. Compare our explicit choice with the oracle's canonical
 * one in the quantities the rules fix: for Build coal, the cubes taken at each distance and
 * the market purchase (closest mines first, market last); for Build and Develop iron, the
 * cubes taken from Iron Works (the market only once every works is empty); for Rail and
 * SecondRail, the closest connected mine once the link is placed (`railCoalIssues`).
 */
function sourcingIssues(g, rec) {
  const issues = [];
  const first = rec.moves[0];
  const kind = Object.keys(first)[0];
  const ironCheck = (need, plans) => {
    const free = E.ironSources(g).reduce((s, x) => s + x.tile.resources, 0);
    const ours = plans.flatMap((p) => p.from_tiles).reduce((s, [, k]) => s + k, 0);
    if (ours !== Math.min(need, free)) {
      issues.push(`iron: ours takes ${ours} from works, oracle canonical ${Math.min(need, free)} (works hold ${free})`);
    }
  };
  if (kind === 'Build') {
    const m = first.Build;
    const def = g.players[rec.player].tiles.find((t) => t.industry === m.industry);
    const loc = T.slots[m.slot][0];
    if (def.costCoal > 0) {
      const dist = distances(g, loc);
      const profile = new Map();
      let need = def.costCoal;
      for (const src of E.coalSources(g, rec.player, loc)) {
        if (need === 0) break;
        const take = Math.min(need, src.tile.resources);
        const d = dist.get(src.location);
        profile.set(d, (profile.get(d) ?? 0) + take);
        need -= take;
      }
      const ourProfile = new Map();
      for (const [s, k] of m.coal.from_tiles) {
        const d = dist.get(T.slots[s][0]);
        ourProfile.set(d, (ourProfile.get(d) ?? 0) + k);
      }
      const show = (p) => JSON.stringify([...p].sort((a, b) => a[0] - b[0]));
      if (show(profile) !== show(ourProfile) || need !== m.coal.from_market) {
        issues.push(
          `coal: ours [distance,cubes] ${show(ourProfile)} + market ${m.coal.from_market}, ` +
            `oracle canonical ${show(profile)} + market ${need}`,
        );
      }
    }
    if (def.costIron > 0) ironCheck(def.costIron, [m.iron]);
  } else if (kind === 'Rail') {
    const seconds = rec.moves.filter((x) => x !== 'EndAction' && 'SecondRail' in x).map((x) => x.SecondRail);
    issues.push(...railCoalIssues(g, [first.Rail, ...seconds]));
  } else if (kind === 'Develop') {
    const steps = [first.Develop, ...rec.moves.filter((x) => x !== 'EndAction' && 'DevelopSecond' in x).map((x) => x.DevelopSecond)];
    ironCheck(steps.length, steps.map((s) => s.iron));
  }
  return issues;
}

/**
 * Rail and SecondRail coal: each link needs one cube, taken from the closest unflipped mine
 * connected once that link is on the board (distance counted in links from the nearer end of
 * the link; mines in an end location are at distance 0), or from the market only when no mine
 * is connected. A second link is judged with the first already placed and the first link's cube
 * already gone. The oracle accepts any connected mine we name, so check our choice here.
 */
function railCoalIssues(g, steps) {
  const issues = [];
  const left = new Map(); // "location#slot" -> cubes still on that coal mine
  for (const [loc, slots] of Object.entries(g.board.slots)) {
    slots.forEach((t, i) => {
      if (t && !t.flipped && t.tile.industry === 'coal' && t.resources > 0) left.set(`${loc}#${i}`, t.resources);
    });
  }
  const placed = [...g.board.links];
  steps.forEach((step, n) => {
    const li = T.linkToTs[step.link];
    placed.push({ linkIndex: li });
    const ends = [E.LINKS[li].a, E.LINKS[li].b, ...(E.LINK_EXTRA_ENDPOINTS[li] ?? [])];
    const view = { board: { links: placed } };
    const dist = new Map();
    for (const end of ends) {
      for (const [loc, d] of distances(view, end)) dist.set(loc, Math.min(d, dist.get(loc) ?? Infinity));
    }
    const reach = [...left.keys()].filter((k) => dist.has(k.split('#')[0]));
    const nearest = Math.min(...reach.map((k) => dist.get(k.split('#')[0])));
    const label = n === 0 ? 'rail' : 'second rail';
    if (step.coal.from_tiles.length === 0) {
      if (reach.length > 0) issues.push(`${label} coal: ours buys from the market, a mine is connected at distance ${nearest}`);
      return;
    }
    const [s] = step.coal.from_tiles[0];
    const [loc, idx] = T.slots[s];
    const k = `${loc}#${idx}`;
    if (!left.has(k) || !dist.has(loc)) {
      issues.push(`${label} coal: ours takes an unavailable or unconnected mine at ${loc}`);
    } else if (dist.get(loc) !== nearest) {
      issues.push(`${label} coal: ours takes a mine at distance ${dist.get(loc)}, closest connected is ${nearest}`);
    }
    if (left.has(k)) left.set(k, left.get(k) - 1);
    if (left.get(k) === 0) left.delete(k);
  });
  return issues;
}

function apply(rec) {
  const before = state;
  const actor = before.turnOrder[before.currentPlayerIdx];
  if (actor !== rec.player) throw new Error(`oracle expects player ${actor}, record is ${rec.player}`);
  const action = toTsAction(rec);
  const sourcing = sourcingIssues(before, rec);
  state = E.applyAction(before, action);
  // The oracle keeps one shared discard pile; attribute what this action added to the actor.
  if (before.era === 'canal' && state.era === 'rail') {
    discards = [[], [], [], []];
    railRoundBase = state.round;
  } else {
    for (const c of state.discard.slice(before.discard.length)) discards[rec.player].push(fromTsCard(c));
  }
  return { action, sourcing };
}

/** The oracle's raw card locations: hands, deck (draw order) and its shared discard pile. */
function cards() {
  return {
    hands: state.players.map((p) => p.hand.map(fromTsCard)),
    deck: state.deck.map(fromTsCard),
    discard_pile: state.discard.map(fromTsCard),
  };
}

/** Replace hands and deck (the harness re-deals the Rail Era from our engine's reshuffle). */
function deal(hands, deck) {
  state = {
    ...state,
    players: state.players.map((p, i) => ({ ...p, hand: hands[i].map(toTsCard) })),
    deck: [...deck].reverse().map(toTsCard),
  };
}

function snapshot() {
  const g = state;
  const over = g.phase === 'game-over';
  const slots = T.slots.map(([loc, idx]) => {
    const t = g.board.slots[loc][idx];
    if (!t) return null;
    return {
      owner: t.player,
      industry: t.tile.industry,
      level: t.tile.level,
      flipped: t.flipped,
      resources: t.resources,
    };
  });
  const links = T.links.map(() => null);
  for (const bl of g.board.links) links[T.linkFromTs.get(bl.linkIndex)] = bl.player;
  const merchantBeer = T.merchant_slots.map(([m, i]) => g.merchants[m].barrels[i]);
  const mat = g.players.map((p) => {
    const m = INDUSTRIES.map(() => [0, 0, 0, 0, 0, 0, 0, 0]);
    for (const t of p.tiles) m[INDUSTRIES.indexOf(t.industry)][t.level - 1] += 1;
    return m;
  });
  const round = g.era === 'rail' ? g.round - railRoundBase + 1 : g.round;
  return {
    era: g.era,
    round: over ? 0 : round,
    turn_order: g.turnOrder,
    turn_idx: over ? 0 : g.currentPlayerIdx,
    actions_left: over ? 0 : E.actionsPerRound(g) - g.actionsThisTurn,
    game_over: over,
    money: g.players.map((p) => p.money),
    income_space: g.players.map((p) => p.incomeSpace),
    vp: g.players.map((p) => p.vp),
    spent: g.players.map((p) => p.spentThisRound),
    hand_sizes: g.players.map((p) => p.hand.length),
    hands: g.players.map((p) => p.hand.map(fromTsCard)),
    discards,
    slots,
    links,
    merchant_beer: merchantBeer,
    coal_market: g.coalMarket,
    iron_market: g.ironMarket,
    mat_remaining: mat,
    deck_size: g.deck.length,
    wild_location_supply: g.wildSupply.location,
    wild_industry_supply: g.wildSupply.industry,
  };
}

/** Coarse legal-move keys (see `legal_summary` in engine/src/record.rs). */
function legal() {
  const g = state;
  if (g.phase === 'game-over') return [];
  const player = g.turnOrder[g.currentPlayerIdx];
  const keys = new Set();
  const linkKey = (i) => {
    const l = E.LINKS[i];
    return [l.a, l.b].sort().join('-');
  };
  for (const a of E.enumerateActions(g, player)) {
    switch (a.type) {
      case 'build':
        keys.add(`build:${a.location}:${a.industry}`);
        break;
      case 'network':
        keys.add(`${g.era}:${linkKey(a.links[0])}`);
        break;
      case 'develop':
        keys.add(`develop:${a.removals[0]}`);
        break;
      case 'sell': {
        const s = a.sales[0];
        keys.add(`sell:${s.location}:${g.board.slots[s.location][s.slotIndex].tile.industry}`);
        break;
      }
      default:
        keys.add(a.type);
    }
  }
  return [...keys].sort();
}

function handle(req) {
  switch (req.cmd) {
    case 'init':
      init(req.spec, req.tables);
      // Every `apply` response carries a `sourcing` array (oracles/protocol.md).
      return { sourcing_check: true };
    case 'apply':
      if (!state) throw new Error('no game: send init first');
      return apply(req.action);
    case 'cards':
      return cards();
    case 'deal':
      deal(req.hands, req.deck);
      return {};
    case 'snapshot':
      return { snapshot: snapshot() };
    case 'legal':
      return { keys: legal() };
    default:
      throw new Error(`unknown cmd ${req.cmd}`);
  }
}

const rl = createInterface({ input: process.stdin, crlfDelay: Infinity });
for await (const line of rl) {
  if (!line.trim()) continue;
  let res;
  try {
    res = { ok: true, ...handle(JSON.parse(line)) };
  } catch (e) {
    res = { ok: false, error: String(e?.message ?? e) };
  }
  process.stdout.write(JSON.stringify(res) + '\n');
}
