// SPDX-License-Identifier: Apache-2.0
// Dumps the TS oracle's static data as a flat {key: value} JSON map to
// oracles/cases/ts-data.json, in the key format built by `our_flat_data()` in
// engine/tests/data.rs. Reads the vendored oracle in place (nothing is copied).
//
// Usage: node oracles/ts-adapter/dump-data.mjs   (Node >= 22.18: built-in type stripping)

import { existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { register } from 'node:module';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const src = join(here, '../vendor/BrassBirmingham/packages/engine/src');
const out = join(here, '../cases/ts-data.json');
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

const load = (f) => import(pathToFileURL(join(src, f)).href);
const { LOCATIONS, LINKS, LINK_EXTRA_ENDPOINTS, MERCHANTS } = await load('data/board.ts');
const { TILES } = await load('data/tiles.ts');
const market = await load('data/market.ts');
const income = await load('data/income.ts');
const cards = await load('data/cards.ts');
const { newGame } = await load('state.ts');
const { scoreEraLinks } = await load('era.ts');

const m = {};
const put = (k, v) => {
  if (k in m) throw new Error(`duplicate key ${k}`);
  m[k] = String(v);
};

for (const [id, loc] of Object.entries(LOCATIONS)) {
  put(`location.${id}.kind`, loc.region === 'farm' ? 'farm-brewery' : 'town');
  put(`location.${id}.slots`, loc.slots.map((s) => s.industries.join('+')).join('|'));
}
LINKS.forEach((l, i) => {
  const ends = [l.a, l.b, ...(LINK_EXTRA_ENDPOINTS[i] ?? [])].sort();
  put(`link.${ends.join('~')}`, `canal=${l.canal},rail=${l.rail}`);
});

// Merchant link VP lives in scoring code, not data: probe it by scoring one link from each
// merchant to a town with no tiles, so the only VP is the merchant end's.
const game = newGame(4, 1);
for (const [id, def] of Object.entries(MERCHANTS)) {
  put(`location.${id}.kind`, 'merchant');
  put(`merchant.${id}.tile_slots`, def.slots);
  put(`merchant.${id}.bonus`, `${def.bonus.type}:${def.bonus.amount}`);
  const linkIndex = LINKS.findIndex((l) => l.a === id || l.b === id);
  const scored = scoreEraLinks({
    ...game,
    board: { ...game.board, links: [{ linkIndex, player: 0, era: 'canal' }] },
  });
  put(`merchant.${id}.link_vp`, scored.players[0].vp - game.players[0].vp);
}
const pool = {};
for (const ms of Object.values(game.merchants)) for (const t of ms.tiles) pool[t] = (pool[t] ?? 0) + 1;
for (const [t, n] of Object.entries(pool)) put(`merchant_tiles_4p.${t}`, n);

for (const t of TILES) {
  const p = `tile.${t.industry}.${t.level}`;
  put(`${p}.count`, t.count);
  put(`${p}.cost_money`, t.costMoney);
  put(`${p}.cost_coal`, t.costCoal);
  put(`${p}.cost_iron`, t.costIron);
  put(`${p}.beer_to_sell`, t.beerToFlip);
  put(`${p}.vp`, t.vp);
  put(`${p}.link_vp`, t.linkIcons);
  put(`${p}.income`, t.incomeAdvance);
  put(`${p}.resources`, t.resourcesPlaced);
  put(`${p}.canal_ok`, !t.railEraOnly);
  put(`${p}.rail_ok`, t.railEraBuildable);
  put(`${p}.developable`, t.developable);
}

put('market.coal_prices', market.COAL_MARKET_PRICES.join(','));
put('market.coal_start', market.COAL_MARKET_INITIAL_FILLED);
put('market.coal_fallback', market.COAL_FALLBACK_PRICE);
put('market.iron_prices', market.IRON_MARKET_PRICES.join(','));
put('market.iron_start', market.IRON_MARKET_INITIAL_FILLED);
put('market.iron_fallback', market.IRON_FALLBACK_PRICE);
put('brewery_beer.canal', market.BREWERY_BARRELS.canal);
put('brewery_beer.rail', market.BREWERY_BARRELS.rail);

for (let s = 0; s <= 99; s++) {
  put(`income.level.${s}`, income.incomeLevelAt(s));
  put(`income.loan.${s}`, income.loanBacktrack(s));
}
for (let l = -10; l <= 30; l++) put(`income.top_space.${l}`, income.INCOME_LEVEL_SPACES(l)[1]);

const p0 = game.players[0];
put('setup.start_money', p0.money);
put('setup.hand_size', p0.hand.length);
put('setup.start_income_space', p0.incomeSpace);
put('wild.location', cards.WILD_LOCATION_COUNT);
put('wild.industry', cards.WILD_INDUSTRY_COUNT);

const deck = {};
for (const c of cards.buildDeck(4)) {
  let k;
  if (c.kind === 'location') k = `location.${c.location}`;
  else if (c.kind === 'industry') k = `industry.${c.industries.length === 2 ? 'cotton-or-manufacturer' : c.industries[0]}`;
  else throw new Error(`unexpected card in deck: ${c.kind}`);
  deck[k] = (deck[k] ?? 0) + 1;
}
for (const [k, n] of Object.entries(deck)) put(`deck.${k}`, n);

mkdirSync(dirname(out), { recursive: true });
const sorted = Object.fromEntries(Object.entries(m).sort(([a], [b]) => (a < b ? -1 : 1)));
writeFileSync(out, JSON.stringify(sorted, null, 1) + '\n');
console.log(`wrote ${Object.keys(sorted).length} keys to ${out}`);
