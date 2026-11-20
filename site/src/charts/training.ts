// SPDX-License-Identifier: Apache-2.0
// Client-side charts for the training record page, drawn with Observable Plot.
// Each chart renders into an element with a matching data-chart attribute and
// redraws when its container changes width.
import * as Plot from '@observablehq/plot';

type Era = 'canal' | 'rail';
interface Window {
  lo: number; hi: number; policies: number; seat_games: number;
  samples: Record<string, Record<string, number>>;
  actions: Record<string, number> | null;
  loans: Record<Era, number | null>; income: Record<Era, number | null>;
  unflipped: Record<Era, number | null>; network: Record<Era, number | null>;
  built: Record<string, number | null>; scored_twice: number | null;
  sources: Record<string, number> | null;
}
export interface TrainingRecord {
  meta: { latest_update: number; window: number; score_window: number };
  scores: [number, number, string][];
  score_mean: { lo: number; hi: number; n: number; mean: number }[];
  windows: Window[];
}

const css = (name: string) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

export const ACTIONS = ['Build', 'Develop', 'Sell', 'Canal', 'Rail', 'Loan', 'Scout', 'Pass'];
export const ACTION_COLORS = ['#5e4d3b', '#6f5c47', '#816c54', '#937d62', '#a68f72', '#d9b66a', '#4a3d30', '#e8743b'];
export const SOURCES = ['iron', 'brewery', 'coal', 'manufacturer', 'pottery', 'cotton', 'canal', 'rail', 'Merchant'];
export const SOURCE_LABELS: Record<string, string> = {
  iron: 'Iron works', brewery: 'Breweries', coal: 'Coal mines', manufacturer: 'Manufacturers', pottery: 'Potteries',
  cotton: 'Cotton mills', canal: 'Canal links', rail: 'Rail links', Merchant: 'Merchant bonuses',
};
export const SOURCE_COLORS = ['#e8743b', '#d9b66a', '#cdbfa8', '#9c8a72', '#7e6e5a', '#655748', '#5b6e73', '#8fa6ac', '#4a3d30'];
export const INDUSTRY_LABELS: Record<string, string> = {
  coal: 'Coal mines', iron: 'Iron works', brewery: 'Breweries', manufacturer: 'Manufacturers', pottery: 'Potteries', cotton: 'Cotton mills',
};
export const INDUSTRY_COLORS: Record<string, string> = {
  coal: '#cdbfa8', iron: '#e8743b', brewery: '#d9b66a', manufacturer: '#9c8a72', pottery: '#7e6e5a', cotton: '#655748',
};

function base(width: number, height: number) {
  return {
    width,
    height,
    marginLeft: 44,
    marginRight: 16,
    marginBottom: 40,
    style: { background: 'transparent', color: css('--color-muted'), fontFamily: css('--font-body'), fontSize: '12px', overflow: 'visible' },
    x: { label: 'Model update →', labelAnchor: 'right' as const, labelArrow: 'none' as const,  ticks: 5, tickFormat: 'd' },
    y: { grid: true, labelArrow: 'none' as const },
  };
}
const tip = () => ({ fill: css('--color-ink'), stroke: css('--color-rule') });
const mid = (w: { lo: number; hi: number }) => (w.lo + w.hi + 1) / 2;
const label = (w: { lo: number; hi: number }) => `Updates ${w.lo}–${w.hi}`;

const charts: Record<string, (d: TrainingRecord, width: number, era?: Era) => SVGSVGElement | HTMLElement> = {
  score(d, width) {
    const accent = css('--color-accent');
    return Plot.plot({
      ...base(width, width < 640 ? 300 : 420),
      ariaLabel: 'Victory points of every AI seat by model update, with the mean per window',
      y: { grid: true, label: 'Victory points',  labelArrow: 'none' },
      color: { domain: ['training', 'evaluation', 'other'], range: [css('--color-paper'), css('--color-brass'), css('--color-muted')] },
      marks: [
        Plot.dot(d.scores, { x: (s) => s[0], y: (s) => s[1], fill: (s) => s[2], r: width < 640 ? 1.1 : 1.5, fillOpacity: 0.28 }),
        Plot.line(d.score_mean, { x: mid, y: 'mean', stroke: css('--color-ink'), strokeWidth: 6 }),
        Plot.line(d.score_mean, { x: mid, y: 'mean', stroke: accent, strokeWidth: 2.5 }),
        Plot.tip(d.score_mean, Plot.pointerX({ ...tip(), x: mid, y: 'mean', title: (m) => `${label(m)}\nMean ${m.mean.toFixed(1)} VP\n${m.n.toLocaleString('en')} seat scores` })),
      ],
    });
  },

  actions(d, width) {
    const rows = d.windows.filter((w) => w.actions).flatMap((w) => ACTIONS.map((a) => ({ x: mid(w), w, action: a, share: w.actions![a], count: w.samples.actions![a] })));
    return Plot.plot({
      ...base(width, width < 640 ? 280 : 380),
      ariaLabel: 'Share of each main action by model update',
      y: { label: 'Share of main actions (%)', domain: [0, 100], grid: true, labelArrow: 'none' },
      color: { domain: ACTIONS, range: ACTION_COLORS },
      marks: [
        Plot.areaY(rows, { x: 'x', y: 'share', fill: 'action', order: ACTIONS }),
        Plot.ruleX(rows, Plot.pointerX({ x: 'x', stroke: css('--color-paper'), strokeOpacity: 0.5 })),
        Plot.tip(rows, Plot.pointerX(Plot.stackY({ ...tip(), x: 'x', y: 'share', z: 'action', order: ACTIONS, title: (r) => `${label(r.w)}\n${r.action}: ${r.share.toFixed(1)}%\n${r.count.toLocaleString('en')} main actions` }))),
      ],
    });
  },

  sources(d, width) {
    const rows = d.windows.filter((w) => w.sources).flatMap((w) => SOURCES.map((s) => ({ w, x1: w.lo, x2: w.hi + 1, source: s, count: w.samples.sources![s], vp: Math.max(0, w.sources![s] ?? 0) })));
    return Plot.plot({
      ...base(width, width < 640 ? 300 : 400),
      ariaLabel: 'Victory points by scoring source per window',
      y: { label: 'Victory points (best game per policy)', grid: true, labelArrow: 'none' },
      color: { domain: SOURCES, range: SOURCE_COLORS },
      marks: [
        Plot.rectY(rows, { x1: 'x1', x2: 'x2', y: 'vp', fill: 'source', order: SOURCES, insetLeft: 2, insetRight: 2 }),
        Plot.tip(rows, Plot.pointer(Plot.stackY({ ...tip(), x: (r) => (r.x1 + r.x2) / 2, y: 'vp', z: 'source', order: SOURCES, title: (r) => `${label(r.w)}\n${SOURCE_LABELS[r.source]}: ${r.vp.toFixed(1)} VP\n${r.count.toLocaleString('en')} policies` }))),
      ],
    });
  },

  built(d, width) {
    const keys = Object.keys(INDUSTRY_LABELS);
    const rows = d.windows.flatMap((w) => keys.map((k) => ({ x: mid(w), w, industry: INDUSTRY_LABELS[k], count: w.samples.built![k], v: w.built[k] })).filter((r) => r.v != null));
    return Plot.plot({
      ...base(width, 300),
      ariaLabel: 'Industries built per seat-game by model update',
      y: { label: null, grid: true, labelArrow: 'none', domain: [0, 4], ticks: 4 },
      color: { domain: keys.map((k) => INDUSTRY_LABELS[k]), range: keys.map((k) => INDUSTRY_COLORS[k]) },
      marks: [
        Plot.line(rows, { x: 'x', y: 'v', stroke: 'industry', strokeWidth: 2 }),
        Plot.tip(rows, Plot.pointer({ ...tip(), x: 'x', y: 'v', title: (r) => `${label(r.w)}\n${r.industry}: ${r.v!.toFixed(2)}\n${r.count.toLocaleString('en')} seat-games` })),
      ],
    });
  },

  twice(d, width) {
    const rows = d.windows.filter((w) => w.scored_twice != null).map((w) => ({ x: mid(w), w, v: w.scored_twice!, count: w.samples.scored_twice!.value! }));
    return single(rows, width, 'Industry VP scored again in the Rail era', null, [0, 8]);
  },

  loans: (d, width, era) => eraChart(d, width, era!, 'loans', 'Loans per seat-game', [0, 4]),
  income: (d, width, era) => eraChart(d, width, era!, 'income', 'Income level at era end', [-8, 8]),
  unflipped: (d, width, era) => eraChart(d, width, era!, 'unflipped', 'Unflipped industries at era end', [0, 4]),
  network: (d, width, era) => eraChart(d, width, era!, 'network', 'Link VP per link built', [0, 4]),
};

function single(rows: { x: number; w: Window; v: number; count: number }[], width: number, aria: string, yLabel: string | null, domain: number[], color = css('--color-accent')) {
  return Plot.plot({
    ...base(width, 260),
    ariaLabel: aria,
    y: { label: yLabel, grid: true, labelArrow: 'none', domain, ticks: 4 },
    marks: [
      domain[0]! < 0 ? Plot.ruleY([0], { stroke: css('--color-muted') }) : null,
      Plot.line(rows, { x: 'x', y: 'v', stroke: color, strokeWidth: 2.5 }),
      Plot.dot(rows, { x: 'x', y: 'v', fill: color, r: (r) => (r.count < 20 ? 0 : 2.5) }),
      Plot.dot(rows, { x: 'x', y: 'v', stroke: color, fill: css('--color-ink'), r: (r) => (r.count < 20 ? 3 : 0) }),
      Plot.tip(rows, Plot.pointerX({ ...tip(), x: 'x', y: 'v', title: (r) => `${label(r.w)}\n${r.v.toFixed(2)}\n${r.count.toLocaleString('en')} seat-games` })),
    ],
  });
}

function eraChart(d: TrainingRecord, width: number, era: Era, key: 'loans' | 'income' | 'unflipped' | 'network', yLabel: string, domain: number[]) {
  const rows = d.windows.filter((w) => w[key][era] != null).map((w) => ({ x: mid(w), w, v: w[key][era]!, count: w.samples[key]![era]! }));
  return single(rows, width, `${yLabel}, ${era} era`, null, domain, era === 'canal' ? css('--color-brass') : css('--color-accent'));
}

export function mount(data: TrainingRecord) {
  const nodes = document.querySelectorAll<HTMLElement>('[data-chart]');
  const draw = (node: HTMLElement) => {
    const width = Math.round(node.clientWidth);
    if (!width || node.dataset.width === String(width)) return;
    node.dataset.width = String(width);
    const make = charts[node.dataset.chart!];
    if (!make) return;
    node.replaceChildren(make(data, width, node.dataset.era as Era | undefined));
  };
  const observer = new ResizeObserver((entries) => entries.forEach((e) => draw(e.target as HTMLElement)));
  nodes.forEach((n) => {
    draw(n);
    observer.observe(n);
  });
}
