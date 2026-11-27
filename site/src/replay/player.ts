// SPDX-License-Identifier: Apache-2.0
// Client-side replay controls. The page is fully rendered at one position at
// build time; this script loads the game data and updates the board, players,
// last decision and move log in place as the reader steps through the game.
import { INDUSTRY_CODES } from './geography';
import type { ReplayData } from './types';

const $ = <T extends Element>(root: ParentNode, sel: string) => root.querySelector<T>(sel);
const $$ = <T extends Element>(root: ParentNode, sel: string) => [...root.querySelectorAll<T>(sel)];
const sign = (v: number) => (v > 0 ? `+${v}` : String(v));

export async function startReplay() {
  const found = document.querySelector<HTMLElement>('[data-replay]');
  if (!found) return;
  const root: HTMLElement = found;
  const data = (await (await fetch(root.dataset.replay!)).json()) as ReplayData;
  const { meta, plies, states, board } = data;
  const N = plies.length;
  const featured = meta.featured;
  let pos = Number(root.dataset.start);
  let timer: number | undefined;

  const scrub = $<HTMLInputElement>(root, '[data-scrub]')!;
  const playBtn = $<HTMLButtonElement>(root, '[data-go="play"]')!;
  const speed = $<HTMLSelectElement>(root, '[data-speed]')!;
  const pauseFeatured = $<HTMLInputElement>(root, '[data-pause-featured]')!;
  const slotEls = new Map($$<SVGGElement>(root, '[data-slot]').map((g) => [Number(g.dataset.slot), g]));
  const linkEls = $$<SVGPathElement>(root, '[data-link]');
  const logItems = $$<HTMLLIElement>(root, '[data-ply]');
  const logList = $<HTMLOListElement>(root, '[data-log]')!;

  // Round boundaries: the first position of each new round or era.
  const roundStarts = states.map((_, i) => i).filter((i) => i === 0 || states[i]!.round !== states[i - 1]!.round || states[i]!.era !== states[i - 1]!.era);

  const setAttr = (el: Element, name: string, value: string | number | null | undefined) =>
    value === null || value === undefined ? el.removeAttribute(name) : el.setAttribute(name, String(value));

  function render() {
    const s = states[pos]!;
    const last = pos > 0 ? plies[pos - 1]! : null;
    const target = last?.target ?? null;

    for (const [slot, g] of slotEls) {
      const t = s.slots[slot];
      setAttr(g, 'data-owner', t ? t[0] : null);
      setAttr(g, 'data-flipped', t ? t[3] : null);
      g.classList.toggle('hl', !!target && 'slot' in target && target.slot === slot);
      const [code, lvl] = $$<SVGTextElement>(g, 'text');
      code!.textContent = t ? INDUSTRY_CODES[board.industries[t[1]]!]! : '';
      lvl!.textContent = t ? String(t[2]) : '';
    }
    linkEls.forEach((path, i) => {
      setAttr(path, 'data-owner', s.links[i]);
      setAttr(path, 'data-era', s.era);
      path.classList.toggle('hl', !!target && 'link' in target && target.link === i);
    });

    const order = [0, 1, 2, 3].sort((a, b) => s.players[b]![0] - s.players[a]![0]);
    const tbody = $<HTMLTableSectionElement>(root, '[data-players]')!;
    for (const i of order) {
      const row = $<HTMLTableRowElement>(tbody, `[data-seat="${i}"]`)!;
      const [vp, income, money] = s.players[i]!;
      const cells = $$<HTMLTableCellElement>(row, 'td');
      cells[0]!.textContent = String(vp);
      cells[1]!.textContent = sign(income);
      cells[2]!.textContent = `£${money}`;
      tbody.appendChild(row);
    }

    $(root, '[data-decision]')!.textContent = last ? last.text : 'Start of game';
    const who = $(root, '[data-who]')!;
    who.innerHTML = '';
    if (last) {
      const sw = document.createElement('span');
      sw.className = 'sw';
      sw.style.background = `var(--seat-${last.p})`;
      who.append(sw, `${meta.seats[last.p]} · move ${pos} of ${N}`);
    } else who.append('No moves yet');
    const cands = $<HTMLElement>(root, '[data-candidates]')!;
    if (last?.candidates?.length) {
      cands.replaceChildren(...last.candidates.map((c) => {
        const row = document.createElement('p');
        row.className = 'cand';
        row.textContent = `${c.text}${c.played ? ' (played)' : ''}: ${Math.round(c.p * 100)}%`;
        return row;
      }));
    }

    const era = s.era === 'canal' ? 'Canal' : 'Rail';
    $(root, '[data-where]')!.innerHTML = `Move <strong>${pos}</strong> of ${N} · ${era} era · round ${s.round}`;
    scrub.value = String(pos);
    scrub.style.setProperty('--pos', `${(pos / N) * 100}%`);

    for (const li of logItems) {
      const i = Number(li.dataset.ply);
      li.classList.toggle('current', i === pos - 1);
      li.classList.toggle('future', i >= pos);
    }
    const current = logItems.find((li) => Number(li.dataset.ply) === pos - 1);
    if (current) {
      const top = current.offsetTop - logList.offsetTop - logList.clientHeight / 2;
      logList.scrollTo({ top, behavior: 'auto' });
    }
  }

  function go(to: number) {
    pos = Math.max(0, Math.min(N, to));
    render();
  }

  function stop() {
    if (timer !== undefined) window.clearTimeout(timer);
    timer = undefined;
    playBtn.textContent = 'Play';
    playBtn.setAttribute('aria-label', 'Play');
  }

  function tick() {
    if (pos >= N) return stop();
    go(pos + 1);
    // Skip over turn-end markers so each step shows a real move.
    while (pos < N && plies[pos - 1]!.end) go(pos + 1);
    if (pauseFeatured.checked && plies[pos - 1]!.p === featured && !plies[pos - 1]!.end) return stop();
    timer = window.setTimeout(tick, Number(speed.value));
  }

  function step(dir: 1 | -1) {
    let to = pos + dir;
    while (to > 0 && to < N && plies[to - 1]!.end) to += dir;
    go(to);
  }

  root.addEventListener('click', (event) => {
    const button = (event.target as Element).closest<HTMLButtonElement>('button');
    if (!button) return;
    if (button.dataset.goto) { stop(); return go(Number(button.dataset.goto)); }
    const action = button.dataset.go;
    if (!action) return;
    if (action === 'play') {
      if (timer !== undefined) return stop();
      if (pos >= N) go(0);
      playBtn.textContent = 'Pause';
      playBtn.setAttribute('aria-label', 'Pause');
      timer = window.setTimeout(tick, 150);
      return;
    }
    stop();
    if (action === 'first') go(0);
    if (action === 'last') go(N);
    if (action === 'prev') step(-1);
    if (action === 'next') step(1);
    if (action === 'prev-round') go([...roundStarts].reverse().find((i) => i < pos) ?? 0);
    if (action === 'next-round') go(roundStarts.find((i) => i > pos) ?? N);
  });
  scrub.addEventListener('input', () => { stop(); go(Number(scrub.value)); });
  render();
}
