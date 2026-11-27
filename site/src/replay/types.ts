// SPDX-License-Identifier: Apache-2.0
// Shape of site/public/data/replays/*.json (written by site/scripts/export_replay.py).

/** [owner seat, industry index, level, flipped 0|1] or null for an empty slot. */
export type Tile = [number, number, number, number] | null;

export interface ReplayState {
  era: 'canal' | 'rail';
  round: number;
  slots: Tile[];
  /** Owner seat of each built link, or null. */
  links: (number | null)[];
  /** [victory points, income level, money] per seat. */
  players: [number, number, number][];
}

export interface ReplayPly {
  p: number;
  text: string;
  end: boolean;
  target: { slot: number } | { link: number } | null;
  /** Recomputed policy probabilities, when available. */
  candidates?: { text: string; p: number; played: boolean }[];
}

export interface ReplayData {
  meta: {
    game: string;
    category: string;
    exported: string;
    featured: number;
    seats: string[];
    final_vp: number[];
    ranking: number[];
  };
  board: {
    locations: { name: string; id: string; kind: string; slots: number[] }[];
    /** [location a, location b, canal 0|1, rail 0|1] */
    links: [number, number, number, number][];
    industries: string[];
  };
  plies: ReplayPly[];
  /** states[i] is the position after i moves; states[0] is the start. */
  states: ReplayState[];
}
