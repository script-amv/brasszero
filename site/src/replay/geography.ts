// SPDX-License-Identifier: Apache-2.0
// Board layout for the replay viewer: real longitude and latitude of each town,
// so the drawing is a map of the Midlands rather than a copy of any game board.
// Merchants lie far outside the region; they are pulled in toward the map edge
// in their true direction. Farm breweries are not real places: they sit beside
// the towns they connect to.

const LON_LAT: Record<string, [number, number]> = {
  belper: [-1.481, 53.024],
  derby: [-1.476, 52.922],
  leek: [-2.026, 53.105],
  'stoke-on-trent': [-2.18, 53.003],
  stone: [-2.15, 52.902],
  uttoxeter: [-1.865, 52.898],
  stafford: [-2.117, 52.806],
  'burton-on-trent': [-1.636, 52.802],
  cannock: [-2.031, 52.691],
  tamworth: [-1.695, 52.634],
  walsall: [-1.962, 52.586],
  wolverhampton: [-2.148, 52.587],
  coalbrookdale: [-2.493, 52.638],
  dudley: [-2.103, 52.505],
  kidderminster: [-2.248, 52.389],
  worcester: [-2.22, 52.193],
  birmingham: [-1.878, 52.476],
  coventry: [-1.512, 52.408],
  nuneaton: [-1.468, 52.523],
  redditch: [-1.944, 52.307],
  'farm-north': [-2.2, 52.705],
  'farm-south': [-2.4, 52.3],
  // merchants, pulled toward the edge in their true direction
  shrewsbury: [-2.72, 52.708],
  gloucester: [-2.26, 52.06],
  oxford: [-1.42, 52.11],
  warrington: [-2.42, 53.2],
  nottingham: [-1.22, 52.97],
};

const MIN_LON = -2.85;
const MAX_LON = -1.08;
const MIN_LAT = 51.98;
const MAX_LAT = 53.28;
const SHRINK = Math.cos((52.6 * Math.PI) / 180); // degrees of longitude are shorter here

export const BOARD_WIDTH = 1000;
const SCALE = BOARD_WIDTH / ((MAX_LON - MIN_LON) * SHRINK);
export const BOARD_HEIGHT = Math.round((MAX_LAT - MIN_LAT) * SCALE);

export function position(id: string): [number, number] {
  const ll = LON_LAT[id];
  if (!ll) throw new Error(`No map position for ${id}`);
  return [(ll[0] - MIN_LON) * SHRINK * SCALE, (MAX_LAT - ll[1]) * SCALE];
}

export const INDUSTRY_CODES: Record<string, string> = {
  cotton: 'Ct',
  manufacturer: 'Mf',
  pottery: 'Po',
  coal: 'Co',
  iron: 'Ir',
  brewery: 'Br',
};
