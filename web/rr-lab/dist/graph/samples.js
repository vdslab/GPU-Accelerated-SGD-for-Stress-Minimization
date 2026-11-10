import { rng } from '../sgd/random.js';
export function demoGraph() {
  const n = 240,
    edges = [],
    random = rng(19);
  for (let i = 0; i < n; i++) {
    edges.push([i, (i + 1) % n]);
    for (let j = i + 2; j < n; j++)
      if (Math.floor(i / 40) === Math.floor(j / 40) && random() < 0.075) edges.push([i, j]);
  }
  return { n, edges };
}

export const SAMPLES = [
  { id: 'community', name: 'Community (240 vertices)', file: null, method: 'full' },
  {
    id: 'powergrid',
    name: 'USpowerGrid (4,941 vertices)',
    file: 'USpowerGrid.mtx',
    method: 'full',
  },
  {
    id: 'luxembourg',
    name: 'Luxembourg OSM (114,599 vertices)',
    file: 'luxembourg_osm.mtx',
    method: 'sparse',
  },
];
