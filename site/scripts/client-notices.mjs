// SPDX-License-Identifier: Apache-2.0
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { createRequire } from 'node:module';
import { join } from 'node:path';

// Plot and D3 distribute prebundled ESM, so a bundler's module list alone cannot
// identify all incorporated packages. Include their installed runtime closure.
export function chartNotices(site, moduleNotices) {
  const seen = new Set();
  const supplemental = new Map();

  function visit(name, parent) {
    const search = createRequire(join(parent, 'package.json')).resolve.paths(name) ?? [];
    const directory = search.map((path) => join(path, name)).find((path) => existsSync(join(path, 'package.json')));
    if (!directory) throw new Error(`Missing installed chart dependency ${name}`);
    if (seen.has(directory)) return;
    seen.add(directory);

    const pkg = JSON.parse(readFileSync(join(directory, 'package.json'), 'utf8'));
    const heading = `## ${pkg.name} - ${pkg.version}`;
    if (!moduleNotices.split('\n').some((line) => line === heading || line.startsWith(heading + ' ('))) {
      const files = readdirSync(directory).filter((file) => /^(?:licen[cs]e|copying)(?:[.-].*)?$/i.test(file)).sort();
      if (!files.length) throw new Error(`Missing license text for ${pkg.name}@${pkg.version}`);
      const notices = files.map((file) => readFileSync(join(directory, file), 'utf8').trim());
      supplemental.set(heading, `${heading} (${pkg.license ?? 'see notice'})\n\n${notices.join('\n\n')}\n`);
    }
    for (const dependency of Object.keys(pkg.dependencies ?? {}).sort()) visit(dependency, directory);
  }

  visit('@observablehq/plot', site);
  const sections = [...supplemental.keys()].sort().map((key) => supplemental.get(key));
  return moduleNotices.trimEnd() + '\n\n# Additional chart package notices\n\n' +
    'The locked chart distribution includes prebundled modules. Its runtime package notices follow.\n\n' +
    sections.join('\n');
}
