// SPDX-License-Identifier: Apache-2.0
// @ts-check
import { readFile, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'astro/config';
import { chartNotices } from './scripts/client-notices.mjs';

// Static site: every page is plain HTML at build time. Interactive pages
// (dashboard, replay viewer) will add client-side islands only where needed.
export default defineConfig({
  // Set SITE_URL when the domain is chosen; it drives canonical and Open Graph URLs.
  site: process.env.SITE_URL || 'https://example.org',
  output: 'static',
  vite: { build: { license: { fileName: 'licenses/client-dependencies.md' } } },
  integrations: [{
    name: 'launch-headers',
    hooks: {
      'astro:build:start': () => {
        const root = new URL('../', import.meta.url);
        if (existsSync(new URL('lab/outreach/CLAIMS.md', root))) {
          const python = fileURLToPath(new URL('.venv/bin/python', root));
          execFileSync(python, ['scripts/site_progress.py', 'validate'], { cwd: root, stdio: 'inherit' });
        }
      },
      'astro:build:done': async ({ dir }) => {
        const headers = process.env.PUBLIC_LAUNCHED === 'true' ? '/*\n  X-Content-Type-Options: nosniff\n' : '/*\n  X-Robots-Tag: noindex, nofollow\n';
        await writeFile(new URL('_headers', dir), headers);
        const noticesPath = new URL('licenses/client-dependencies.md', dir);
        const licenses = chartNotices(fileURLToPath(new URL('.', import.meta.url)), await readFile(noticesPath, 'utf8'));
        await writeFile(noticesPath, licenses);
        const plot = await readFile(new URL('node_modules/@observablehq/plot/LICENSE', import.meta.url), 'utf8');
        if (!licenses.includes(plot.trim())) throw new Error('Missing full Observable Plot license in client artifact');
      },
    },
  }],
  trailingSlash: 'never',
  build: { format: 'file' },
  image: { layout: 'constrained', responsiveStyles: true },
});
