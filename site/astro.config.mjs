// SPDX-License-Identifier: Apache-2.0
// @ts-check
import { writeFile } from 'node:fs/promises';
import { defineConfig } from 'astro/config';

// Static site: every page is plain HTML at build time. Interactive pages
// (dashboard, replay viewer) will add client-side islands only where needed.
export default defineConfig({
  // Set SITE_URL when the domain is chosen; it drives canonical and Open Graph URLs.
  site: process.env.SITE_URL || 'https://example.org',
  output: 'static',
  integrations: [{
    name: 'launch-headers',
    hooks: {
      'astro:build:done': async ({ dir }) => {
        const headers = process.env.PUBLIC_LAUNCHED === 'true' ? '/*\n  X-Content-Type-Options: nosniff\n' : '/*\n  X-Robots-Tag: noindex, nofollow\n';
        await writeFile(new URL('_headers', dir), headers);
      },
    },
  }],
  trailingSlash: 'never',
  build: { format: 'file' },
  image: { layout: 'constrained', responsiveStyles: true },
});
