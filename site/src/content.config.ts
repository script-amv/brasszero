// SPDX-License-Identifier: Apache-2.0
import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';

// Every number on the site must map to an approved claim in
// docs/outreach/CLAIMS.md. `claim` records which one.

const home = defineCollection({
  loader: glob({ pattern: 'home.md', base: './src/content' }),
  schema: ({ image }) =>
    z.object({
      kicker: z.string(),
      title: z.string(),
      status: z.object({
        label: z.string(),
        updated: z.coerce.date(),
        checkpoint: z.number().int(),
      }),
      image: image(),
      imageAlt: z.string(),
      facts: z.array(
        z.object({ value: z.string(), label: z.string(), claim: z.string().regex(/^C-\d{2}$/) }),
      ),
      credits: z.array(z.string()),
    }),
});

const sections = defineCollection({
  loader: glob({ pattern: '*.md', base: './src/content/sections' }),
  schema: ({ image }) =>
    z.object({
      order: z.number().int(),
      kicker: z.string(),
      title: z.string(),
      figure: z
        .object({ image: image(), alt: z.string(), caption: z.string() })
        .optional(),
      milestones: z
        .array(z.object({ title: z.string(), state: z.enum(['done', 'active', 'planned']), note: z.string() }))
        .optional(),
    }),
});

export const collections = { home, sections };
