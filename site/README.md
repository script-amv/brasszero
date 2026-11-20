# BrassZero website

The public site: landing page (`/`) and training record (`/training`); training log and replay viewer next.
Copy comes from `docs/outreach/LANDING.md`; every number must map to a claim in `docs/outreach/CLAIMS.md`.

```bash
cd site
npm install
npm run dev       # http://localhost:4321
npm run build     # type-check, then static HTML in dist/
npm run preview   # serve dist/
```

## Architecture

| Decision | Choice | Why |
| --- | --- | --- |
| Framework | Astro, static output | Content-first pages ship as plain HTML with no JavaScript by default. Dashboard and replay pages can add interactive islands only where needed, in React if we reuse code from `gui/web`. |
| Content | Markdown in `src/content/`, validated by a schema in `src/content.config.ts` | Copy is edited as text, not in components. The build fails if a section is missing a field, an image path is wrong, or a fact has no claim ID. The training log will be another collection. |
| Styling | Plain CSS with a short token list (`src/styles/tokens.css`) and scoped component styles | Five colours, two fonts, three spacing values. No CSS framework to learn or update. |
| Fonts | Self-hosted via Fontsource (Cormorant Garamond, Work Sans) | No requests to Google, so no visitor IP addresses leave the site, and no third-party outage can break the type. |
| Images | Originals in `src/assets/art/`, resized and converted to WebP at build time by `astro:assets` | Each visitor downloads a size that fits their screen; the whole landing page transfers about 0.7 MB. |
| Hosting | Any static host (Cloudflare Pages, Netlify or GitHub Pages) | `dist/` is plain files. No server, no database, nothing to patch. |
| Charts | Observable Plot, drawn in the browser on `/training` only | A charting library from the D3 authors, built for exploratory and research charts. Charts redraw to their container width, so they stay legible on phones; the landing page still ships no JavaScript. |
| Chart data | `public/data/training-record.json`, made by `scripts/export_training_record.py` | One reproducible export from the replay archive, also offered to readers as a download. It holds scores and window aggregates only: no seeds, actions or weights. |
| Types | TypeScript, `astro/tsconfigs/strictest`; `npm run build` runs `astro check` first | Errors stop the build instead of reaching the site. |

## Components

- `Hero`: full-viewport painting with a centred kicker, heading and paragraph.
- `Stats`: key numbers with short captions on a light band. Its top row carries the dated training status (label in kicker style, checkpoint in accent, date), from `status` in `src/content/home.md`.
- `Section`: kicker, heading and Markdown body at a readable width, with an optional `Figure` beside it. The page alternates the figure side: left, right, left, right.
  Text and figure align to the top; when the text is shorter than the figure it centres against it.
- `Figure`: outlined image with a caption.
- `Roadmap`: milestone steps: state label, numbered title and note. Add `milestones` to a section's front matter.
- Sections are separated by the same hairline divider as list items.
- Accent orange (`--color-accent`) marks kickers, Roman numerals and key numbers; brass marks figure outlines and links.
- Numbered lists in section Markdown render with upright Roman numerals.
- Lines are used only as dividers (between list items and above the footer).

## Adding or changing a section

Create or edit a file in `src/content/sections/`. The file name becomes the anchor (`#roadmap`).

```markdown
---
order: 6
kicker: Roadmap
title: Each test is written down before it runs
figure:
  image: ../../assets/art/os-lichfield.jpg
  alt: Describe what the image shows
  caption: Artist, Title, year.
---
Body text. A numbered list (`1. **Title.** text`) renders with Roman numerals.
```

The hero, status line and key facts are in `src/content/home.md`. Update `status.updated` and `status.checkpoint` with every training-log entry.

## Updating the training record

Start the local GUI server (it reduces the replay archive and finishes the strategy analysis), then:

```bash
python3 site/scripts/export_training_record.py --url http://localhost:8771
cd site && npm run build
```

Headings and captions on `/training` are computed from the export, so they update with it. Check `docs/outreach/CLAIMS.md` (C-09, C-16, C-17) against the new numbers before publishing.

## Images

All art is public domain; credits are listed in `src/content/home.md` and shown in the footer. Never add Roxley Games artwork (see `docs/outreach/RELEASE.md`).

## Before launch

- Set `SITE_URL` in the build environment to the real domain (canonical and Open Graph links use it).
- Set `contactEmail` in `src/site.config.ts`; the contact button stays hidden until then.
- Add the training log, dashboard and replay pages, then the hero buttons that link to them.
