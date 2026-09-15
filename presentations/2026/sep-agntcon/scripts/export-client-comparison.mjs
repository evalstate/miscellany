import { chromium } from 'playwright';
import { parse } from '@slidev/parser';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import path from 'node:path';
const base = process.env.SLIDEV_URL ?? 'http://localhost:3031';
const out = path.resolve('recordings');
await mkdir(out, { recursive: true });
const markdown = await readFile('slides.md', 'utf8');
const deck = await parse(markdown);
const slide = deck.slides.findIndex(s => s.content.includes('<ClientAdoptionComparison')) + 1;
if (!slide) throw new Error('Comparison slide not found');
const browser = await chromium.launch();
try {
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
  await page.goto(`${base}/${slide}?capture=1`, { waitUntil: 'networkidle' });
  const chart = page.locator(`.slidev-page-${slide} [data-role="comparison-svg"]`);
  await chart.waitFor(); await page.evaluate(() => document.fonts.ready);
  await page.addStyleTag({ content: 'html,body,#app,#page-root,.slidev-slide-container { background:white!important; } * { cursor:none!important; } .slidev-nav-controls {display:none!important;}' });
  await page.screenshot({ path: path.join(out, 'client-adoption-contact-sheet.png') });
  const svg = await chart.evaluate(el => {
    const clone = el.cloneNode(true);
    clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
    clone.setAttribute('width', '1600'); clone.setAttribute('height', '900');
    clone.setAttribute('style', 'font-family:Arial,Helvetica,sans-serif;font-variant-numeric:tabular-nums;background:white');
    return new XMLSerializer().serializeToString(clone);
  });
  await writeFile(path.join(out, 'client-adoption-contact-sheet.svg'), svg);
  await writeFile(path.join(out, 'client-adoption-contact-sheet.capture.json'), JSON.stringify({
    slide, source: 'Completed reference snapshot through September 14, not partial September 15 dashboard',
    privateAggregateWarning: 'Review before public sharing. No upload performed.',
    minValidCalls: 100, last7Window: ['2026-09-08', '2026-09-14'],
    sourceSha256: createHash('sha256').update(await readFile('data/client-adoption-comparison/daily.csv')).digest('hex'),
    svgSha256: createHash('sha256').update(svg).digest('hex'),
  }, null, 2) + '\n');
  console.log(`Saved PNG and native SVG contact sheet from slide ${slide} to ${out}`);
} finally { await browser.close(); }
