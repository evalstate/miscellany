import { chromium } from 'playwright';
import { parse } from '@slidev/parser';
import { readFile } from 'node:fs/promises';
import assert from 'node:assert/strict';
const base = process.env.SLIDEV_URL ?? 'http://localhost:3031';
const deck = await parse(await readFile('slides.md', 'utf8'));
const slideFor = chart => deck.slides.findIndex(s => s.content.includes(`chart="${chart}"`)) + 1;
const adoption = JSON.parse(await readFile('data/protocol-adoption/data.json', 'utf8'));
const quality = JSON.parse(await readFile('data/tool-quality-version/data.json', 'utf8'));
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
const errors = [];
page.on('pageerror', e => { if (!e.message.includes('Wake Lock permission')) errors.push(e.message); });
async function open(chart, capture = true) {
  await page.goto(`${base}/${slideFor(chart)}${capture ? '?capture=1' : ''}`);
  await page.locator(`[data-chart="${chart}"]`).waitFor();
  if (capture) await page.waitForFunction(name => window.__deckCapture?.chart === name, chart);
  await page.evaluate(() => document.fonts.ready);
}
try {
  await open('protocol-adoption');
  assert.equal(await page.locator('[data-role="controls"]').count(), 0);
  assert.equal(await page.evaluate(() => window.__deckCapture.state.speed), 4);
  await page.evaluate(() => window.__deckCapture.renderAt(3750));
  assert.equal(await page.evaluate(() => window.__deckCapture.state.position), 24.5);
  const headline = await page.locator('.pa-stage svg').textContent();
  assert(headline.includes('2026-07-28 Protocol Adoption (Tool Calls)'));
  assert(!headline.includes('Plot viewport:') && !headline.includes('Static snapshot:'));
  for (const series of adoption.series) {
    await page.evaluate(client => window.__deckCapture.configure({ client, viewport: 'seven', speed: 1 }), series.id);
    for (const position of [0, 5.5, 20.25, 49]) {
      await page.evaluate(ms => window.__deckCapture.renderAt(ms), position / 49 * 30000);
      const state = await page.evaluate(() => window.__deckCapture.state);
      const index = Math.floor(position);
      assert.equal(state.index, index); assert.equal(state.dailyShare, series.rows[index].daily_share); assert.equal(state.rollingShare, series.rows[index].rolling_share);
      assert.deepEqual(state.counts, series.rows[index].counts);
      const markerIndices = await page.locator('circle[data-role="point"]').evaluateAll(els => els.map(el => Number(el.dataset.index)));
      assert(markerIndices.every(i => i <= index));
      if (position === 0) assert.equal(state.rollingShare, null);
    }
  }
  await page.evaluate(() => window.__deckCapture.configure({ client: 'overall', viewport: 'full' }));
  await page.evaluate(() => window.__deckCapture.renderAt(30000));
  assert.equal(await page.locator('[data-role="daily"]').textContent().then(s => s.trim()), '54.9%');
  assert.equal(await page.locator('[data-role="rolling"]').textContent().then(s => s.trim()), '49.3%');
  assert.equal(await page.locator('.pa-stage text').first().evaluate(el => getComputedStyle(el).fontSize), '54px');
  // Real embedded controls, data downloads, pause and scrub behaviour.
  await open('protocol-adoption', false);
  assert.equal(await page.evaluate(() => window.__deckCapture === undefined), true);
  await page.locator('.protocol-adoption-chart [data-action="provenance"]').click();
  const download = page.getByRole('link', { name: 'data.json', exact: true });
  assert.equal((await page.request.get(new URL(await download.getAttribute('href'), base).href)).status(), 200);
  await page.getByRole('button', { name: 'Close data & provenance' }).click();
  await page.locator('.protocol-adoption-chart [data-action="reset"]').click();
  await page.locator('.protocol-adoption-chart [data-action="play"]').click();
  await page.waitForFunction(() => Number(document.querySelector('.protocol-adoption-chart')?.dataset.position) > .1);
  await page.locator('.protocol-adoption-chart [data-action="play"]').click();
  const frozen = await page.locator('.protocol-adoption-chart').getAttribute('data-position');
  await page.waitForTimeout(150); assert.equal(await page.locator('.protocol-adoption-chart').getAttribute('data-position'), frozen);

  const comparisonSlide = deck.slides.findIndex(s => s.content.includes('<ClientAdoptionComparison')) + 1;
  await page.goto(`${base}/${comparisonSlide}?capture=1`);
  const comparison = page.locator(`.slidev-page-${comparisonSlide} .client-adoption-comparison`);
  await comparison.waitFor();
  assert.equal(await comparison.getAttribute('data-through'), '2026-09-14');
  assert.equal(await comparison.getAttribute('data-min-valid-calls'), '100');
  assert.deepEqual(await comparison.locator('[data-role="client-panel"]').evaluateAll(els => els.map(el => el.dataset.client)), ['claude-code','Anthropic/ClaudeAI','chat-ui-mcp','openai-mcp','openai-mcp (Codex)','codex-mcp-client']);
  assert.deepEqual(await comparison.locator('[data-role="last7"]').allTextContents(), ['Last 7: 95.2%','Last 7: 99.3%','Last 7: 100.0%','Last 7: 28.4%','Last 7: 15.1%','Last 7: 0.0%']);
  assert.equal(await comparison.locator('[data-role="activity"][data-scale="independent"]').count(), 6);
  assert.equal(await comparison.locator('[data-action="provenance"]').count(), 0);
  const normalized = JSON.parse(await readFile('data/client-adoption-comparison/normalized.json', 'utf8'));
  for (const series of normalized.series) {
    const panel = comparison.locator(`[data-client="${series.id}"]`);
    for (const field of ['daily_share','trailing7_share']) {
      const values = await panel.locator(`[data-series="${field}"] [data-role="rate-point"]`).evaluateAll(els => els.map(el => Number(el.dataset.value)));
      assert.deepEqual(values, series.rows.filter(row => row[field] !== null).map(row => row[field]));
    }
  }
  await open('tool-quality-version');
  const eventTime = date => 1200 + (Date.parse(date) - Date.parse('2026-08-25T00:00:00Z')) / 86400000 * 1500;
  for (const event of quality.server_events) {
    await page.evaluate(ms => window.__deckCapture.renderAt(ms), eventTime(event.date) + 225);
    const state = await page.evaluate(() => window.__deckCapture.state);
    assert.equal(state.serverVersion, event.version);
    assert.equal(state.activeChangeVersion, event.version === '0.4.14' ? '0.4.13' : event.version);
    const guide = page.locator(`.tqv-version-marker[data-kind="server"][data-version="${event.version}"]`);
    assert.equal(await guide.getAttribute('data-visible'), 'true');
    assert.equal(await guide.getAttribute('data-landed'), 'false');
  }
  await page.evaluate(ms => window.__deckCapture.renderAt(ms), eventTime('2026-08-27T00:00:00Z'));
  const narrow = await page.locator('[data-ribbon="server"] [data-version="0.4.14"]').evaluate(el => ({ start: +el.dataset.start, end: +el.dataset.end, width: +el.querySelector('rect').getAttribute('width') }));
  assert.equal(narrow.end - narrow.start, 6 * 3600000); assert(Math.abs(narrow.width - 1110 / 28) < .001);
  await page.evaluate(() => window.__deckCapture.renderAt(25000.25));
  const svg = await page.locator('.tqv-stage svg').evaluate(el => el.outerHTML);
  await page.evaluate(() => window.__deckCapture.renderAt(27000));
  await page.evaluate(() => window.__deckCapture.renderAt(25000.25));
  assert.equal(await page.locator('.tqv-stage svg').evaluate(el => el.outerHTML), svg);
  await page.evaluate(() => window.__deckCapture.renderAt(33700));
  assert.equal((await page.locator('[data-rate="claude"]').textContent()).trim(), '1.28%');
  assert.equal((await page.locator('[data-rate="others"]').textContent()).trim(), '5.24%');
  assert.equal(await page.locator('[data-change-note]').evaluate(el => getComputedStyle(el).opacity), '1');
  assert.equal(await page.locator('[data-axes]').evaluate(el => getComputedStyle(el).opacity), '1');
  await open('tool-quality-version', false);
  await page.locator('[data-play]').click();
  await page.waitForFunction(() => Number(document.querySelector('[data-tool-quality-version]')?.dataset.elapsed) > 100);
  await page.locator('[data-play]').click();
  const paused = await page.locator('[data-tool-quality-version]').getAttribute('data-elapsed');
  await page.waitForTimeout(150); assert.equal(await page.locator('[data-tool-quality-version]').getAttribute('data-elapsed'), paused);
  await page.locator('[data-play]').click();
  await page.evaluate(() => document.activeElement.blur()); await page.keyboard.press('ArrowLeft');
  await page.waitForFunction(() => !document.querySelector('[data-tool-quality-version]') || document.querySelector('[data-tool-quality-version]').dataset.state !== 'playing');
  await page.goto(`${base}/print`);
  await page.locator('.protocol-adoption-chart').waitFor();
  assert.equal(await page.locator('.protocol-adoption-chart').getAttribute('data-index'), '49');
  assert.equal(await page.locator('[data-tool-quality-version]').getAttribute('data-elapsed'), '33700');
  assert.equal(await page.locator('.pa-controls,.tqv-interface').count(), 0);
  assert.deepEqual(errors, []);
  console.log('Native charts: data values, fractional reveal, no future markers, controls, downloads, exact version boundaries, guide effects, deterministic replay, navigation cleanup and print passed.');
} finally { await browser.close(); }
