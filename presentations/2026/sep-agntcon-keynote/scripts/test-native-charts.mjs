import { chromium } from 'playwright';
import { parse } from '@slidev/parser';
import { readFile } from 'node:fs/promises';
import assert from 'node:assert/strict';
const base = process.env.SLIDEV_URL ?? 'http://localhost:3030';
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
  const deprecatedSlide = deck.slides.findIndex(s => s.content.includes('title="2026-07-28 MCP Communications" deprecated')) + 1;
  assert(deprecatedSlide > 0);
  await page.goto(`${base}/${deprecatedSlide}`);
  const staticDiagram = page.locator(`.slidev-page-${deprecatedSlide} .lpv-root`);
  await staticDiagram.waitFor();
  assert.equal(await staticDiagram.getAttribute('data-phase'), 'static');
  assert.equal(await staticDiagram.locator('.lpv-arrow').count(), 1);
  assert.equal(await staticDiagram.locator('[data-arrow="both"]').count(), 1);
  assert.deepEqual(await staticDiagram.locator('[data-deprecated="true"]').evaluateAll(els => els.map(el => el.dataset.capability)), ['roots', 'sampling']);
  assert.equal(await staticDiagram.locator('[data-role="deprecated-cross"]').count(), 2);
  assert.equal(await staticDiagram.locator('button,.lpv-highlight,.lpv-pulse').count(), 0);
  const staticSvg = await staticDiagram.innerHTML();
  await page.waitForTimeout(250);
  assert.equal(await staticDiagram.innerHTML(), staticSvg);
  await open('protocol-adoption');
  assert.equal(await page.locator('[data-role="controls"]').count(), 0);
  assert.equal(await page.evaluate(() => window.__deckCapture.state.speed), 4);
  await page.evaluate(() => window.__deckCapture.renderAt(3750));
  assert.equal(await page.evaluate(() => window.__deckCapture.state.position), 24.5);
  assert.equal(await page.locator('[data-role="endpoint"]').count(), 0);
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
  assert.equal(await page.locator('[data-role="endpoint"] polygon').count(), 2);
  assert.deepEqual((await page.locator('[data-role="endpoint"] text').allTextContents()).map(s => s.trim()).sort(), ['49.3%', '54.9%']);
  assert.equal((await page.locator('[data-role="half-reference"] text').textContent()).trim(), '50%');
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
  // Legacy ratio: independent source-clock contract, not timings read back from the component.
  await open('legacy-message-ratio');
  const ratio = page.locator(`[data-chart="legacy-message-ratio"] .message-ratio-chart`);
  const kinds = ['tool', 'initialize', 'listing', 'other'];
  const totals = { tool: 1, initialize: 27, listing: 39, other: 7 };
  const arrivals = [
    ...Array.from({ length: 1 }, (_, i) => ({ at: 450 + i * 105, kind: 'tool' })),
    ...Array.from({ length: 27 }, (_, i) => ({ at: 1650 + i * 105, kind: 'initialize' })),
    ...Array.from({ length: 39 }, (_, i) => ({ at: 4925 + i * 105, kind: 'listing' })),
    ...Array.from({ length: 7 }, (_, i) => ({ at: 9460 + i * 145, kind: 'other' })),
  ];
  const ratioDuration = 11580;
  assert.equal(await page.evaluate(() => window.__deckCapture.durationMs), ratioDuration);
  assert.equal(await ratio.locator('.message-ratio-controls, button, select, input').count(), 0);
  assert.deepEqual(await ratio.locator('.message-ratio-cell').evaluateAll(els => els.map(el => ({
    at: Number(el.dataset.revealAt), kind: el.dataset.kind,
  }))), arrivals);
  async function checkRatio(ms) {
    const elapsed = Math.max(0, Math.min(ratioDuration, ms));
    const shown = arrivals.filter(event => event.at <= elapsed);
    const counts = Object.fromEntries(kinds.map(kind => [kind, shown.filter(event => event.kind === kind).length]));
    const phase = elapsed === ratioDuration ? 'complete' : shown.at(-1)?.kind ?? 'ready';
    const mode = elapsed === ratioDuration ? 'complete' : elapsed === 0 ? 'idle' : 'playing';
    const state = await page.evaluate(ms => window.__deckCapture.renderAt(ms), ms);
    assert.deepEqual(state, { mode, phase, elapsedMs: elapsed, durationMs: ratioDuration,
      revealed: shown.length, counts, otherShown: shown.length - counts.tool }, `capture at ${ms}ms`);
    assert.equal(await ratio.getAttribute('data-phase'), phase);
    assert.equal(await ratio.getAttribute('data-state'), mode);
    assert.equal(await ratio.getAttribute('data-other-shown'), String(state.otherShown));
    assert.deepEqual(await ratio.locator('.message-ratio-cell.is-visible').evaluateAll(els => els.map(el => el.dataset.kind)), shown.map(event => event.kind));
    assert.deepEqual(await ratio.locator('.message-ratio-count').allTextContents(), kinds.map(kind => String(counts[kind])));
    assert.equal(await ratio.locator('.message-ratio-grid').getAttribute('aria-label'),
      `${counts.tool} tool call, ${counts.initialize} initialization, ${counts.listing} listing and ${counts.other} other message squares shown. Rounded aggregate ratio, not a session trace.`);
  }
  await checkRatio(0);
  for (const { at } of arrivals) {
    await checkRatio(at - .25);
    await checkRatio(at);
    await checkRatio(at + .25);
  }
  for (const ms of [ratioDuration - .25, ratioDuration, ratioDuration + 1000, -1]) await checkRatio(ms);

  // Sample actual rendered animation effects (not just identical inline declarations).
  // Two paints flush newly created CSS animations; paused source time must then be stable.
  async function ratioVisuals() {
    return ratio.evaluate(async root => {
      await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
      return [...root.querySelectorAll('.message-ratio-cell, .message-ratio-legend-content, .message-ratio-legend-content i')].map(el => {
        const style = getComputedStyle(el);
        return { className: el.className, opacity: style.opacity, transform: style.transform,
          color: style.color, background: style.backgroundColor, shadow: style.boxShadow,
          animation: style.animationName, delay: style.animationDelay, playState: style.animationPlayState,
          animations: el.getAnimations().map(a => ({ name: a.animationName, time: a.currentTime, state: a.playState })) };
      });
    });
  }
  for (const ms of [500.25, 1700.25, 4975.25, 9510.25, ratioDuration]) {
    await checkRatio(ms);
    const visual = await ratioVisuals();
    const animated = visual.filter(el => el.animation !== 'none');
    assert(animated.length > 0, `animations sampled at ${ms}`);
    // Non-filling legend effects disappear from getAnimations() once their source age ends.
    assert(animated.every(el => el.playState === 'paused' && el.animations.every(a => a.state === 'paused')), `paused effects at ${ms}`);
    assert(visual.slice(0, arrivals.filter(event => event.at <= ms).length).every(el => el.animations.length > 0), 'revealed cells retain their paused filling animation');
    if (ms === 500.25) assert(Number(visual[0].opacity) > 0 && Number(visual[0].opacity) < 1, 'arrival is sampled mid-animation');
    await page.waitForTimeout(1250); // Longer than the longest arrival animation.
    assert.deepEqual(await ratioVisuals(), visual, `wall-clock stability at ${ms}`);
    await checkRatio(ratioDuration);
    await checkRatio(0);
    await checkRatio(ms);
    assert.deepEqual(await ratioVisuals(), visual, `backward replay at ${ms}`);
    await checkRatio(ms);
    assert.deepEqual(await ratioVisuals(), visual, `repeated seek at ${ms}`);
  }
  await checkRatio(ratioDuration);
  assert.equal(await ratio.locator('.message-ratio-cell.is-visible').count(), 74);
  assert.deepEqual(await page.evaluate(() => window.__deckCapture.state.counts), totals);
  const finalCells = (await ratioVisuals()).slice(0, 74);
  assert(finalCells.every(cell => cell.opacity === '1'), 'all final squares fully visible');

  await open('legacy-message-ratio', false);
  assert.equal(await page.evaluate(() => window.__deckCapture === undefined), true);
  const primary = ratio.locator('.message-ratio-controls .is-primary');
  assert.equal(await ratio.getAttribute('data-state'), 'idle');
  assert.equal(await ratio.locator('.message-ratio-cell.is-visible').count(), 0);
  // Slidev's bottom toolbar overlaps these controls; exercise native keyboard activation.
  await primary.focus();
  await primary.press('Space');
  await page.waitForFunction(() => document.querySelector('.message-ratio-chart')?.dataset.phase === 'initialize');
  await primary.press('Space');
  await page.waitForFunction(() => document.querySelector('.message-ratio-chart')?.dataset.state === 'paused');
  assert.equal(await primary.textContent(), 'Resume');
  const pausedCount = await ratio.locator('.message-ratio-cell.is-visible').count();
  const pausedVisual = await ratioVisuals();
  await page.waitForTimeout(1250);
  assert.equal(await ratio.locator('.message-ratio-cell.is-visible').count(), pausedCount);
  assert.deepEqual(await ratioVisuals(), pausedVisual, 'live pause freezes CSS effects too');
  await primary.press('Space');
  await page.waitForFunction(count => document.querySelectorAll('.message-ratio-cell.is-visible').length > count, pausedCount);
  await ratio.getByRole('button', { name: 'Show all', exact: true }).press('Space');
  assert.equal(await ratio.getAttribute('data-state'), 'complete');
  assert.equal(await ratio.locator('.message-ratio-cell.is-visible').count(), 74);
  assert.deepEqual(await ratio.locator('.message-ratio-count').allTextContents(), ['1', '27', '39', '7']);
  assert.equal(await ratio.locator('.is-animated, .is-active').count(), 0);
  await page.waitForTimeout(250);
  assert.equal(await ratio.getAttribute('data-state'), 'complete');
  // Legacy protocol video: discover its ChartRecordingStage via the parsed deck.
  assert(slideFor('legacy-protocol-video') > 0, 'legacy protocol recording slide exists');
  await open('legacy-protocol-video');
  const video = page.locator('[data-chart="legacy-protocol-video"] .lpv-root');
  const videoDuration = 37650;
  const travel = 1275;
  const hold = 5000;
  const activations = ['tools', 'sampling', 'resources', 'elicitation', 'prompts', 'roots'];
  assert.equal(await page.evaluate(() => window.__deckCapture.durationMs), videoDuration);
  assert.equal(await video.locator('.lpv-controls, button, input, select, [data-role="controls"]').count(), 0);
  assert.deepEqual(await video.locator('svg text').allTextContents(),
    ['MCP Communications', 'Roots', 'Sampling', 'Elicitation', 'Tools', 'Resources', 'Prompts', 'Client', 'Server']);
  assert.equal(await video.locator('.lpv-arrow').count(), 2);
  for (const direction of ['right', 'left']) {
    const arrow = video.locator(`[data-arrow="${direction}"]`);
    assert.equal(await arrow.count(), 1);
    assert.equal(await arrow.evaluate(el => el.tagName), 'path');
    assert.equal(await arrow.evaluate(el => getComputedStyle(el).markerEnd), 'none');
    assert.equal((await arrow.getAttribute('d')).match(/M/g)?.length, 1, 'single seamless arrow path');
  }
  async function checkVideo(ms) {
    const elapsedMs = Math.max(0, Math.min(videoDuration, ms));
    const index = Math.min(5, Math.floor(elapsedMs / (travel + hold)));
    const offset = elapsedMs - index * (travel + hold);
    const phase = elapsedMs === videoDuration ? 'end' : offset < travel ? 'message' : 'hold';
    const direction = index % 2 ? 'left' : 'right';
    const flash = activations[index];
    const frameElapsedMs = phase === 'end' ? 0 : phase === 'message' ? offset : offset - travel;
    const expected = { phase, direction, flash, elapsedMs, progress: elapsedMs / videoDuration,
      frameElapsedMs, frameProgress: phase === 'end' ? 1 : frameElapsedMs / (phase === 'message' ? travel : hold),
      durationMs: videoDuration, receiver: direction === 'right' ? 'server' : 'client', mode: phase === 'end' ? 'end' : 'capture' };
    assert.deepEqual(await page.evaluate(ms => window.__deckCapture.renderAt(ms), ms), expected, `video at ${ms}ms`);
    assert.deepEqual(await page.evaluate(() => window.__deckCapture.state), expected);
    for (const key of ['phase', 'direction', 'flash', 'mode']) assert.equal(await video.getAttribute(`data-${key}`), expected[key]);
    assert.equal(await video.locator('.lpv-pulse').count(), phase === 'message' ? 1 : 0);
    assert.deepEqual(await video.locator('.lpv-highlight').evaluateAll(els => els.map(el => el.parentElement.dataset.capability ?? el.parentElement.dataset.actor)),
      phase === 'message' ? [] : [flash, expected.receiver]);
  }
  async function videoVisuals() {
    return video.evaluate(async root => {
      await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
      return { svg: root.querySelector('svg').outerHTML,
        elements: [...root.querySelectorAll('svg, svg *')].map(el => {
          const style = getComputedStyle(el);
          const box = el.getBoundingClientRect();
          return { tag: el.tagName, opacity: style.opacity, transform: style.transform,
            fill: style.fill, stroke: style.stroke, strokeWidth: style.strokeWidth,
            animation: style.animationName, duration: style.animationDuration,
            delay: style.animationDelay, playState: style.animationPlayState,
            box: { x: box.x, y: box.y, width: box.width, height: box.height },
            pulse: el.matches('.lpv-pulse') ? { cx: el.getAttribute('cx'), cy: el.getAttribute('cy'),
              matrix: [...['a', 'b', 'c', 'd', 'e', 'f']].map(k => el.getCTM()[k]) } : null };
        }) };
    });
  }
  for (let i = 0; i < activations.length; i++) {
    const start = i * (travel + hold);
    for (const boundary of [start, start + travel, start + travel + hold]) {
      for (const delta of [-.25, 0, .25]) await checkVideo(boundary + delta);
    }
    await checkVideo(start + travel + hold / 2);
    const highlights = await video.locator('.lpv-highlight').evaluateAll(els => els.map(el => {
      const s = getComputedStyle(el);
      return { opacity: s.opacity, duration: s.animationDuration, playState: s.animationPlayState };
    }));
    assert.deepEqual(highlights, Array(2).fill({ opacity: '1', duration: '5s', playState: 'paused' }), 'full highlight midway through each 5000ms hold');
    for (const offset of [431.375, travel + 123.625, travel + hold / 2]) {
      const ms = start + offset;
      await checkVideo(ms);
      const snapshot = await videoVisuals();
      if (offset < travel) {
        const pulse = snapshot.elements.find(el => el.pulse);
        const right = i % 2 === 0;
        const matrix = pulse.transform.match(/matrix\((.*)\)/)[1].split(',').map(Number);
        assert(Math.abs(matrix[4] - (right ? 627 + 346 * offset / travel : 973 - 346 * offset / travel)) < .01);
        assert.equal(matrix[5], right ? 448 : 546);
        assert.equal(pulse.playState, 'paused');
      }
      await page.waitForTimeout(100);
      assert.deepEqual(await videoVisuals(), snapshot, `wall-clock video stability at ${ms}`);
      await checkVideo(videoDuration);
      await checkVideo(ms);
      assert.deepEqual(await videoVisuals(), snapshot, `backwards video seek at ${ms}`);
      await checkVideo(0);
      await checkVideo(ms);
      assert.deepEqual(await videoVisuals(), snapshot, `fractional video replay at ${ms}`);
      await checkVideo(ms);
      assert.deepEqual(await videoVisuals(), snapshot, `repeated video seek at ${ms}`);
    }
  }
  await checkVideo(videoDuration + 1000);
  assert.deepEqual(await video.locator('.lpv-highlight').evaluateAll(els => els.map(el => getComputedStyle(el).opacity)), ['1', '1']);

  await open('legacy-protocol-video', false);
  assert.equal(await page.evaluate(() => window.__deckCapture === undefined), true);
  assert.equal(await video.getAttribute('data-mode'), 'ready');
  await page.waitForTimeout(1400);
  assert.equal(await video.getAttribute('data-mode'), 'ready', 'no automatic first run');
  assert.equal(await video.locator('.lpv-pulse, .lpv-highlight').count(), 0);
  await video.getByRole('button', { name: 'Play', exact: true }).press('Space');
  await page.waitForFunction(() => document.querySelector('.lpv-root')?.dataset.mode === 'playing');
  await page.waitForFunction(() => document.querySelector('.lpv-root')?.dataset.phase === 'hold');
  assert.equal(await video.getAttribute('data-flash'), 'tools');
  await video.getByRole('button', { name: 'Pause', exact: true }).press('Space');
  await page.waitForFunction(() => document.querySelector('.lpv-root')?.dataset.mode === 'paused');
  const pausedVideo = await videoVisuals();
  await page.waitForTimeout(1400);
  assert.deepEqual(await videoVisuals(), pausedVideo, 'live video pause freezes highlight');
  await video.getByRole('button', { name: 'Resume', exact: true }).press('Space');
  await page.waitForFunction(() => document.querySelector('.lpv-root')?.dataset.mode === 'playing');
  assert(await video.locator('.lpv-highlight').evaluateAll(els => els.every(el => getComputedStyle(el).animationPlayState === 'running')));
  // SPA navigation, not a reload: retained slide instances must stop their timers.
  await page.evaluate(() => document.activeElement.blur());
  await page.keyboard.press('ArrowRight');
  await page.waitForFunction(() => !document.querySelector('.lpv-root') || document.querySelector('.lpv-root').dataset.mode === 'ready');
  await page.waitForTimeout(1400);
  await page.keyboard.press('ArrowLeft');
  await video.waitFor();
  assert.equal(await video.getAttribute('data-mode'), 'ready', 'stopped on leave, no restart on return');
  assert.equal(await video.locator('.lpv-pulse, .lpv-highlight').count(), 0);
  console.log('Legacy protocol video: semantic boundaries, 37650ms duration, alternating travel/holds, computed-style replay, manual playback, pause/resume and leave cleanup passed.');
  await page.goto(`${base}/print`);
  await page.locator('.protocol-adoption-chart').waitFor();
  assert.equal(await page.locator('.protocol-adoption-chart').getAttribute('data-index'), '49');
  assert.equal(await page.locator('[data-tool-quality-version]').getAttribute('data-elapsed'), '33700');
  assert.equal(await page.locator('.pa-controls,.tqv-interface').count(), 0);
  assert.deepEqual(errors, []);
  console.log('Native charts: data values, fractional reveal, no future markers, controls, downloads, exact version boundaries, guide effects, deterministic replay, navigation cleanup, legacy ratio exact arrivals/counts, paused visual determinism, live controls and print passed.');
} finally { await browser.close(); }
