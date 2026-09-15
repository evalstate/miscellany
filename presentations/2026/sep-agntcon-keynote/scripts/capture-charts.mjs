import { chromium } from 'playwright';
import { parse } from '@slidev/parser';
import { readFile, mkdir, writeFile, rename, unlink } from 'node:fs/promises';
import { spawn, execFileSync } from 'node:child_process';
import { once } from 'node:events';
import path from 'node:path';
import { createHash } from 'node:crypto';

const args = Object.fromEntries(process.argv.slice(2).map(arg => {
  const [key, ...value] = arg.replace(/^--/, '').split('='); return [key, value.join('=') || true];
}));
if (args.help) {
  console.log('npm run capture:charts -- --chart=all|protocol-adoption|tool-quality-version|legacy-message-ratio|legacy-protocol-video --base=http://localhost:3030 --fps=30 --width=1920 --out=recordings --client=overall --viewport=full|seven --speed=4');
  process.exit(0);
}
const base = args.base ?? 'http://localhost:3030';
const fps = Number(args.fps ?? 30), width = Number(args.width ?? 1920), height = width * 9 / 16;
if (!Number.isInteger(fps) || fps < 1 || fps > 60 || !Number.isInteger(width) || width < 640 || width % 2 || !Number.isInteger(height) || height % 2) throw new Error('Use 1–60 FPS and even 16:9 dimensions, e.g. width 1280 or 1920.');
const requested = args.chart ?? 'all';
const charts = ['protocol-adoption', 'tool-quality-version', 'legacy-message-ratio', 'legacy-protocol-video'].filter(chart => requested === 'all' || chart === requested);
if (!charts.length) throw new Error('Unknown chart');
const out = path.resolve(String(args.out ?? 'recordings'));
await mkdir(out, { recursive: true });
const markdown = await readFile('slides.md', 'utf8');
const deck = await parse(markdown);
const leadMs = 1000, tailMs = 2500;
const browser = await chromium.launch({ headless: true });
try {
  for (const chart of charts) {
    const index = deck.slides.findIndex(slide => slide.content.includes(`chart="${chart}"`));
    if (index < 0) throw new Error(`No slide for ${chart}`);
    const slide = index + 1;
    const client = String(args.client ?? 'overall'), viewport = String(args.viewport ?? 'full');
    const captureSpeed = chart === 'protocol-adoption' ? Number(args.speed ?? 4) : 1;
    if (!Number.isFinite(captureSpeed) || captureSpeed <= 0) throw new Error('Speed must be finite and positive');
    const suffix = chart === 'protocol-adoption' ? `-${client}-${viewport}-${captureSpeed}x` : '';
    const name = `${chart}${suffix}`.replace(/[^a-zA-Z0-9_-]/g, '_');
    const output = path.join(out, `${name}.mp4`), temporary = path.join(out, `${name}.partial.mp4`);
    const page = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: 1, reducedMotion: 'no-preference' });
    const errors = [];
    page.on('pageerror', error => { if (!error.message.includes('Wake Lock permission')) errors.push(error.message); });
    await page.goto(`${base}/${slide}?capture=1`, { waitUntil: 'networkidle' });
    await page.waitForFunction(expected => window.__deckCapture?.chart === expected && window.__deckCapture.durationMs > 0, chart);
    await page.evaluate(() => document.fonts.ready);
    await page.addStyleTag({ content: 'html,body,#app,#page-root,.slidev-slide-container { background: white !important; } * { cursor: none !important; } .slidev-nav-controls { display:none !important; }' });
    if (chart === 'protocol-adoption') await page.evaluate(options => window.__deckCapture.configure(options), { client, viewport, speed: captureSpeed });
    const sourceDurationMs = await page.evaluate(() => window.__deckCapture.durationMs);
    const durationMs = sourceDurationMs / captureSpeed;
    const frameCount = Math.ceil((leadMs + durationMs + tailMs) / 1000 * fps);
    const encoderArgs = ['-hide_banner', '-loglevel', 'warning', '-y', '-f', 'image2pipe', '-framerate', String(fps), '-vcodec', 'png', '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-r', String(fps), temporary];
    const encoder = spawn('ffmpeg', encoderArgs, { stdio: ['pipe', 'ignore', 'pipe'] });
    let encoderLog = '', encoderError;
    encoder.stderr.on('data', data => { encoderLog = (encoderLog + data.toString()).slice(-10000); });
    encoder.on('error', error => { encoderError = error; });
    encoder.stdin.on('error', error => { encoderError = error; });
    const exit = new Promise(resolve => encoder.on('close', code => resolve(code)));
    let lastMs = -1, png;
    console.log(`${chart}: slide ${slide}, ${width}×${height}, ${fps} FPS, ${frameCount} frames`);
    try {
      for (let i = 0; i < frameCount; i++) {
        const ms = Math.min(durationMs, Math.max(0, i / fps * 1000 - leadMs));
        if (ms !== lastMs) {
          await page.evaluate(time => window.__deckCapture.renderAt(time), ms);
          png = await page.screenshot({ type: 'png' });
          lastMs = ms;
        }
        if (encoderError || encoder.exitCode !== null) throw encoderError ?? new Error(`ffmpeg exited: ${encoderLog}`);
        if (!encoder.stdin.write(png)) await once(encoder.stdin, 'drain');
        if (i % (fps * 5) === 0) console.log(`  ${i}/${frameCount}`);
      }
      encoder.stdin.end();
      const code = await exit;
      if (code !== 0 || encoderError) throw encoderError ?? new Error(`ffmpeg exited ${code}: ${encoderLog}`);
      if (errors.length) throw new Error(`Browser errors: ${errors.join('; ')}`);
      await rename(temporary, output);
      await page.screenshot({ path: path.join(out, `${name}-poster.png`) });
      const probe = JSON.parse(execFileSync('ffprobe', ['-v', 'error', '-show_streams', '-show_format', '-of', 'json', output], { encoding: 'utf8' }));
      const video = probe.streams.find(stream => stream.codec_type === 'video');
      if (video.codec_name !== 'h264' || video.width !== width || video.height !== height || video.pix_fmt !== 'yuv420p' || Number(video.nb_frames) !== frameCount) throw new Error('Encoded output failed validation');
      const dataFile = chart === 'legacy-protocol-video' ? 'data/legacy-protocol-video.provenance.json' : chart === 'legacy-message-ratio' ? 'data/legacy-message-ratio.provenance.json' : `data/${chart}/data.json`;
      const provenanceFile = ['legacy-message-ratio', 'legacy-protocol-video'].includes(chart) ? dataFile : `data/${chart}/provenance.json`;
      const metadata = {
        chart, slide, dimensions: [width, height], fps, frameCount, leadMs, sourceDurationMs, playbackDurationMs: durationMs, tailMs,
        settings: chart === 'protocol-adoption' ? { client, viewport, speed: captureSpeed } : { speed: 1 },
        capture: 'Deterministic native SVG frames, H.264 MP4 / yuv420p / faststart, no audio, controls or cursor',
        privateAggregateWarning: 'Review the underlying aggregate data before external sharing. No upload performed.',
        sourceDataSha256: createHash('sha256').update(await readFile(dataFile)).digest('hex'),
        provenanceSha256: createHash('sha256').update(await readFile(provenanceFile)).digest('hex'),
        deckSha256: createHash('sha256').update(markdown).digest('hex'),
        outputSha256: createHash('sha256').update(await readFile(output)).digest('hex'),
        finalState: await page.evaluate(() => window.__deckCapture.state), probe,
      };
      await writeFile(path.join(out, `${name}.capture.json`), JSON.stringify(metadata, null, 2) + '\n');
      console.log(`Saved ${output} (${probe.format.duration}s)`);
    } catch (error) {
      encoder.kill(); await unlink(temporary).catch(() => {}); throw error;
    } finally { await page.close(); }
  }
} finally { await browser.close(); }
