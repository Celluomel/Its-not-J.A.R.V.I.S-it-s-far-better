// Run against Vite with Playwright and sharp available on NODE_PATH.
// All API responses are isolated fixtures, never the user's live cognitive state.
const { chromium } = require('playwright');
const sharp = require('sharp');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');

async function pixels(canvas) {
  return sharp(await canvas.screenshot()).raw().toBuffer({ resolveWithObject: true });
}
function difference(a, b) {
  assert.equal(a.data.length, b.data.length);
  let changed = 0;
  for (let i = 0; i < a.data.length; i += a.info.channels) {
    if (Math.abs(a.data[i] - b.data[i]) > 5) changed++;
  }
  return changed;
}
async function main() {
  const browser = await chromium.launch({ headless: true,
    ...(process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : {}) });
  const output = path.resolve('artifacts/observatory');
  await fs.mkdir(output, { recursive: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1600, height: 1050 } });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.route('**/api/interface/**', route => {
      const endpoint = new URL(route.request().url()).pathname.split('/').at(-1);
      const responses = {
        status: { ready: true, busy: false, providers: {}, engines: {}, events: [], timings: {}, cycle: 42, uptime: 120 },
        network: { enabled: false, children: [] },
        'cognitive-health': { fetched_at: Date.now() / 1000, slow_cycle: 42,
          attention: { primary: 'semantic_memory', weights: { semantic_memory: .34 }, workspace_topic: 'Comparing evidence across memory and current perception' },
          mf: { dominant: 'curiosity', drive_vector: { curiosity: .82 }, needs_detected: 3 },
          ec: { cognitive_energy: .76, attention: .91 },
          global_workspace: { focus: 'Exploring a new observation', confidence: .67, active_hypotheses: ['a', 'b'] },
          long_horizon: { active_plans: 1, plans: [{ objective: 'Evaluate the observation against recalled evidence', completed: 2, steps: 4, next: 'memory_recall' }] },
          orchestrator: { available: true, running: true, cycle_count: 18, last_activity: 'idle_reflection', queued_events: 2,
            clock: { medium_in_sec: 43, slow_in_sec: 163, evolution_in_sec: 1640 },
            drives: { curiosity: .82, coherence: .67, energy: .76, social: .43, goal_progress: .5, homeostasis: .91 } } },
      };
      return route.fulfill({ json: responses[endpoint] || {} });
    });
    await page.goto(process.env.LUMINA_PREVIEW_URL || 'http://127.0.0.1:5173/next/');
    const panel = page.locator('.observatory');
    await panel.scrollIntoViewIfNeeded();
    const canvas = panel.locator('canvas');
    await canvas.waitFor();
    await page.waitForTimeout(1500);
    assert.equal(await page.locator('.cognition-scene').count(), 0, 'Old diagram must not render');
    assert.equal(await canvas.count(), 1);
    const a = await pixels(canvas);
    let lit = 0;
    for (let i = 0; i < a.data.length; i += a.info.channels) if (a.data[i] > 50) lit++;
    assert.ok(lit > a.info.width * a.info.height * .03, 'Canvas contains a visible sculpture');
    await page.waitForTimeout(700);
    assert.ok(difference(a, await pixels(canvas)) > 100, 'Live sculpture moves');
    await panel.getByRole('button', { name: 'Pause sculpture motion' }).click();
    await page.waitForTimeout(350);
    const paused = await pixels(canvas);
    await page.waitForTimeout(400);
    assert.ok(difference(paused, await pixels(canvas)) < 100, 'Pause stops motion');
    await panel.locator('.network-selector button').nth(3).click();
    assert.match(await panel.locator('.network-insight').innerText(), /Evaluate the observation/);
    assert.match(await panel.locator('.sculpture-caption').innerText(), /04 \/ Planning/i);
    const beforeDrag = await pixels(canvas);
    const box = await canvas.boundingBox();
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await page.mouse.down();
    await page.mouse.move(box.x + box.width / 2 + 130, box.y + box.height / 2 + 40, { steps: 12 });
    await page.mouse.up();
    assert.ok(difference(beforeDrag, await pixels(canvas)) > 100, 'Dragging rotates 3D geometry');
    await panel.getByRole('button', { name: 'Reset sculpture view' }).click();
    await panel.screenshot({ path: path.join(output, 'desktop.png') });
    for (const width of [390, 768]) {
      await page.setViewportSize({ width, height: 920 });
      await panel.scrollIntoViewIfNeeded();
      await page.waitForTimeout(400);
      const data = await pixels(canvas);
      assert.ok(data.info.width > 250, 'Responsive canvas retains usable width');
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false, 'No horizontal overflow');
      const overflow = await panel.locator('button, h2, h3, .network-insight p').evaluateAll(elements => elements.filter(el => el.scrollWidth > el.clientWidth + 1).map(el => el.textContent));
      assert.deepEqual(overflow, [], 'Text fits its containers');
      await panel.screenshot({ path: path.join(output, `width-${width}.png`) });
    }
    await page.emulateMedia({ reducedMotion: 'reduce' });
    await page.reload();
    await panel.scrollIntoViewIfNeeded();
    await canvas.waitFor();
    await page.waitForTimeout(700);
    const reduced = await pixels(canvas);
    await page.waitForTimeout(400);
    assert.ok(difference(reduced, await pixels(canvas)) < 100, 'Reduced-motion preference respected');
    assert.deepEqual(errors, [], 'No browser runtime errors');
    console.log('PASS: nonblank WebGL, live motion, pause, selection, rotation, reset, desktop/mobile layout, reduced motion. Screenshots:', output);
  } finally { await browser.close(); }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
