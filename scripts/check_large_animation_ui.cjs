const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');

(async () => {
  const jobId = process.argv[2];
  if (!jobId) throw new Error('Usage: node scripts/check_large_animation_ui.cjs <1000-path job id>');
  const qa = path.resolve('data/qa');
  fs.mkdirSync(qa, { recursive: true });
  const browser = await chromium.launch({ headless: true, channel: 'msedge' });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.addInitScript(() => {
      window.largeAnimationMetrics = { maxInstances: 0 };
      const original = WebGL2RenderingContext.prototype.drawElementsInstanced;
      WebGL2RenderingContext.prototype.drawElementsInstanced = function (...args) {
        window.largeAnimationMetrics.maxInstances = Math.max(window.largeAnimationMetrics.maxInstances, args[4]);
        return original.apply(this, args);
      };
    });
    await page.goto('http://127.0.0.1:8765/');
    const count = page.getByLabel(/アニメーション用の代表軌道数/);
    await count.waitFor();
    assert.equal(await count.getAttribute('max'), '1000');
    for (const value of [1000, 1001, 12]) {
      const checked = page.waitForResponse(response => response.url().endsWith('/api/geometry') &&
        response.request().method() === 'POST' &&
        response.request().postDataJSON()?.numerics?.representative_trajectories === value);
      await count.fill(String(value));
      assert.equal((await checked).status(), value === 1001 ? 422 : 200);
    }
    const preview = await (await page.request.get(`http://127.0.0.1:8765/api/jobs/${jobId}/results`)).json();
    assert.equal(preview.trajectories.length, 1000);
    assert.equal(Object.keys(preview.trajectory_fields).length, 1, 'Use a single-phase job to test 1000 simultaneous particles');
    await page.getByRole('button', { name: /場と軌道/ }).click();
    await page.getByLabel('結果ジョブ').selectOption(jobId);
    await page.getByRole('button', { name: '粒子の飛行', exact: true }).click();
    await page.getByText('イオン 500', { exact: true }).waitFor();
    await page.getByText('電子 500', { exact: true }).waitFor();
    const canvas = await page.locator('.three-canvas canvas').elementHandle();
    await page.getByRole('button', { name: 'アニメーションを再生' }).click();
    await page.waitForFunction(() => Number(document.querySelector('.animation-time').dataset.progress) > 0.1);
    const measure = () => page.evaluate(() => new Promise(resolve => {
      const intervals = [];
      let last, start;
      function tick(now) {
        if (start === undefined) start = now;
        if (last !== undefined) intervals.push(now - last);
        last = now;
        if (now - start < 1500) return requestAnimationFrame(tick);
        const sorted = [...intervals].sort((a, b) => a - b);
        resolve({ frames: intervals.length, observedFps: intervals.length * 1000 / (now - start),
          p95FrameMs: sorted[Math.floor(sorted.length * 0.95)] });
      }
      requestAnimationFrame(tick);
    }));
    const desktop = await measure();
    assert(desktop.frames >= 5, 'Playback did not keep rendering');
    assert.equal(await page.evaluate(() => window.largeAnimationMetrics.maxInstances), 1000);
    await page.getByRole('button', { name: '再生を一時停止' }).click();
    const paused = await page.locator('.animation-time').getAttribute('data-progress');
    await page.waitForTimeout(120);
    assert.equal(await page.locator('.animation-time').getAttribute('data-progress'), paused);
    await page.getByLabel('粒子の再生位置').focus();
    await page.getByLabel('粒子の再生位置').press('End');
    assert.equal(await page.locator('.animation-time').getAttribute('data-progress'), '1.0000');
    await page.getByRole('button', { name: '先頭へ戻る' }).click();
    assert.equal(await page.locator('.animation-time').getAttribute('data-progress'), '0.0000');
    assert(await page.evaluate(element => element === document.querySelector('.three-canvas canvas'), canvas));
    await page.setViewportSize({ width: 390, height: 844 });
    await page.getByRole('button', { name: 'アニメーションを再生' }).click();
    const mobile = await measure();
    await page.getByRole('button', { name: '再生を一時停止' }).click();
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    await page.screenshot({ path: path.join(qa, 'animation_1000.png'), fullPage: true });
    await page.getByRole('button', { name: /衝突統計/ }).click();
    assert.equal(await page.locator('.animation-controls').count(), 0);
    assert.deepEqual(errors, []);
    const report = { jobId, paths: 1000, instanceDraw: 'passed', settingsLimit: 'passed',
      pauseSeekReset: 'passed', persistentCanvas: 'passed', desktop, mobile, errors };
    fs.writeFileSync(path.join(qa, 'animation_1000_ui_check.json'), JSON.stringify(report, null, 2));
    console.log(JSON.stringify(report));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exit(1); });
