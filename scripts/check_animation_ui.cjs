const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');

(async () => {
  const qa = path.resolve('data/qa');
  const jobId = JSON.parse(fs.readFileSync(path.join(qa, 'animation_check.json'), 'utf8')).job_id;
  const legacyId = 'c95a16acbe094270b8aa96fe963eff2c';
  const cpuId = '6671062584ec4c12be2237b13a3c6e44';
  const browser = await chromium.launch({ headless: true, channel: 'msedge' });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));
    await page.goto('http://127.0.0.1:8765/');
    await page.getByRole('button', { name: /場と軌道/ }).click();
    await page.getByLabel('結果ジョブ').selectOption(jobId);
    await page.getByRole('heading', { name: '3D場と代表軌道' }).waitFor();
    await page.locator('.three-canvas canvas').waitFor();
    await page.getByRole('button', { name: '電場ベクトル', exact: true }).click();
    const canvas = await page.locator('.three-canvas canvas').elementHandle();
    await page.getByLabel('帯電時刻').focus();
    await page.getByLabel('帯電時刻').press('Home');
    await page.waitForFunction(() => document.querySelector('.time-select select').value === '1');
    await page.getByLabel('繰り返す').uncheck();
    await page.getByLabel('再生速度').selectOption('4');
    await page.getByRole('button', { name: 'アニメーションを再生' }).click();
    await page.getByRole('button', { name: '再生を一時停止' }).waitFor();
    await page.waitForFunction(() => document.querySelector('.time-select select').value === '4');
    await page.getByRole('button', { name: 'アニメーションを再生' }).waitFor();
    assert(await page.evaluate(el => el === document.querySelector('.three-canvas canvas'), canvas), 'Field playback recreated the camera canvas');
    await page.screenshot({ path: path.join(qa, 'animation_field.png'), fullPage: true });

    await page.getByRole('button', { name: '粒子の飛行', exact: true }).click();
    await page.getByLabel('軌道のRF位相').waitFor();
    assert((await page.locator('.animation-time').innerText()).includes('ns'));
    assert((await page.locator('.particle-legend').innerText()).includes('電子 6'));
    await page.getByRole('button', { name: 'アニメーションを再生' }).click();
    await page.waitForFunction(() => Number(document.querySelector('.animation-time').dataset.progress) > 0.15);
    await page.getByRole('button', { name: '再生を一時停止' }).click();
    const paused = await page.locator('.animation-time').getAttribute('data-progress');
    await page.waitForTimeout(180);
    assert.equal(await page.locator('.animation-time').getAttribute('data-progress'), paused);
    await page.screenshot({ path: path.join(qa, 'animation_particles.png'), fullPage: true });
    assert(await page.evaluate(el => el === document.querySelector('.three-canvas canvas'), canvas), 'Particle playback recreated the camera canvas');
    await page.getByRole('button', { name: '先頭へ戻る' }).click();
    assert.equal(await page.locator('.animation-time').getAttribute('data-progress'), '0.0000');
    await page.getByLabel('粒子の再生位置').focus();
    await page.getByLabel('粒子の再生位置').press('End');
    assert.equal(await page.locator('.animation-time').getAttribute('data-progress'), '1.0000');
    await page.getByRole('button', { name: 'アニメーションを再生' }).click();
    await page.getByRole('button', { name: '再生を一時停止' }).waitFor();
    await page.getByRole('button', { name: 'アニメーションを再生' }).waitFor();
    assert.equal(await page.locator('.animation-time').getAttribute('data-progress'), '1.0000');
    await page.getByLabel('繰り返す').check();
    await page.getByRole('button', { name: '先頭へ戻る' }).click();
    await page.getByRole('button', { name: 'アニメーションを再生' }).click();
    await page.waitForFunction(() => Number(document.querySelector('.animation-time').dataset.progress) > 0.85);
    await page.waitForFunction(() => Number(document.querySelector('.animation-time').dataset.progress) < 0.2);
    assert(await page.getByRole('button', { name: '再生を一時停止' }).isVisible());
    await page.getByRole('button', { name: '再生を一時停止' }).click();

    await page.getByLabel('結果ジョブ').selectOption(legacyId);
    await page.getByRole('heading', { name: '3D場と代表軌道' }).waitFor();
    await page.getByRole('button', { name: '粒子の飛行', exact: true }).click();
    await page.getByText(/旧データ：経路に沿った表示再生/).waitFor();
    assert(!(await page.locator('.animation-time').innerText()).includes('ns'));
    await page.getByRole('button', { name: 'アニメーションを再生' }).click();
    await page.waitForFunction(() => Number(document.querySelector('.animation-time').dataset.progress) > 0.02);
    await page.getByRole('button', { name: '再生を一時停止' }).click();
    await page.screenshot({ path: path.join(qa, 'animation_legacy.png'), fullPage: true });

    const emptyRoute = `**/api/jobs/${cpuId}/results`;
    await page.route(emptyRoute, async route => {
      const response = await route.fetch();
      const data = await response.json();
      data.trajectories = [];
      data.trajectory_fields = {};
      await route.fulfill({ response, json: data });
    });
    await page.getByLabel('結果ジョブ').selectOption(cpuId);
    await page.getByText(/この保存時刻には代表軌道がありません/).waitFor();
    assert(await page.getByRole('button', { name: 'アニメーションを再生' }).isDisabled());
    await page.unroute(emptyRoute);

    const delayedRoute = `**/api/jobs/${jobId}/results?step=1`;
    await page.route(delayedRoute, async route => {
      const response = await route.fetch();
      await new Promise(resolve => setTimeout(resolve, 700));
      try { await route.fulfill({ response }); } catch (e) { /* owner was unmounted */ }
    });
    await page.getByLabel('結果ジョブ').selectOption(jobId);
    await page.getByLabel('帯電時刻').waitFor();
    const pending = page.waitForRequest(request => request.url().endsWith(`${jobId}/results?step=1`));
    await page.getByLabel('帯電時刻').focus();
    await page.getByLabel('帯電時刻').press('Home');
    await pending;
    await page.getByLabel('結果ジョブ').selectOption(cpuId);
    await page.getByText(/旧データ：経路に沿った表示再生/).waitFor();
    await page.waitForTimeout(850);
    assert.equal(await page.getByLabel('結果ジョブ').inputValue(), cpuId);
    assert.equal(await page.getByLabel('軌道のRF位相').count(), 0);
    await page.unroute(delayedRoute);

    await page.getByLabel('結果ジョブ').selectOption(jobId);
    await page.getByLabel('帯電時刻').waitFor();
    await page.setViewportSize({ width: 390, height: 844 });
    await page.getByRole('button', { name: '粒子の飛行', exact: true }).click();
    await page.getByRole('button', { name: '電場ベクトル', exact: true }).click();
    await page.screenshot({ path: path.join(qa, 'animation_mobile.png'), fullPage: true });
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'Mobile horizontal overflow');
    await page.getByRole('button', { name: 'アニメーションを再生' }).click();
    await page.getByRole('button', { name: /衝突統計/ }).click();
    assert.equal(await page.locator('.animation-controls').count(), 0);
    assert.deepEqual(errors, []);
    fs.writeFileSync(path.join(qa, 'animation_ui_check.json'), JSON.stringify({
      errors, fieldPlayback: 'passed', timedParticlePlayback: 'passed', pauseSeekSpeedLoop: 'passed',
      cameraCanvasPreserved: 'passed', legacyPlayback: 'passed', emptyPaths: 'passed',
      delayedFrameJobSwitch: 'passed', mobile: 'passed', tabCleanup: 'passed',
    }, null, 2));
    console.log('Field/particle playback, pause/seek/speed/loop, persistent canvas, legacy/empty results, job switch and mobile: passed');
  } finally { await browser.close(); }
})().catch(e => { console.error(e); process.exit(1); });
