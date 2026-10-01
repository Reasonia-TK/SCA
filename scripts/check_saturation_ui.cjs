const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');

(async () => {
  const qa = path.resolve('data/qa');
  const cases = JSON.parse(fs.readFileSync(path.join(qa, 'saturation_api_check.json'), 'utf8'));
  const browser = await chromium.launch({ headless: true, channel: 'msedge' });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto('http://127.0.0.1:8765/');
    await page.getByRole('heading', { name: 'ホール内輸送・帯電', exact: true }).waitFor();
    const untilSaturation = page.waitForResponse(response => response.url().endsWith('/api/geometry') &&
      response.request().postDataJSON()?.numerics?.run_until === 'saturation');
    await page.getByRole('button', { name: /^飽和帯電まで/ }).click();
    const checked = await untilSaturation;
    assert.equal(checked.status(), 200);
    assert.equal(checked.request().postDataJSON().mode, 'self_consistent');
    assert.equal(await page.getByLabel('必要な連続安定窓数').inputValue(), '5');
    const edited = page.waitForResponse(response => response.url().endsWith('/api/geometry') &&
      Math.abs(response.request().postDataJSON()?.numerics?.saturation?.window_s - 12e-6) < 1e-15);
    await page.getByLabel('判定時間幅').fill('12');
    assert.equal((await edited).status(), 200);
    await page.setViewportSize({ width: 390, height: 844 });
    await page.waitForFunction(() => document.documentElement.scrollWidth <= innerWidth);
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'Settings overflow');
    await page.screenshot({ path: path.join(qa, 'saturation_settings_mobile.png'), fullPage: true });
    const untilTime = page.waitForResponse(response => response.url().endsWith('/api/geometry') &&
      response.request().postDataJSON()?.numerics?.run_until === 'time');
    await page.getByRole('button', { name: /^帯電なし/ }).click();
    assert.equal((await untilTime).status(), 200);
    assert.equal(await page.getByLabel('判定時間幅').count(), 0);
    assert(await page.getByLabel('帯電時間').isVisible());
    await page.getByRole('button', { name: '実行', exact: true }).click();
    for (const [reason, data] of Object.entries(cases)) {
      const card = page.locator('.job-card').filter({ has: page.getByRole('heading', { name: data.name, exact: true }) });
      assert((await card.innerText()).includes(reason === 'saturated' ? '飽和判定成立' : '終了・未飽和'));
    }
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'Job card overflow');
    await page.getByRole('button', { name: '場と軌道', exact: true }).click();
    await page.getByLabel('結果ジョブ').selectOption(cases.saturated.job_id);
    await page.getByRole('heading', { name: '表示時刻の飽和判定' }).waitFor();
    await page.getByText('飽和判定成立（指定許容差内）', { exact: true }).waitFor();
    const times = page.locator('.time-select select');
    const options = await times.locator('option').evaluateAll(list => list.map(option => option.value));
    await times.selectOption(options[0]);
    await page.getByText('飽和判定を継続中', { exact: true }).waitFor();
    await times.selectOption(options.at(-1));
    await page.getByText('飽和判定成立（指定許容差内）', { exact: true }).waitFor();
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'Result overflow');
    await page.screenshot({ path: path.join(qa, 'saturation_result_mobile.png'), fullPage: true });
    await page.setViewportSize({ width: 1440, height: 1100 });
    for (const [reason, label] of [['time_limit', '最大時間で終了（未飽和）'], ['update_limit', '最大更新回数で終了（未飽和）']]) {
      await page.getByLabel('結果ジョブ').selectOption(cases[reason].job_id);
      await page.getByText(label, { exact: true }).waitFor();
    }
    assert.deepEqual(errors, []);
    fs.writeFileSync(path.join(qa, 'saturation_ui_check.json'), JSON.stringify({
      modeSelection: 'passed', settings: 'passed', durationModePreserved: 'passed',
      completionReasons: 'passed', savedWindowDiagnostics: 'passed', mobile: 'passed', errors,
    }, null, 2));
    console.log('Saturation settings, completion reasons, snapshot diagnostics and mobile: passed');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exit(1); });
