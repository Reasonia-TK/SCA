const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const base = 'http://127.0.0.1:8765';
async function api(url, body) {
  const response = await fetch(base + '/api' + url, body === undefined ? {} : {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  });
  const result = await response.json();
  assert(response.ok, JSON.stringify(result));
  return result;
}
async function waitJob(url) {
  const deadline = Date.now() + 120000;
  while (Date.now() < deadline) {
    const job = await api(url);
    if (['completed', 'failed', 'cancelled'].includes(job.status)) {
      assert.equal(job.status, 'completed', JSON.stringify(job));
      return job;
    }
    await new Promise(resolve => setTimeout(resolve, 500));
  }
  throw new Error('Simulation timed out: ' + url);
}

(async () => {
  const qa = path.resolve('data/qa');
  fs.mkdirSync(qa, { recursive: true });
  const browser = await chromium.launch({ headless: true, channel: 'msedge' });
  const report = {}, errors = [];
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1050 }, acceptDownloads: true });
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(base);
    const tab = page.locator('.tabs').getByRole('button', { name: /IAEDF/ });
    await tab.click();
    const workspace = page.locator('.iaedf-workspace');
    await workspace.getByLabel('ケース名', { exact: true }).fill('IAEDF UI検証 · 1D CSV');
    await workspace.getByRole('button', { name: '小規模検証の設定', exact: true }).click();
    await workspace.getByLabel('圧力ケース [mTorr / カンマ区切り]', { exact: true }).fill('0, 10');
    await workspace.getByLabel('駆動波形の種類', { exact: true }).selectOption('csv');
    await workspace.getByRole('button', { name: '検証用5高調波を使用', exact: true }).click();
    const submit1d = page.waitForResponse(r => r.url().endsWith('/api/iaedf/jobs') && r.request().method() === 'POST');
    await workspace.getByRole('button', { name: 'IAEDFを計算', exact: true }).click();
    const source1d = await (await submit1d).json();
    const one = await waitJob('/iaedf/jobs/' + source1d.id);
    assert(one.summary.validation.passed);
    await workspace.getByRole('img', { name: '角度とエネルギーの相関分布', exact: true }).waitFor();
    await workspace.getByLabel('圧力ケース', { exact: true }).selectOption('1');
    await page.waitForResponse(r => r.url().includes('/distribution?') && r.url().includes('pressure_index=1'));
    await page.screenshot({ path: path.join(qa, 'iaedf_1d_desktop.png'), fullPage: true });
    const applied1d = page.waitForResponse(r => r.url().endsWith('/apply') && r.request().method() === 'POST');
    await workspace.getByRole('button', { name: 'ホール入口に適用', exact: true }).click();
    const appliedOne = await (await applied1d).json();
    assert(appliedOne.config.waveform.samples.length >= 4);
    assert.equal(appliedOne.metadata.pressure_mTorr, 10);
    await page.getByText(/IAEDFの任意波形/).waitFor();
    assert(await page.locator('.waveform-fields input').first().isDisabled());
    report.one_dimensional = { id: source1d.id, samples: appliedOne.metadata.sample_count, distribution: appliedOne.id, passed: true };
    console.log('PASS: 1D CSV simulation, plots and phase-preserving inlet application');

    const hole = structuredClone(appliedOne.config);
    hole.name = '移管検証 · 1D CSV IAEDF → ホール輸送';
    hole.mode = 'uncharged';
    Object.assign(hole.numerics, { backend: 'cpu', run_until: 'time', samples_per_species: 100, batch_size: 100, charging_steps: 1, duration_s: 1e-7, representative_trajectories: 24 });
    const submitted = await api('/jobs', hole);
    await waitJob('/jobs/' + submitted.id);
    const result = await api('/jobs/' + submitted.id + '/results');
    assert.equal(result.metadata.input_files[0].source_job_id, source1d.id);
    assert(result.metadata.input_files[0].velocity_frame === 'sca_surface_local_xyz');
    report.hole = { id: submitted.id, source: source1d.id, passed: true };
    console.log('PASS: generated IAEDF → SCA ion/electron transport');

    await tab.click();
    await workspace.getByLabel('IAEDFモデル', { exact: true }).selectOption('2d');
    await workspace.getByLabel('ケース名', { exact: true }).fill('IAEDF UI検証 · 2D 空間電荷');
    await workspace.getByRole('button', { name: '小規模検証の設定', exact: true }).click();
    await workspace.getByLabel('圧力ケース [mTorr / カンマ区切り]', { exact: true }).fill('10');
    const ring = workspace.locator('.iaedf-settings .panel').filter({ has: page.getByRole('heading', { name: 'リング波形', exact: true }) });
    await ring.getByLabel('リング波形の種類', { exact: true }).selectOption('scaled_wafer');
    await ring.getByLabel('ウェハ倍率', { exact: true }).fill('0.9');
    const submit2d = page.waitForResponse(r => r.url().endsWith('/api/iaedf/jobs') && r.request().method() === 'POST');
    await workspace.getByRole('button', { name: 'IAEDFを計算', exact: true }).click();
    const source2d = await (await submit2d).json();
    const two = await waitJob('/iaedf/jobs/' + source2d.id);
    assert(two.request.config.space_charge.enabled);
    assert(two.plots.phi_sc.length > 0);
    await workspace.getByRole('img', { name: '角度とエネルギーの相関分布', exact: true }).waitFor();
    await workspace.getByLabel('コレクタ x下限', { exact: true }).fill('3.2');
    await workspace.getByLabel('コレクタ x上限', { exact: true }).fill('4.5');
    await workspace.getByLabel('ホール内の方位角', { exact: true }).fill('45');
    await workspace.getByRole('button', { name: 'ホール入口に適用', exact: true }).waitFor({ state: 'visible' });
    await page.waitForFunction(() => !document.querySelector('.iaedf-results button.primary').disabled);
    await page.screenshot({ path: path.join(qa, 'iaedf_2d_desktop.png'), fullPage: true });
    const applied2d = page.waitForResponse(r => r.url().endsWith('/apply') && r.request().method() === 'POST');
    await workspace.getByRole('button', { name: 'ホール入口に適用', exact: true }).click();
    const appliedTwo = await (await applied2d).json();
    assert.equal(appliedTwo.metadata.source_model, '2d');
    assert(Math.abs(appliedTwo.metadata.collector_x_max_m - 0.0045) < 1e-12);
    assert.equal(appliedTwo.metadata.azimuth_deg, 45);
    assert(appliedTwo.metadata.sample_count > 20);
    report.two_dimensional = { id: source2d.id, samples: appliedTwo.metadata.sample_count, distribution: appliedTwo.id, passed: true, validation: two.summary.validation };
    console.log('PASS: 2D space charge, scaled ring waveform, collector and azimuth');

    await tab.click();
    const cancellation = structuredClone(one.request);
    cancellation.name = 'IAEDF UI検証 · 中止';
    cancellation.config.tpmc.n_particles = 200000;
    const toCancel = await api('/iaedf/jobs', cancellation);
    await page.waitForFunction(id => [...document.querySelectorAll('.iaedf-results select option')].some(o => o.value === id), toCancel.id);
    await workspace.getByLabel('計算結果', { exact: true }).selectOption(toCancel.id);
    await workspace.getByRole('button', { name: '計算を中止', exact: true }).click();
    await page.waitForFunction(async id => (await (await fetch('/api/iaedf/jobs/' + id)).json()).status === 'cancelled', toCancel.id);
    report.cancellation = { id: toCancel.id, passed: true };
    await workspace.getByLabel('計算結果', { exact: true }).selectOption(source2d.id);
    await workspace.getByRole('img', { name: '角度とエネルギーの相関分布', exact: true }).waitFor();
    await page.setViewportSize({ width: 390, height: 844 });
    await page.screenshot({ path: path.join(qa, 'iaedf_mobile.png'), fullPage: true });
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), 'Mobile layout overflowed');
    assert.deepEqual(errors, []);
    report.browser_errors = errors;
    fs.writeFileSync(path.join(qa, 'iaedf_ui_check.json'), JSON.stringify(report, null, 2));
    console.log('PASS: cancellation, responsive layout and no browser errors');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
