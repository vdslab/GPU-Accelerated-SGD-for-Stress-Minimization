import { chromium } from '@playwright/test';
import { startStaticServer } from './static-server.mjs';
import { mkdir } from 'node:fs/promises';
import { resolve } from 'node:path';
import assert from 'node:assert/strict';
const root = resolve(import.meta.dirname, '../dist');
const { server, base: localBase } = await startStaticServer(root);
const base = process.env.RR_TEST_URL ?? localBase;
const browser = await chromium.launch({
  executablePath:
    process.env.CHROME_PATH ??
    (process.platform === 'linux'
      ? '/usr/bin/google-chrome'
      : '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'),
  headless: true,
  args: [
    '--enable-unsafe-webgpu',
    ...(process.platform === 'linux'
      ? ['--use-angle=vulkan', '--enable-features=Vulkan,DefaultANGLEVulkan,VulkanFromANGLE']
      : []),
  ],
});
const page = await browser.newPage({ viewport: { width: 1366, height: 768 } });
page.on('pageerror', (error) => console.error('Browser error:', error.message));
page.on('requestfailed', (request) =>
  console.error('Request failed:', request.url(), request.failure()?.errorText),
);
async function assertFits(state) {
  for (const [width, height] of [
    [1440, 900],
    [1366, 768],
    [1280, 720],
    [1280, 600],
  ]) {
    await page.setViewportSize({ width, height });
    const geometry = await page.evaluate(() => {
      const sidebar = document.querySelector('.sidebar'),
        settings = document.querySelector('.settings-panel');
      return {
        sidebarScroll: sidebar.scrollHeight - sidebar.clientHeight,
        sidebarBottom: sidebar.getBoundingClientRect().bottom,
        settingsBottom: settings.getBoundingClientRect().bottom,
        settingsScroll: settings.scrollHeight - settings.clientHeight,
        parameterRows: [...document.querySelectorAll('.fields > label')].map((label) => ({
          top: label.getBoundingClientRect().top,
          bottom: label.getBoundingClientRect().bottom,
        })),
        runBottom: document.getElementById('run').getBoundingClientRect().bottom,
        actionsBottom: document.querySelector('.run-actions').getBoundingClientRect().bottom,
        height: innerHeight,
      };
    });
    const { parameterRows, ...dimensions } = geometry;
    console.log(JSON.stringify({ state, width, height, ...dimensions }));
    for (let i = 1; i < parameterRows.length; i++) {
      assert(
        parameterRows[i].top >= parameterRows[i - 1].bottom,
        'Parameters must appear in one vertical column',
      );
    }
    assert(
      Math.abs(geometry.sidebarBottom - geometry.settingsBottom) <= 1,
      `${state}: settings should fill the sidebar height`,
    );
    assert(geometry.sidebarScroll <= 1, `${state}: sidebar needs scrolling at ${height}px`);
    assert(geometry.settingsScroll <= 1, `${state}: settings need scrolling at ${height}px`);
    assert(
      geometry.actionsBottom <= height - 6,
      `${state}: actions are below the viewport at ${height}px`,
    );
  }
}
try {
  await page.goto(base);
  await page.waitForFunction(() => document.getElementById('status').textContent === 'Ready');
  assert.equal(
    await page.locator('#method option[value=full]').evaluate((el) => el.disabled),
    false,
  );
  await assertFits('sample');
  await page.locator('#source-upload').check();
  const n = 180000;
  const lines = ['%%MatrixMarket matrix coordinate pattern symmetric', `${n} ${n} ${n - 1}`];
  for (let i = 1; i < n; i++) lines.push(`${i} ${i + 1}`);
  await page.locator('#file').setInputFiles({
    name: 'large.mtx',
    mimeType: 'text/plain',
    buffer: Buffer.from(lines.join('\n')),
  });
  await page.waitForFunction(
    () => document.getElementById('active-graph-name').textContent === 'large.mtx',
  );
  await page.waitForFunction(
    () =>
      document.getElementById('status').textContent === 'Ready' &&
      !document.getElementById('method').disabled,
  );
  assert.equal(await page.locator('#method').inputValue(), 'sparse');
  assert(await page.locator('#method option[value=full]').evaluate((el) => el.disabled));
  assert(!(await page.locator('#method option[value=sparse]').evaluate((el) => el.disabled)));
  assert.match(
    await page.locator('#method-support').innerText(),
    /Full unavailable.*distance matrix/,
  );
  const maxPivots = Number(await page.locator('#pivots').getAttribute('max'));
  assert(maxPivots > 0 && maxPivots <= 1000);
  await page.locator('#pivots').fill(String(maxPivots + 1));
  assert(await page.locator('#run').evaluate((el) => el.disabled));
  await assertFits('large-upload-pivot-warning');
  await page.locator('#pivots').fill('200');
  assert(await page.locator('#run').isEnabled());
  await page.locator('#method').selectOption('atomic_sparse');
  assert.equal(await page.locator('#pivots').inputValue(), '149');
  assert(await page.locator('#run').isEnabled());
  await page.locator('#method').selectOption('sparse');
  await page.locator('#pivots').fill('100');
  assert(await page.locator('#run').isEnabled());
  await assertFits('large-upload-ready');

  const nMid = 9000;
  const midLines = [
    '%%MatrixMarket matrix coordinate pattern symmetric',
    `${nMid} ${nMid} ${nMid - 1}`,
  ];
  for (let i = 1; i < nMid; i++) midLines.push(`${i} ${i + 1}`);
  await page
    .locator('#file')
    .setInputFiles(resolve(import.meta.dirname, '../tests/fixtures/cycle257.mtx'));
  await page.waitForFunction(
    () =>
      document.getElementById('active-graph-name').textContent === 'cycle257.mtx' &&
      document.getElementById('status').textContent === 'Ready',
  );
  await page.locator('#method').selectOption('atomic_full');
  await page.locator('#file').setInputFiles({
    name: 'mid9000.mtx',
    mimeType: 'text/plain',
    buffer: Buffer.from(midLines.join('\n')),
  });
  await page.waitForFunction(
    () =>
      document.getElementById('active-graph-name').textContent === 'mid9000.mtx' &&
      document.getElementById('status').textContent === 'Ready',
  );
  assert.equal(await page.locator('#method').inputValue(), 'atomic_sparse');
  assert(!(await page.locator('#method option[value=full]').evaluate((el) => el.disabled)));
  assert(await page.locator('#method option[value=atomic_full]').evaluate((el) => el.disabled));
  assert.match(await page.locator('#method-support').innerText(), /Atomic Full unavailable/);
  assert(await page.locator('#run').isEnabled());
  await assertFits('mid-upload-atomic-full-unavailable');
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.waitForTimeout(350);
  const dir = process.env.SCREENSHOT_DIR ?? '/tmp/rrgraph-layout';
  await mkdir(dir, { recursive: true });
  await page.screenshot({ path: resolve(dir, 'compact-sidebar.png'), fullPage: true });
  await page
    .locator('#file')
    .setInputFiles(resolve(import.meta.dirname, '../tests/fixtures/cycle257.mtx'));
  await page.waitForFunction(
    () => document.getElementById('active-graph-name').textContent === 'cycle257.mtx',
  );
  await page.waitForFunction(
    () =>
      document.getElementById('status').textContent === 'Ready' &&
      !document.getElementById('method').disabled,
  );
  assert(!(await page.locator('#method option[value=full]').evaluate((el) => el.disabled)));
  await page.locator('#method').selectOption('full');
  assert(await page.locator('#method-support').isHidden());
  console.log(
    'Full disables for large uploads, re-enables for small uploads; all desktop sidebar sizes fit.',
  );
} catch (error) {
  console.error(
    'Page state:',
    await page.locator('#status').textContent(),
    await page.locator('#gpu-state').textContent(),
  );
  throw error;
} finally {
  await browser.close();
  await new Promise((r) => server.close(r));
}
