import { chromium } from '@playwright/test';
import { startStaticServer } from './static-server.mjs';
import { readFile, mkdir } from 'node:fs/promises';
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
await page.addInitScript(() => {
  const NativeWorker = Worker;
  window.requests = [];
  window.Worker = class extends NativeWorker {
    constructor(url, options) {
      super(url, options);
      this.source = String(url);
    }
    postMessage(data, ...rest) {
      window.requests.push({ source: this.source, kind: data.kind });
      return super.postMessage(data, ...rest);
    }
  };
});
async function assertLoadOnly(name) {
  await page.waitForFunction(
    (name) =>
      document.getElementById('active-graph-name').textContent === name &&
      !document.getElementById('method').disabled,
    name,
  );
  await page.waitForTimeout(300);
  const requests = await page.evaluate(() => window.requests);
  assert.equal(await page.locator('#epoch').innerText(), '0 / 15');
  assert.equal(requests.filter((r) => r.kind === 'prepare' || r.kind === 'evaluate').length, 0);
  assert.equal(
    requests.filter((r) => r.source.includes('render-worker')).length,
    0,
    'Load must not create or submit a rendering worker',
  );
  assert(await page.locator('#before-placeholder').isVisible());
  assert(await page.locator('#after-placeholder').isVisible());
  assert(await page.locator('#png').isDisabled());
  console.log('Read only, no SGD/render:', name, requests);
}
try {
  await page.goto(base);
  await page.waitForFunction(() => document.getElementById('status').textContent === '準備完了');
  await assertLoadOnly('USpowerGrid');
  await page.locator('#source-upload').check();
  await page.evaluate(() => (window.requests = []));
  await page
    .locator('#file')
    .setInputFiles(resolve(import.meta.dirname, '../tests/fixtures/cycle257.mtx'));
  await assertLoadOnly('cycle257.mtx');
  await page.locator('#seed').fill('17');
  await page.locator('#resolution').selectOption('2048');
  await assertLoadOnly('cycle257.mtx');
  // A click not preceded by a fresh press must not start a solver.
  await page.locator('#run').dispatchEvent('click');
  await assertLoadOnly('cycle257.mtx');
  const text = await readFile(
    resolve(import.meta.dirname, '../tests/fixtures/cycle257.mtx'),
    'utf8',
  );
  const dropped = await page.evaluateHandle((text) => {
    const d = new DataTransfer();
    d.items.add(new File([text], 'dropped.mtx', { type: 'text/plain' }));
    return d;
  }, text);
  await page.evaluate(() => (window.requests = []));
  await page.locator('#drop').dispatchEvent('drop', { dataTransfer: dropped });
  await assertLoadOnly('dropped.mtx');
  // Explicit keyboard activation starts the initial frame and the solver.
  await page.locator('#run').focus();
  await page.keyboard.press('Enter');
  await page.waitForFunction(() => document.getElementById('status').textContent === '完了');
  assert.equal(await page.locator('#epoch').innerText(), '15 / 15');
  assert(!(await page.locator('#before-placeholder').isVisible()));
  assert(!(await page.locator('#after-placeholder').isVisible()));
  assert(await page.locator('#png').isEnabled());
  assert(
    (await page.evaluate(() => window.requests.filter((r) => r.kind === 'prepare').length)) === 1,
  );
  // A second file selection must stay idle even after a previous completed run.
  await page.evaluate(() => (window.requests = []));
  await page
    .locator('#file')
    .setInputFiles(resolve(import.meta.dirname, '../tests/fixtures/cycle257.mtx'));
  await assertLoadOnly('cycle257.mtx');
  await page.locator('#run').click();
  await page.waitForFunction(() => document.getElementById('status').textContent === '完了');
  assert.equal(await page.locator('#epoch').innerText(), '15 / 15');
  console.log('Explicit keyboard and mouse starts passed.');
} finally {
  await browser.close();
  await new Promise((r) => server.close(r));
}
