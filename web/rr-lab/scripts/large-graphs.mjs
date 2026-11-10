import { chromium } from '@playwright/test';
import { startStaticServer } from './static-server.mjs';
import { readFile, mkdir } from 'node:fs/promises';
import { resolve, basename } from 'node:path';
import assert from 'node:assert/strict';
const root = process.env.RR_DIST_DIR ?? resolve(import.meta.dirname, '../dist');
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
if (process.env.GPU_BUFFER_MB)
  await page.addInitScript((mb) => {
    const requestDevice = GPUAdapter.prototype.requestDevice;
    GPUAdapter.prototype.requestDevice = function (descriptor = {}) {
      return requestDevice.call(this, {
        ...descriptor,
        requiredLimits: {
          ...descriptor.requiredLimits,
          maxStorageBufferBindingSize: mb * 2 ** 20,
          maxBufferSize: mb * 2 ** 20,
        },
      });
    };
  }, Number(process.env.GPU_BUFFER_MB));
page.setDefaultTimeout(120000);
try {
  await page.goto(base);
  await page.waitForFunction(() => document.getElementById('status').textContent === 'Ready');
  if (process.env.EDGE_LIMIT)
    await page.locator('#edge-detail').selectOption(process.env.EDGE_LIMIT);
  await page.locator('#source-upload').check();
  const path =
    process.env.GRAPH_FILE ?? resolve(import.meta.dirname, '../../../data/web-Stanford.mtx');
  const start = Date.now();
  await page.locator('#file').setInputFiles(path);
  await page.waitForFunction(() => !document.getElementById('method').disabled, null, {
    timeout: Number(process.env.LOAD_TIMEOUT_MS ?? 600000),
  });
  assert.equal(await page.locator('#active-graph-name').innerText(), basename(path));
  console.log(
    'LOAD',
    Date.now() - start,
    await page.locator('#active-graph-stats').innerText(),
    await page.locator('#status').innerText(),
  );
  if (process.env.PIVOTS) await page.locator('#pivots').fill(process.env.PIVOTS);
  await page.locator('#iterations').fill(process.env.ITERATIONS ?? '3');
  console.log(
    'BEFORE RUN',
    await page.locator('#gpu-state').innerText(),
    await page.locator('#run-hint').innerText(),
  );
  await page.locator('#run').click();
  await page.waitForFunction(
    () =>
      document.getElementById('status').textContent === 'Completed' ||
      document.getElementById('status').classList.contains('error'),
    null,
    { timeout: 1200000 },
  );
  console.log(
    'RENDER STATS',
    await page
      .locator('#before, #graph')
      .evaluateAll((canvases) => canvases.map((canvas) => ({ id: canvas.id, ...canvas.dataset }))),
  );
  if (!process.env.RR_DIST_DIR) {
    assert.equal(await page.locator('#graph').getAttribute('data-renderer'), 'webgpu');
    assert(Number(await page.locator('#graph').getAttribute('data-draw-calls')) > 0);
    assert.equal(
      await page.locator('#graph').getAttribute('data-gpu-fallback'),
      'false',
      'Validation must use a hardware GPU',
    );
    if (process.env.EDGE_LIMIT === '0')
      assert.equal(
        Number(await page.locator('#graph').getAttribute('data-drawn-edges')),
        Number((await page.locator('#edges').innerText()).replaceAll(',', '')),
      );
  }
  console.log('RUN', Date.now() - start, await page.locator('#status').innerText());
  console.log(await page.locator('#result-text').inputValue());
  const output = process.env.SCREENSHOT_DIR ?? '/tmp/rr-large';
  await mkdir(output, { recursive: true });
  await page.screenshot({ path: resolve(output, 'large-graph.png'), fullPage: true });
  assert.equal(await page.locator('#status').innerText(), 'Completed');
  for (const id of ['before', 'graph']) {
    await page.waitForFunction(
      (id) => document.getElementById(id).dataset.rendering !== 'true',
      id,
    );
    const colored = await page.locator('#' + id).evaluate((canvas) => {
      const preview = document.createElement('canvas');
      preview.width = preview.height = 256;
      const ctx = preview.getContext('2d');
      ctx.drawImage(canvas, 0, 0, 256, 256);
      const rgba = ctx.getImageData(0, 0, 256, 256).data;
      let pixels = 0;
      for (let i = 0; i < rgba.length; i += 4)
        if (rgba[i] < 245 || rgba[i + 1] < 245 || rgba[i + 2] < 245) pixels++;
      return pixels;
    });
    console.log('VISIBLE', id, colored);
    assert(colored > 100, 'Graph must not be blank');
  }
  // Enlarged views also use the worker and must finish rendering after interaction.
  await page.locator('#expand-after').click();
  await page.waitForFunction(
    () => document.getElementById('expanded-image').dataset.rendering !== 'true',
  );
  const image = () => page.locator('#expanded-image').evaluate((canvas) => canvas.toDataURL());
  const fitted = await image();
  await page.locator('#image-zoom-in').click();
  await page.waitForFunction(
    () => document.getElementById('expanded-image').dataset.rendering !== 'true',
  );
  assert.notEqual(await image(), fitted);
  await page.locator('#image-fit').click();
  await page.waitForFunction(
    () => document.getElementById('expanded-image').dataset.rendering !== 'true',
  );
  assert.equal(await image(), fitted);
  await page.locator('#close-image').click();
  const downloaded = page.waitForEvent('download');
  await page.locator('#png').click();
  const png = await downloaded;
  const bytes = await readFile(await png.path());
  assert.equal(bytes.readUInt32BE(16), 4096);
  assert.equal(bytes.readUInt32BE(20), 4096);
  console.log('Large-graph enlarged zoom/fit and real 4096×4096 PNG passed.');
} catch (error) {
  console.error(
    'FAIL STATE',
    await page
      .locator('#status')
      .textContent()
      .catch(() => ''),
    await page
      .locator('#gpu-state')
      .textContent()
      .catch(() => ''),
    await page
      .locator('#run-hint')
      .textContent()
      .catch(() => ''),
  );
  throw error;
} finally {
  await browser.close();
  await new Promise((r) => server.close(r));
}
