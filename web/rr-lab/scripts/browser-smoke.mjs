import { chromium } from '@playwright/test';
import { startStaticServer } from './static-server.mjs';
import { readFile, mkdir } from 'node:fs/promises';
import { resolve } from 'node:path';
import assert from 'node:assert/strict';
import { prepareSparse } from '../dist/preprocess/sparse.js';
import { sparseReference } from '../tests/sparse-reference.js';
import { reference } from '../tests/reference.js';
import { prepareFull } from '../dist/preprocess/full.js';

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
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } }),
  errors = [];
page.on('pageerror', (error) => errors.push(error.message));
try {
  await page.goto(base);
  await page.waitForFunction(() => document.getElementById('status').textContent === '準備完了');
  assert.match(await page.locator('#gpu-state').innerText(), /利用可能/);
  assert.equal(await page.locator('h1').innerText(), 'RR-SGD');
  assert.equal(await page.title(), 'RR-SGD');
  assert.equal(await page.locator('.app-shell > .active-graph').count(), 0);
  assert.equal(await page.locator('.source-panel .active-graph').count(), 1);
  assert.equal(
    await page
      .locator('.info-button')
      .evaluate((button) => getComputedStyle(button).borderTopWidth),
    '0px',
  );
  await page.getByRole('button', { name: 'RR-SGDの概要を読む' }).click();
  assert(await page.getByRole('dialog', { name: 'RR-SGDについて' }).isVisible());
  await page.keyboard.press('Escape');
  assert(!(await page.locator('#about-rrgraph').isVisible()));
  assert(!(await page.locator('body').innerText()).includes('実験'));

  // Changing candidates or picker modes must not change the actual active graph.
  assert.equal(await page.locator('#active-graph-kind').innerText(), 'サンプル');
  assert.equal(await page.locator('#active-graph-name').innerText(), 'USpowerGrid');
  await page.locator('#sample').selectOption('luxembourg');
  assert.equal(await page.locator('#active-graph-name').innerText(), 'USpowerGrid');
  assert.match(await page.locator('#source-hint').innerText(), /切り替わりません/);
  await page.locator('#source-upload').check();
  assert.equal(await page.locator('#active-graph-kind').innerText(), 'サンプル');
  await page
    .locator('#file')
    .setInputFiles(resolve(import.meta.dirname, '../tests/fixtures/cycle257.mtx'));
  await page.waitForFunction(
    () => document.getElementById('active-graph-name').textContent === 'cycle257.mtx',
  );
  assert.equal(await page.locator('#active-graph-kind').innerText(), 'アップロード');
  assert.match(await page.locator('#active-graph-stats').innerText(), /257 頂点/);
  await page.locator('#run').click();
  await page.waitForFunction(() => document.getElementById('status').textContent === '完了');
  assert(await page.locator('#run').isEnabled());
  assert.equal(await page.locator('#run').innerText(), 'もう一度計算');
  const originalStress = await page.locator('#stress').innerText();
  await page.locator('#run').click();
  await page.waitForFunction(() => document.getElementById('status').textContent === '完了');
  assert.equal(await page.locator('#stress').innerText(), originalStress);
  assert(await page.locator('#seed').isEnabled());
  const previousResult = await page.locator('#result-text').inputValue();
  await page.locator('#seed').fill('7');
  await page.locator('#iterations').fill('5');
  assert.equal(await page.locator('#run').innerText(), 'この設定で再計算');
  assert.equal(await page.locator('#result-text').inputValue(), previousResult);
  await page.locator('#run').click();
  await page.waitForFunction(() => document.getElementById('status').textContent === '完了');
  assert.equal(await page.locator('#epoch').innerText(), '5 / 5');
  assert.match(await page.locator('#result-text').inputValue(), /seed: 7/);
  console.log('Repeated calculation and changed parameters passed without resetting.');
  const uploadedResult = await page.locator('#result-text').inputValue();
  assert.match(uploadedResult, /読み込み元: アップロード/);
  await page.locator('#file').setInputFiles({
    name: 'invalid.mtx',
    mimeType: 'text/plain',
    buffer: Buffer.from('not a Matrix Market file'),
  });
  await page.waitForFunction(() => document.getElementById('status').classList.contains('error'));
  assert.equal(await page.locator('#active-graph-name').innerText(), 'cycle257.mtx');
  assert.equal(await page.locator('#result-text').inputValue(), uploadedResult);
  assert.match(await page.locator('#source-hint').innerText(), /そのまま/);
  await page.locator('#source-sample').check();
  assert.equal(await page.locator('#active-graph-kind').innerText(), 'アップロード');
  await page.locator('#sample').selectOption('powergrid');
  await page.locator('#load-sample').click();
  await page.waitForFunction(() => document.getElementById('status').textContent === '準備完了');
  assert.equal(await page.locator('#active-graph-name').innerText(), 'USpowerGrid');
  assert.equal(await page.locator('#active-graph-kind').innerText(), 'サンプル');
  console.log(
    'Active graph identity, uploads, candidate selection, failed-load preservation, and overview passed.',
  );
  await page.locator('#seed').fill('0');
  await page.locator('#iterations').fill('15');
  // Verify the Sparse WGSL numerically against an independent serial implementation.
  const graph = { n: 17, edges: Array.from({ length: 17 }, (_, i) => [i, (i + 1) % 17]) };
  for (const pivots of [1, 5, 17]) {
    const config = { method: 'sparse', seed: 42, pivots, iterations: 3, epsilon: 0.1 };
    const actual = await page.evaluate(
      async ({ graph, config }) => {
        const { createGpuContext } = await import(new URL('sgd/context.js', document.baseURI).href);
        const { prepareSparse } = await import(
          new URL('preprocess/sparse.js', document.baseURI).href
        );
        const { SparseEngine } = await import(
          new URL('sgd/sparse-engine.js', document.baseURI).href
        );
        const { learningRate } = await import(
          new URL('sgd/learning-rate.js', document.baseURI).href
        );
        const context = await createGpuContext(),
          prepared = prepareSparse(graph, config, context.limits),
          engine = await SparseEngine.create(context, prepared, config.seed);
        let positions;
        for (let i = 0; i < config.iterations; i++)
          positions = await engine.epoch(learningRate(prepared, config, i));
        engine.dispose();
        context.device.destroy();
        return Array.from(positions);
      },
      { graph, config },
    );
    const prepared = prepareSparse(graph, config, {
        maxStorageBufferBindingSize: 256 * 2 ** 20,
        maxBufferSize: 256 * 2 ** 20,
      }),
      expected = sparseReference(prepared, config);
    let mse = 0;
    actual.forEach((value, i) => {
      assert(Number.isFinite(value));
      mse += (value - expected[i]) ** 2;
    });
    const rmse = Math.sqrt(mse / actual.length);
    assert(rmse < 0.001, `Sparse CPU/GPU RMSE ${rmse}`);
    console.log(JSON.stringify({ test: 'sparse-cpu-gpu', pivots, rmse }));
  }
  const fullGraph = { n: 257, edges: Array.from({ length: 257 }, (_, i) => [i, (i + 1) % 257]) };
  const fullConfig = { seed: 42, iterations: 3, epsilon: 0.1 };
  const fullActual = await page.evaluate(
    async ({ graph, config }) => {
      const { createGpuContext } = await import(new URL('sgd/context.js', document.baseURI).href);
      const { prepareFull } = await import(new URL('preprocess/full.js', document.baseURI).href);
      const { FullEngine } = await import(new URL('sgd/full-engine.js', document.baseURI).href);
      const { learningRate } = await import(new URL('sgd/learning-rate.js', document.baseURI).href);
      const context = await createGpuContext(),
        prepared = prepareFull(graph, config, context.limits);
      const engine = await FullEngine.create(context, prepared, config.seed);
      let positions;
      for (let i = 0; i < config.iterations; i++)
        positions = await engine.epoch(learningRate(prepared, config, i));
      engine.dispose();
      context.device.destroy();
      return Array.from(positions);
    },
    { graph: fullGraph, config: fullConfig },
  );
  const fullPrepared = prepareFull(fullGraph, fullConfig, {
    maxStorageBufferBindingSize: 256 * 2 ** 20,
    maxBufferSize: 256 * 2 ** 20,
  });
  const fullExpected = reference(
    fullGraph,
    fullPrepared.dist,
    fullPrepared.stats.diameter,
    fullConfig,
  );
  let fullMSE = 0;
  fullActual.forEach((value, i) => {
    fullMSE += (value - fullExpected[i]) ** 2;
  });
  const fullRMSE = Math.sqrt(fullMSE / fullActual.length);
  assert(fullRMSE < 0.001);
  console.log(JSON.stringify({ test: 'full-cpu-gpu', rmse: fullRMSE }));
  // Real sample: Full USpowerGrid, well above the former 4096-node cutoff.
  await page.locator('#iterations').fill('15');
  assert(await page.locator('#before-placeholder').isVisible());
  await page.locator('#run').click();
  await page.waitForFunction(
    () =>
      document.getElementById('before-placeholder').hidden &&
      document.getElementById('before').dataset.rendering !== 'true',
  );
  // First rendering adds the backend caption; let its layout and resize redraw settle.
  await page.evaluate(
    () => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))),
  );
  await page.waitForFunction(() => document.getElementById('before').dataset.rendering !== 'true');
  const before = await page.locator('#before').evaluate((canvas) => canvas.toDataURL());
  await page.waitForFunction(
    () => {
      const status = document.getElementById('status');
      if (status.classList.contains('error')) throw new Error(status.textContent);
      return status.textContent === '完了';
    },
    null,
    {
      timeout: 120000,
    },
  );
  assert.equal(await page.locator('#nodes').innerText(), '4,941');
  const afterBefore = await page.locator('#before').evaluate((canvas) => canvas.toDataURL());
  const dimensions = (url) => {
    const bytes = Buffer.from(url.split(',')[1], 'base64');
    return [bytes.readUInt32BE(16), bytes.readUInt32BE(20)];
  };
  console.log(
    JSON.stringify({
      test: 'before-image-geometry',
      before: dimensions(before),
      after: dimensions(afterBefore),
      same: before === afterBefore,
    }),
  );
  assert(
    before === afterBefore,
    'Before image should retain the same initial coordinates and canvas geometry',
  );
  console.log(
    JSON.stringify({
      test: 'USpowerGrid-full',
      result: await page.locator('#result-text').inputValue(),
    }),
  );
  // Compare the same layout at equal display size, independent of bitmap resolution.
  const ink = async () => {
    await page.waitForFunction(() => document.getElementById('graph').dataset.rendering !== 'true');
    return page.locator('#graph').evaluate((canvas) => {
      const sample = document.createElement('canvas');
      sample.width = sample.height = 256;
      const ctx = sample.getContext('2d');
      ctx.drawImage(canvas, 0, 0, 256, 256);
      const pixels = ctx.getImageData(0, 0, 256, 256).data;
      let total = 0;
      for (let i = 0; i < pixels.length; i += 4)
        total += 765 - pixels[i] - pixels[i + 1] - pixels[i + 2];
      return total;
    });
  };
  await page.locator('#resolution').selectOption('512');
  const lowResolutionInk = await ink();
  await page.locator('#resolution').selectOption('4096');
  const highResolutionInk = await ink();
  console.log('Resolution optical density', lowResolutionInk, highResolutionInk);
  assert(
    highResolutionInk / lowResolutionInk > 0.75 && highResolutionInk / lowResolutionInk < 1.25,
    'Higher resolution must retain the apparent node/edge weight',
  );
  assert(await page.locator('figure:has(#graph) #png').isVisible());
  assert.equal(await page.locator('.measurements #png').count(), 0);
  const visibilityDir = process.env.SCREENSHOT_DIR ?? '/tmp/rr-graph-lab-qa';
  await mkdir(visibilityDir, { recursive: true });
  await page.screenshot({ path: resolve(visibilityDir, 'powergrid-readable.png'), fullPage: true });
  console.log(
    JSON.stringify({ test: 'resolution-visibility', lowResolutionInk, highResolutionInk }),
  );
  // The full Luxembourg graph is loaded; no hidden truncation/downsampling.
  await page.locator('#sample').selectOption('luxembourg');
  await page.locator('#load-sample').click();
  await page.waitForFunction(() => document.getElementById('status').textContent === '準備完了');
  assert.equal(await page.locator('#nodes').innerText(), '114,599');
  assert.equal(await page.locator('#method').inputValue(), 'sparse');
  await page.locator('#iterations').fill('3');
  await page.locator('#run').click();
  await page.waitForFunction(
    () => {
      const status = document.getElementById('status');
      if (status.classList.contains('error')) throw new Error(status.textContent);
      return status.textContent === '完了';
    },
    null,
    {
      timeout: 180000,
    },
  );
  assert.match(await page.locator('#result-text').inputValue(), /pivots: 200/);
  assert.match(await page.locator('#evaluation-kind').innerText(), /sampled/);
  console.log(
    JSON.stringify({
      test: 'Luxembourg-sparse',
      result: await page.locator('#result-text').inputValue(),
    }),
  );
  assert(await page.locator('#iterations').isEnabled());
  assert.equal(await page.locator('#run').innerText(), 'もう一度計算');
  await page.locator('#iterations').fill('100');
  await page.locator('#run').click();
  await page.locator('#pause').click();
  await page.waitForFunction(
    () => document.getElementById('status').textContent.startsWith('一時停止'),
    null,
    { timeout: 120000 },
  );
  const pausedEpoch = Number((await page.locator('#epoch').innerText()).split('/')[0]);
  assert(pausedEpoch > 0 && pausedEpoch < 100);
  assert.equal(await page.locator('#run').innerText(), '続きから再開');
  await page.locator('#seed').fill('1');
  assert.equal(await page.locator('#run').innerText(), 'この設定で再計算');
  await page.locator('#seed').fill('0');
  assert.equal(await page.locator('#run').innerText(), '続きから再開');
  assert.equal(Number((await page.locator('#epoch').innerText()).split('/')[0]), pausedEpoch);

  await page.locator('#run').click();
  await page.waitForFunction(
    () => {
      const status = document.getElementById('status');
      if (status.classList.contains('error')) throw new Error(status.textContent);
      return status.textContent === '完了';
    },
    null,
    {
      timeout: 180000,
    },
  );
  assert.equal(await page.locator('#epoch').innerText(), '100 / 100');
  console.log(JSON.stringify({ test: 'pause-resume', pausedEpoch }));
  // Change the draft and recalculate 15 epochs directly, without clearing the result first.
  await page.locator('#iterations').fill('15');
  await page.locator('#run').click();
  await page.waitForFunction(
    () => {
      const status = document.getElementById('status');
      if (status.classList.contains('error')) throw new Error(status.textContent);
      return status.textContent === '完了';
    },
    null,
    {
      timeout: 180000,
    },
  );
  console.log(
    JSON.stringify({
      test: 'Luxembourg-sparse-15',
      result: await page.locator('#result-text').inputValue(),
    }),
  );
  const dir = process.env.SCREENSHOT_DIR ?? '/tmp/rr-graph-lab-qa';
  await mkdir(dir, { recursive: true });
  await page.screenshot({ path: resolve(dir, 'luxembourg-sparse.png'), fullPage: true });
  // These are real bitmap dimensions, including downloads and the enlarged image.
  const downloadPromise = page.waitForEvent('download');
  await page.locator('#png').click();
  const png = await downloadPromise;
  const pngBytes = await readFile(await png.path());
  assert.equal(pngBytes.readUInt32BE(16), 4096);
  assert.equal(pngBytes.readUInt32BE(20), 4096);
  const layoutResult = await page.locator('#result-text').inputValue();
  const layoutEpoch = await page.locator('#epoch').innerText();
  await page.locator('#resolution').selectOption('2048');
  await page.waitForFunction(() =>
    ['before', 'graph'].every((id) => {
      const canvas = document.getElementById(id);
      return canvas.width === 2048 && canvas.dataset.rendering !== 'true';
    }),
  );
  assert.equal(await page.locator('#before').evaluate((canvas) => canvas.width), 2048);
  assert.equal(await page.locator('#graph').evaluate((canvas) => canvas.height), 2048);
  assert.equal(await page.locator('#epoch').innerText(), layoutEpoch);
  assert.equal(await page.locator('#result-text').inputValue(), layoutResult);
  const mainImage = await page.locator('#graph').evaluate((canvas) => canvas.toDataURL());
  for (const target of ['after', 'before']) {
    await page.locator(`#expand-${target}`).click();
    const expanded = page.locator('#expanded-image');
    await page.waitForFunction(
      () => document.getElementById('expanded-image').dataset.rendering !== 'true',
    );
    assert.equal(await expanded.evaluate((canvas) => canvas.width), 2048);
    const snapshot = async () => {
      await page.waitForFunction(
        () => document.getElementById('expanded-image').dataset.rendering !== 'true',
      );
      return expanded.evaluate((canvas) => canvas.toDataURL());
    };
    const fitted = await snapshot();
    await page.locator('#image-zoom-in').click();
    assert.notEqual(await snapshot(), fitted);
    await page.locator('#image-zoom-out').click();
    assert.equal(await snapshot(), fitted);
    await expanded.hover();
    await page.mouse.wheel(0, -200);
    await page.waitForFunction(
      (baseline) => document.getElementById('expanded-image').toDataURL() !== baseline,
      fitted,
    );
    await page.locator('#image-fit').click();
    assert.equal(await snapshot(), fitted);
    const box = await expanded.boundingBox();
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await page.mouse.down();
    await page.mouse.move(box.x + box.width / 2 + 40, box.y + box.height / 2 + 20);
    await page.mouse.up();
    assert.notEqual(await snapshot(), fitted);
    await page.locator('#image-fit').click();
    assert.equal(await snapshot(), fitted);
    const sizes = await page
      .locator('#image-zoom-in, #image-zoom-out, #zoom-in, #zoom-out')
      .evaluateAll((buttons) =>
        buttons.map((button) => {
          const rect = button.getBoundingClientRect();
          return [rect.width, rect.height];
        }),
      );
    assert(
      sizes.every((size) => size[0] === sizes[0][0] && size[1] === sizes[0][1]),
      'All zoom buttons must have identical dimensions',
    );
    if (target === 'after')
      await page.screenshot({ path: resolve(dir, 'expanded-interactive.png'), fullPage: true });
    await page.locator('#close-image').click();
  }
  assert.equal(await page.locator('#graph').evaluate((canvas) => canvas.toDataURL()), mainImage);
  assert.equal(await page.locator('#result-text').inputValue(), layoutResult);
  console.log(
    'Verified 4096×4096 PNG, independent Before/After enlarged zoom/pan/fit, equal zoom buttons, and unchanged results.',
  );
  // A separate mobile page avoids full-page capture artifacts during viewport changes.
  const mobile = await browser.newPage({ viewport: { width: 390, height: 844 } });
  await mobile.goto(base);
  await mobile.waitForFunction(() => document.getElementById('status').textContent === '準備完了');
  await mobile.screenshot({ path: resolve(dir, 'mobile.png'), fullPage: true });
  assert(await mobile.locator('#run').isEnabled());
  await mobile.close();
  await page.locator('#run').click();
  await page.locator('#cancel').click();
  await page.waitForFunction(() => !document.getElementById('load-sample').disabled);
  assert(await page.locator('#run').isDisabled());
  await page.locator('#sample').selectOption('community');
  await page.locator('#load-sample').click();
  await page.waitForFunction(() => document.getElementById('status').textContent === '準備完了');
  assert(await page.locator('#run').isEnabled());
  console.log('Preprocessing cancellation and recovery passed.');
  assert.deepEqual(errors, []);
  console.log('Browser smoke checks passed. Screenshots: ' + dir);
} finally {
  await browser.close();
  await new Promise((resolve) => server.close(resolve));
}
