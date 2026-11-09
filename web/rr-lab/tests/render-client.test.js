import { test } from 'node:test';
import assert from 'node:assert/strict';
import { RenderClient } from '../dist/ui/render-client.js';
test('a cancelled renderer cannot paint a stale frame or consume the new worker request', async () => {
  const original = globalThis.Worker,
    workers = [];
  globalThis.Worker = class {
    constructor() {
      workers.push(this);
    }
    postMessage() {}
    terminate() {}
  };
  try {
    const client = new RenderClient({ n: 2 }, Uint32Array.from([0, 1]));
    const drawn = [];
    const target = {
      dataset: {},
      setAttribute() {},
      getContext() {
        return {
          drawImage(image) {
            drawn.push(image.id);
          },
        };
      },
    };
    client.draw(target, { resolution: 512, positions: new Float32Array(4) });
    const staleHandler = workers[0].onmessage;
    client.restart();
    client.draw(target, { resolution: 4096, positions: new Float32Array(4) });
    let closed = 0;
    staleHandler({
      data: {
        bitmap: {
          id: 'stale',
          close() {
            closed++;
          },
        },
      },
    });
    assert(client.active);
    assert.deepEqual(drawn, []);
    const idle = client.whenIdle();
    workers[1].onmessage({
      data: {
        bitmap: {
          id: 'current',
          close() {
            closed++;
          },
        },
      },
    });
    await idle;
    assert.deepEqual(drawn, ['current']);
    assert.equal(target.width, 4096);
    assert.equal(closed, 2);
    client.dispose();
  } finally {
    globalThis.Worker = original;
  }
});

test('identical Before/After frames share a GPU image; a different camera needs a new frame', async () => {
  const original = globalThis.Worker,
    workers = [];
  globalThis.Worker = class {
    constructor() {
      this.messages = [];
      workers.push(this);
    }
    postMessage(message) {
      this.messages.push(message);
    }
    terminate() {}
  };
  try {
    const client = new RenderClient({ n: 2 }, new Uint32Array(2));
    const paints = [];
    const canvas = (id) => ({
      id,
      dataset: {},
      setAttribute() {},
      getContext() {
        return {
          drawImage() {
            paints.push(id);
          },
        };
      },
    });
    const before = canvas('before'),
      after = canvas('graph'),
      positions = new Float32Array(4);
    const job = { positions, resolution: 512, view: { zoom: 1 } };
    client.draw(before, job);
    client.draw(after, job);
    workers[0].onmessage({ data: { bitmap: { close() {} } } });
    await client.whenIdle();
    assert.deepEqual(paints, ['before', 'graph']);
    assert.equal(workers[0].messages.filter((message) => message.kind === 'render').length, 1);
    client.draw(before, job);
    client.draw(after, { ...job, view: { zoom: 2 } });
    workers[0].onmessage({ data: { bitmap: { close() {} } } });
    assert(client.active);
    assert.equal(workers[0].messages.filter((message) => message.kind === 'render').length, 3);
    workers[0].onmessage({ data: { bitmap: { close() {} } } });
    await client.whenIdle();
    client.dispose();
  } finally {
    globalThis.Worker = original;
  }
});

test('continuous camera changes show completed frames and coalesce pending work to the latest camera', async () => {
  const original = globalThis.Worker,
    workers = [];
  globalThis.Worker = class {
    constructor() {
      this.messages = [];
      workers.push(this);
    }
    postMessage(message) {
      this.messages.push(message);
    }
    terminate() {}
  };
  try {
    const client = new RenderClient({ n: 2 }, new Uint32Array(2)),
      painted = [];
    const target = {
      id: 'graph',
      dataset: {},
      setAttribute() {},
      getContext() {
        return {
          drawImage(image) {
            painted.push(image.id);
          },
        };
      },
    };
    const positions = new Float32Array(4);
    for (const zoom of [1, 2, 3])
      client.draw(target, { positions, resolution: 512, view: { zoom } });
    workers[0].onmessage({ data: { bitmap: { id: 1, close() {} } } });
    assert.deepEqual(painted, [1]);
    assert.equal(target.dataset.rendering, 'true');
    assert.equal(workers[0].messages.at(-1).view.zoom, 3);
    workers[0].onmessage({ data: { bitmap: { id: 3, close() {} } } });
    await client.whenIdle();
    assert.deepEqual(painted, [1, 3]);
    assert.equal(target.dataset.rendering, 'false');
    client.dispose();
  } finally {
    globalThis.Worker = original;
  }
});
