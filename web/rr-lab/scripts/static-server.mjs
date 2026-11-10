import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, resolve } from 'node:path';

/** Serve QA at a project-site prefix as well as the origin root. */
export async function startStaticServer(root, prefix = process.env.RR_TEST_BASE_PATH ?? '/') {
  root = resolve(root);
  prefix = '/' + prefix.split('/').filter(Boolean).join('/') + '/';
  if (prefix === '//') prefix = '/';
  const types = {
    '.js': 'application/javascript',
    '.html': 'text/html',
    '.css': 'text/css',
    '.json': 'application/json',
    '.wasm': 'application/wasm',
  };
  const server = createServer(async (request, response) => {
    try {
      const pathname = decodeURIComponent(new URL(request.url, 'http://localhost').pathname);
      if (!pathname.startsWith(prefix)) throw Error('Outside the site prefix');
      const relative = pathname.slice(prefix.length) || 'index.html';
      const file = resolve(root, relative);
      if (!file.startsWith(root + '/')) throw Error('Invalid path');
      response.setHeader('Content-Type', types[extname(file)] ?? 'text/plain');
      response.end(await readFile(file));
    } catch {
      response.statusCode = 404;
      response.end();
    }
  });
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  return { server, base: `http://127.0.0.1:${server.address().port}${prefix}` };
}
