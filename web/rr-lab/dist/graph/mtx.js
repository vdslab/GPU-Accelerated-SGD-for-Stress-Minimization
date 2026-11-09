// Matrix Market coordinate parsing. Values are ignored; edges are undirected.
class MatrixReader {
  constructor() {
    this.header = null;
    this.dimensions = null;
    this.count = 0;
    this.size = 0;
  }
  line(line) {
    const s = line.trim();
    if (!s) return;
    if (!this.header) {
      const h = s.split(/\s+/).map((x) => x.toLowerCase());
      if (h.length !== 5 || h[0] !== '%%matrixmarket' || h[1] !== 'matrix' || h[2] !== 'coordinate')
        throw Error('Matrix Market coordinate形式の.mtxファイルを選んでください。');
      if (
        !['pattern', 'real', 'integer', 'complex'].includes(h[3]) ||
        !['general', 'symmetric', 'hermitian', 'skew-symmetric'].includes(h[4])
      )
        throw Error('未対応のMatrix Marketヘッダーです。');
      this.header = h;
      this.fields = { pattern: 2, real: 3, integer: 3, complex: 4 }[h[3]];
      return;
    }
    if (s.startsWith('%')) return;
    const parts = s.split(/\s+/).map(Number);
    if (!this.dimensions) {
      const [n, m, k] = parts;
      if (
        parts.length !== 3 ||
        !parts.every(Number.isSafeInteger) ||
        n < 1 ||
        n !== m ||
        n > 0x7ffffffe ||
        k < 0 ||
        k > 0x7fffffff
      )
        throw Error('正方行列と32bitで表せる頂点数・要素数が必要です。');
      this.dimensions = parts;
      // Sort canonical endpoint pairs in native code; no string-key Set proportional to file size.
      this.keys = new BigUint64Array(k);
      this.words = new Uint32Array(this.keys.buffer);
      this.littleEndian = new Uint8Array(new Uint32Array([1]).buffer)[0] === 1;
      return;
    }
    const row = ++this.count;
    if (row > this.dimensions[2] || parts.length !== this.fields || !parts.every(Number.isFinite))
      throw Error(`データ行 ${row} が不正です。`);
    const u = parts[0] - 1,
      v = parts[1] - 1;
    if (
      !Number.isInteger(u) ||
      !Number.isInteger(v) ||
      u < 0 ||
      v < 0 ||
      u >= this.dimensions[0] ||
      v >= this.dimensions[0]
    )
      throw Error(`データ行 ${row} の頂点番号が範囲外です。`);
    if (u === v) return;
    const at = this.size++ * 2,
      a = Math.min(u, v),
      b = Math.max(u, v);
    this.words[at + (this.littleEndian ? 1 : 0)] = a;
    this.words[at + (this.littleEndian ? 0 : 1)] = b;
  }
  finish() {
    if (!this.dimensions || this.count !== this.dimensions[2])
      throw Error('ヘッダーの非ゼロ要素数とデータ行数が一致しません。');
    const keys = this.keys.subarray(0, this.size);
    keys.sort();
    let unique = 0;
    for (let i = 0; i < keys.length; i++) if (i === 0 || keys[i] !== keys[i - 1]) unique++;
    const edgeData = new Uint32Array(unique * 2);
    let next = 0;
    for (let i = 0; i < keys.length; i++) {
      if (i && keys[i] === keys[i - 1]) continue;
      edgeData[next++] = this.words[i * 2 + (this.littleEndian ? 1 : 0)];
      edgeData[next++] = this.words[i * 2 + (this.littleEndian ? 0 : 1)];
    }
    return { n: this.dimensions[0], edgeData };
  }
}
export function parseMTX(text) {
  const reader = new MatrixReader();
  // Do not split the complete file into an array of line strings.
  let start = 0;
  for (let i = 0; i < text.length; i++)
    if (text.charCodeAt(i) === 10) {
      reader.line(text.slice(start, i));
      start = i + 1;
    }
  reader.line(text.slice(start));
  return reader.finish();
}
/** File/Response streams stay bounded even for files larger than the JS string limit. */
export async function parseMTXStream(stream, totalBytes = 0, report = () => {}) {
  const reader = new MatrixReader(),
    decoder = new TextDecoder();
  let pending = '',
    loaded = 0;
  for await (const chunk of stream) {
    loaded += chunk.byteLength;
    pending += decoder.decode(chunk, { stream: true });
    let start = 0,
      end;
    while ((end = pending.indexOf('\n', start)) !== -1) {
      reader.line(pending.slice(start, end));
      start = end + 1;
    }
    pending = pending.slice(start);
    if (pending.length > 1024 * 1024) throw Error('MTXの1行が長すぎます。');
    report(
      totalBytes ? loaded / totalBytes : 0,
      `MTXを読み込み中（${Math.round(loaded / 2 ** 20)} MiB）`,
    );
  }
  reader.line(pending + decoder.decode());
  report(1, '辺の重複を除去中');
  return reader.finish();
}
