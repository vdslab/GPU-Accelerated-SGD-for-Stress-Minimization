/** Shared by the method picker and the Full preprocessor: never allocate n² to decide support. */
export function fullMemoryBytes(vertexCount) {
  return vertexCount * vertexCount * Float32Array.BYTES_PER_ELEMENT;
}

export function graphMethodSupport(vertexCount, limits) {
  const limit = limits ? Math.min(limits.maxStorageBufferBindingSize, limits.maxBufferSize) : 0;
  const requiredBytes = fullMemoryBytes(vertexCount);
  return {
    full: { available: limit > 0 && requiredBytes <= limit, requiredBytes, limit },
    sparse: { maxPivots: limit > 0 ? Math.min(1000, Math.floor(limit / (8 * vertexCount))) : 0 },
  };
}

export function memoryLabel(bytes) {
  const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];
  let unit = 0,
    value = bytes;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit++;
  }
  return `${Number(value.toFixed(value < 10 ? 2 : 1))} ${units[unit]}`;
}
