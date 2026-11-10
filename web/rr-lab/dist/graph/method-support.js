const BANK_BYTES = 128 * 2 ** 20;
const MAX_BANKS = 4;
const FULL_MAX_NODES = 16384;
const ATOMIC_FULL_MAX_NODES = 8192;

/** Shared by the method picker and the Full preprocessor: lossless u16 distance matrix (2 * n² bytes) across up to 4 storage banks (512 MiB). */
export function fullMemoryBytes(vertexCount) {
  return vertexCount * vertexCount * Uint16Array.BYTES_PER_ELEMENT;
}

/** Atomic Full SGD stores n(n - 1)/2 explicit 16-byte EdgeInfo pairs (8 * n * (n - 1) bytes) across up to 4 storage banks (512 MiB). */
export function atomicFullMemoryBytes(vertexCount) {
  return 8 * vertexCount * Math.max(0, vertexCount - 1);
}

export function multiBankCapacity(limits) {
  const limit = limits ? Math.min(limits.maxStorageBufferBindingSize, limits.maxBufferSize) : 0;
  return limit >= BANK_BYTES ? MAX_BANKS * BANK_BYTES : limit;
}

export function graphMethodSupport(vertexCount, limits) {
  const limit = multiBankCapacity(limits);
  const requiredBytes = fullMemoryBytes(vertexCount);
  const atomicRequiredBytes = atomicFullMemoryBytes(vertexCount);
  return {
    full: {
      available: limit > 0 && vertexCount <= FULL_MAX_NODES && requiredBytes <= limit,
      requiredBytes,
      limit,
    },
    atomic_full: {
      available: limit > 0 && vertexCount <= ATOMIC_FULL_MAX_NODES && atomicRequiredBytes <= limit,
      requiredBytes: atomicRequiredBytes,
      limit,
    },
    sparse: { maxPivots: limit > 0 ? Math.min(1000, Math.floor(limit / (8 * vertexCount))) : 0 },
    atomic_sparse: {
      maxPivots: limit > 0 ? Math.min(1000, Math.floor(limit / (20 * vertexCount))) : 0,
    },
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
