const VIEW_SIZE = 1024;

/** Sizes are logical coordinates with a minimum on-screen width, never bitmap pixels. */
export function graphStyle(vertexCount, displayedSide) {
  const logicalPerPixel = VIEW_SIZE / Math.max(1, Math.min(VIEW_SIZE, displayedSide));
  const large = vertexCount > 20000;
  return {
    radius: Math.max(large ? 1.1 : vertexCount > 2000 ? 2 : 3, (large ? 0.5 : 1) * logicalPerPixel),
    edgeWidth: Math.max(large ? 0.9 : 1.2, (large ? 0.45 : 0.7) * logicalPerPixel),
    edgeOpacity: large ? 0.24 : 0.5,
    edgeColor: large ? 'rgba(100, 116, 139, 0.24)' : 'rgba(100, 116, 139, 0.5)',
  };
}
