// Level-synchronous sparse frontier BFS. CSR is uploaded once per context.
#include <cuda_runtime.h>
#include <cstdint>
#include <chrono>
#include <new>
#include <utility>
#include <limits>
using Clock = std::chrono::steady_clock;
static double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
struct Bfs {
    uint32_t n = 0;
    uint32_t *rows = nullptr, *cols = nullptr, *distance = nullptr;
    uint32_t *front = nullptr, *next = nullptr, *count = nullptr;
    double transfer = 0.0;
    ~Bfs() {
        cudaFree(rows); cudaFree(cols); cudaFree(distance);
        cudaFree(front); cudaFree(next); cudaFree(count);
    }
};
__global__ void initialize(uint32_t n, uint32_t source, uint32_t *distance, uint32_t *front) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) distance[i] = i == source ? 0 : UINT32_MAX;
    if (i == 0) front[0] = source;
}
__global__ void expand(const uint32_t *rows, const uint32_t *cols,
                       const uint32_t *front, uint32_t size, uint32_t level,
                       uint32_t *distance, uint32_t *next, uint32_t *count) {
    // One warp per frontier vertex; threads only visit its CSR neighbors.
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t item = tid / 32, lane = tid % 32;
    if (item >= size) return;
    const uint32_t u = front[item];
    for (uint64_t e = uint64_t(rows[u]) + lane; e < rows[u + 1]; e += 32) {
        const uint32_t v = cols[e];
        if (atomicCAS(&distance[v], UINT32_MAX, level + 1) == UINT32_MAX) {
            // CAS wins once per vertex, so next has at most n entries.
            const uint32_t slot = atomicAdd(count, 1u);
            next[slot] = v;
        }
    }
}
static cudaError_t copy(Bfs *b, void *dst, const void *src, size_t bytes, cudaMemcpyKind kind) {
    auto start = Clock::now();
    auto err = cudaMemcpy(dst, src, bytes, kind);
    b->transfer += seconds(start);
    return err;
}
extern "C" int pivot_bfs_create(uint32_t n, uint32_t edges, const uint32_t *rows,
                                const uint32_t *cols, Bfs **result, double *transfer) {
    *result = nullptr;
    if (!n || !rows || rows[0] != 0 || rows[n] != edges) return cudaErrorInvalidValue;
    for (uint32_t u = 0; u < n; ++u) if (rows[u] > rows[u+1]) return cudaErrorInvalidValue;
    for (uint32_t e = 0; e < edges; ++e) if (cols[e] >= n) return cudaErrorInvalidValue;
    Bfs *b = new(std::nothrow) Bfs;
    if (!b) return cudaErrorMemoryAllocation;
    b->n = n;
    cudaError_t err = cudaSuccess;
#define TRY_CREATE(expr) do { err = (expr); if (err != cudaSuccess) { delete b; return err; } } while(0)
    TRY_CREATE(cudaMalloc(&b->rows, (size_t(n) + 1) * sizeof(uint32_t)));
    TRY_CREATE(cudaMalloc(&b->cols, size_t(edges ? edges : 1) * sizeof(uint32_t)));
    TRY_CREATE(cudaMalloc(&b->distance, size_t(n) * sizeof(uint32_t)));
    TRY_CREATE(cudaMalloc(&b->front, size_t(n) * sizeof(uint32_t)));
    TRY_CREATE(cudaMalloc(&b->next, size_t(n) * sizeof(uint32_t)));
    TRY_CREATE(cudaMalloc(&b->count, sizeof(uint32_t)));
    TRY_CREATE(copy(b, b->rows, rows, (size_t(n)+1)*sizeof(uint32_t), cudaMemcpyHostToDevice));
    if (edges) TRY_CREATE(copy(b, b->cols, cols, size_t(edges)*sizeof(uint32_t), cudaMemcpyHostToDevice));
    *transfer = b->transfer;
    *result = b;
    return cudaSuccess;
#undef TRY_CREATE
}
extern "C" int pivot_bfs_run(Bfs *b, uint32_t source, uint32_t *output, double *transfer) {
    if (!b || source >= b->n || !output) return cudaErrorInvalidValue;
    b->transfer = 0;
#define TRY_RUN(expr) do { auto err = (expr); if (err != cudaSuccess) return err; } while(0)
    initialize<<<(uint64_t(b->n)+255)/256, 256>>>(b->n, source, b->distance, b->front);
    TRY_RUN(cudaGetLastError());
    TRY_RUN(cudaDeviceSynchronize());
    uint32_t size = 1, level = 0;
    while (size) {
        TRY_RUN(cudaMemset(b->count, 0, sizeof(uint32_t)));
        expand<<<(uint64_t(size)*32+255)/256, 256>>>(b->rows, b->cols, b->front, size,
                                                  level, b->distance, b->next, b->count);
        TRY_RUN(cudaGetLastError());
        // Keep kernel completion in BFS time, outside transfer time.
        TRY_RUN(cudaDeviceSynchronize());
        TRY_RUN(copy(b, &size, b->count, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        if (size > b->n) return cudaErrorInvalidValue;
        std::swap(b->front, b->next);
        ++level;
    }
    TRY_RUN(copy(b, output, b->distance, size_t(b->n)*sizeof(uint32_t), cudaMemcpyDeviceToHost));
    *transfer = b->transfer;
    return cudaSuccess;
#undef TRY_RUN
}
extern "C" void pivot_bfs_destroy(Bfs *b) { delete b; }
extern "C" const char *pivot_bfs_error(int code) { return cudaGetErrorString(static_cast<cudaError_t>(code)); }
