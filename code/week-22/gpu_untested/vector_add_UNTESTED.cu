// UNTESTED: no GPU and no nvcc on this machine. Shown for reading only (Day 154).
// Build on a CUDA machine with:  nvcc -O3 vector_add_UNTESTED.cu -o vadd && ./vadd
#include <cstdio>
#include <cuda_runtime.h>

__global__ void add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // one thread per element; neighbouring threads touch neighbouring
    if (i < n) c[i] = a[i] + b[i];                    // addresses, so loads are coalesced. Memory-bound: 12 bytes per FLOP.
}

int main() {
    const int n = 1 << 24; size_t bytes = n * sizeof(float);
    float *a, *b, *c;
    cudaMallocManaged(&a, bytes); cudaMallocManaged(&b, bytes); cudaMallocManaged(&c, bytes);
    for (int i = 0; i < n; ++i) { a[i] = 1.0f; b[i] = 2.0f; }
    int threads = 256, blocks = (n + threads - 1) / threads;
    add<<<blocks, threads>>>(a, b, c, n);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(err)); return 1; }
    int bad = 0; for (int i = 0; i < n; ++i) bad += (c[i] != 3.0f);
    printf("mismatches: %d\n", bad);
    cudaFree(a); cudaFree(b); cudaFree(c);
}
