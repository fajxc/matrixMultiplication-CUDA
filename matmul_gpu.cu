#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "matmul_cpu.c"

__global__ void matmul_kernel(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void random_matrix(float* M, int N) {
    for (int i = 0; i < N * N; i++) {
        M[i] = (float)(rand() % 100) / 10.0f;
    }
}

int main() {
    int N = 512; // test size
    size_t size = N * N * sizeof(float);

    // allocate host memory (cpu)
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    random_matrix(h_A, N);
    random_matrix(h_B, N);

    // allocate device memory (gpu)
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // copy to device (gpu)
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // kernel launch
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N+15)/16, (N+15)/16);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    matmul_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    // back to cpu
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    printf("GPU time (N=%d): %f ms\n", N, ms);

    // cpu check
    float* h_C_cpu = (float*)malloc(size);

    clock_t start_cpu = clock();
    matmul_cpu(h_A, h_B, h_C_cpu, N);
    clock_t end_cpu = clock();
    double cpu_time = 1000.0 * (end_cpu - start_cpu) / CLOCKS_PER_SEC;
    printf("CPU time (N=%d): %f ms\n", N, cpu_time);
    
    // memory deallocation
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);





    return 0;
}
