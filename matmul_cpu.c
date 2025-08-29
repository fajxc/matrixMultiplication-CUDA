#include <stdio.h>
#include <stdlib.h>
#include <time.h>


void matmul_cpu(float* A, float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void random_matrix_cpu(float* M, int N) {
    for (int i = 0; i < N * N; i++) {
        M[i] = (float)(rand() % 100) / 10.0f; // values 0–9.9
    }
}

/*int main() {
    int N = 512; // test size
    size_t size = N * N * sizeof(float);

    float* A = (float*)malloc(size);
    float* B = (float*)malloc(size);
    float* C = (float*)malloc(size);

    srand(time(NULL));
    random_matrix(A, N);
    random_matrix(B, N);

    clock_t start = clock();
    matmul_cpu(A, B, C, N);
    clock_t end = clock();

    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;
    printf("CPU time (N=%d): %f seconds\n", N, time_taken);

    free(A); free(B); free(C);
    return 0;
}*/
