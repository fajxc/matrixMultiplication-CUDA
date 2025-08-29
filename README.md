**GPU-Accelerated Matrix Multiplication**

This project implements matrix multiplication on both CPU (C) and GPU (CUDA) and compares performance across different matrix sizes. The goal was to see how GPU parallelism impacts runtime compared to a straightforward CPU implementation.




**What I Did:**

Wrote a simple CPU baseline using triple nested loops.

Implemented a CUDA kernel where each thread computes one output element.

Benchmarked CPU vs GPU for matrix sizes up to 1024×1024.

Logged results into CSV and plotted runtimes and speedup.


**What I Learned**

CPU is fine (or sometimes better) for small problems, but runtime grows quickly with matrix size.

GPUs crush large matrix multiplies because thousands of threads work in parallel.

Memory transfers between CPU and GPU matter, but once the problem is large enough the GPU dominates.

