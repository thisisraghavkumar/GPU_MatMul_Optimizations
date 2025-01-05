#ifndef MY_KERNELS_CUH
#define MY_KERNELS_CUH

#include <cuda_runtime.h>
#include <cublas_v2.h>

void invoke_cublas_kernel(float *A, float *B, float *C, int m, int k, int n, cublasHandle_t &handle);
void invoke_naive_matmul(float *A, float *B, float *C, int m, int k, int n);
void invoke_rowmajor_matmul(float *A, float *B, float *C, int m, int k, int n);

float run_kernel(const char *kernel_name, void (*invoke_kernel)(float *, float *, float *, int, int, int), float *d_A, float *d_B, float *d_C, int m, int k, int n, float *h_C, float *h_C_ref, std::mt19937 gen, int warmup_runs, int measurement_runs);

#endif