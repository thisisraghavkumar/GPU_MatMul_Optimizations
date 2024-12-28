#ifndef MY_KERNELS_CUH
#define MY_KERNELS_CUH

#include <cuda_runtime.h>
#include <cublas_v2.h>

void invoke_cublas_kernel(float *A, float *B, float *C, int m, int k, int n, cublasHandle_t &handle);
void invoke_naive_matmul(float *A, float *B, float *C, int m, int k, int n);
void invoke_rowmajor_matmul(float *A, float *B, float *C, int m, int k, int n);

#endif