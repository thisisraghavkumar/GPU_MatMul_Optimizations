#include "mykernels.cuh"

/**
 * Output in C is stored in column major format.
 */
void invoke_cublas_kernel(float *A, float *B, float *C, int m, int k, int n, cublasHandle_t &handle){
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, n, m, k, &alpha, B, k, A, n, &beta, C, n);
}
