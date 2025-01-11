#ifndef MY_KERNELS_CUH
#define MY_KERNELS_CUH

#include "../helpers/myhelpers.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <nvtx3/nvToolsExt.h>
#include <random>
#include <iostream>
#include <iomanip>

void invoke_cublas_kernel(float *A, float *B, float *C, int m, int k, int n, cublasHandle_t &handle);
void invoke_naive_matmul(float *A, float *B, float *C, int m, int k, int n);
void invoke_rowmajor_matmul(float *A, float *B, float *C, int m, int k, int n);
void invoke_shared_memory_matmul(float *A, float *B, float *C, int m, int k, int n);
void invoke_oned_tiled_matmul(float *A, float *B, float *C, int m, int k, int n);
void invoke_twod_tiled_matmul(float *A, float *B, float *C, int m, int k, int n);
void invoke_vectorized_matmul(float *A, float *B, float *C, int m, int k, int n);
template <const int BM, const int BN, const int BK, const int TM, const int TN> void invoke_parameterized_vectorized_matmul(float *A, float *B, float *C, int m, int k, int n);

template <typename KernelFunc> float run_kernel(const char *kernel_name, KernelFunc invoke_kernel, float *d_A, float *d_B, float *d_C, int m, int k, int n, float *h_C, float *h_C_ref, std::mt19937 gen, int warmup_runs, int measurement_runs){
    cudaEvent_t beg, end;
    float elapsed_time;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);
    int sizeC = m * n;
    invoke_kernel(d_A, d_B, d_C, m, k, n);
    cudaMemcpy(h_C, d_C, sF*sizeC, cudaMemcpyDeviceToHost);
    for(int i=0; i < 500; i++){
    	int randomRow = gen() % m;
    	int randomCol = gen() % n;
    	float tolerance = 4;
    	if(fabs(h_C[randomRow * n + randomCol] - h_C_ref[randomRow * n + randomCol]) > tolerance){
        	std::cout <<"For kernel "<<kernel_name<<std::endl;
        	std::cout <<"Error: Cublas and my kernel results do not match at "<<randomRow<<", "<<randomCol << std::endl;
        	std::cout <<"Content of h_C = "<<std::setprecision(32)<<h_C[randomRow * n + randomCol]<<std::endl;
        	std::cout <<"Content of h_C_cublas = "<<std::setprecision(32)<<h_C_ref[randomRow * n + randomCol]<<std::endl;
        	return 1;
    	}
    }
    for(int i=0; i<warmup_runs-1; i++){
        invoke_kernel(d_A, d_B, d_C, m, k, n);
    }
    cudaEventRecord(beg);
    nvtxRangePush(kernel_name);
    for(int i=0; i<measurement_runs; i++){
        invoke_kernel(d_A, d_B, d_C, m, k, n);
        cudaDeviceSynchronize();
    }
    nvtxRangePop();
    cudaEventRecord(end);
    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, beg, end);
    return elapsed_time;
}

#endif
