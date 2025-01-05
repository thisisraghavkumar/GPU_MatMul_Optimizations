#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include "mykernels.cuh"
#include "../helpers/myhelpers.h"

/**
 * Returns the time taken by invoking the passed kernel on matrices A and B measurement_iterations times.
 */
float run_kernel(const char* kernel_name, void (*invoke_kernel)(float *, float *, float *, int, int, int), float *d_A, float *d_B, float *d_C, int m, int k, int n, float *h_C, float *h_C_ref, std::mt19937 gen, int warmup_runs, int measurement_runs){
    cudaEvent_t beg, end;
    float elapsed_time;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);
    int sizeC = m * n;
    invoke_kernel(d_A, d_B, d_C, m, k, n);
    cudaMemcpy(h_C, d_C, sF*sizeC, cudaMemcpyDeviceToHost);
    int randomRow = gen() % m;
    int randomCol = gen() % n;
    float tolerance = 1;
    if(fabs(h_C[randomRow * n + randomCol] - h_C_ref[randomRow * n + randomCol]) > tolerance){
        std::cout <<"For kernel "<<kernel_name<<std::endl;
        std::cout <<"Error: Cublas and my kernel results do not match at "<<randomRow<<", "<<randomCol << std::endl;
        std::cout <<"Content of h_C = "<<std::setprecision(32)<<h_C[randomRow * n + randomCol]<<std::endl;
        std::cout <<"Content of h_C_cublas = "<<std::setprecision(32)<<h_C_ref[randomRow * n + randomCol]<<std::endl;
        return 1;
    }
    for(int i=0; i<warmup_runs-1; i++){
        invoke_kernel(d_A, d_B, d_C, m, k, n);
    }
 
    cudaEventRecord(beg);
    for(int i=0; i<measurement_runs; i++){
        invoke_kernel(d_A, d_B, d_C, m, k, n);
	cudaDeviceSynchronize();
    }
    cudaEventRecord(end);
    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, beg, end);

    return elapsed_time;
}