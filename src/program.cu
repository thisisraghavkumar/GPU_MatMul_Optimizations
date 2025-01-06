#include "kernels/mykernels.cuh"
#include "helpers/myhelpers.h"
#include <algorithm>
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <random>

/*
* Function to populate an array of floats with random values
*/
void populate_array(float *arr, int size, std::mt19937 &gen, std::uniform_real_distribution<float> &dis){
    for(int i=0; i<size; i++){
        arr[i] = dis(gen);
    }
}

void CudaDeviceInfo() {
    int deviceId;

    cudaGetDevice(&deviceId);

    cudaDeviceProp props{};
    cudaGetDeviceProperties(&props, deviceId);
    std::cout<<"Device ID                         : "<<deviceId<<std::endl;
    std::cout<<"Name                              : "<<props.name<<std::endl;
    std::cout<<"Compute Capability                : "<<props.major<<"."<<props.minor<<std::endl;
    std::cout<<"Memory Bus Width                  : "<<props.memoryBusWidth<<std::endl;
    std::cout<<"Max threads per block             : "<<props.maxThreadsPerBlock<<std::endl;
    std::cout<<"Max threads per multi-processor   : "<<props.maxThreadsPerMultiProcessor<<std::endl;
    std::cout<<"Registers per block               : "<<props.regsPerBlock<<std::endl;
    std::cout<<"Registers per multi-processor     : "<<props.regsPerMultiprocessor<<std::endl;
    std::cout<<"Total Global Memory               : "<<props.totalGlobalMem/1024/1024<<"MB"<<std::endl;
    std::cout<<"Shared Memory per block           : "<<props.sharedMemPerBlock/1024<<"KB"<<std::endl;
    std::cout<<"Shared Memory per multi-processor : "<<props.sharedMemPerMultiprocessor/1024<<"KB"<<std::endl;
    std::cout<<"Total Constant Memory             : "<<props.totalConstMem/1024<<"KB"<<std::endl;
    std::cout<<"Multi-processor count             : "<<props.multiProcessorCount<<std::endl;
    std::cout<<"Warp Size                         : "<<props.warpSize<<std::endl;
    std::cout<<"----------------------------------------------------------------"<<std::endl;
}

/*
* Invocation starts here.
*/
int main(){
    CudaDeviceInfo();
    int m = MMM;
    int n = MMN;
    int k = MMK;
    int sizeA = m * k;
    int sizeB = k * n;
    int sizeC = m * n;
    //int sF = sizeof(float);
    int warmup_runs = 5;
    int measurement_runs = 50;
    long long numoperations = 2LL * m * n * k;
    float *h_A, *h_B, *h_C, *h_C_cublas;//, *h_C_ref;
    float *d_A, *d_B, *d_C;
    float cublas_elapsed_time;
    cudaEvent_t beg, end, cublasBeg, cublasEnd;


    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(50.0, 25.0);
    
    h_A = new float[sizeA]();
    h_B = new float[sizeB]();
    h_C = new float[sizeC]();
    h_C_cublas = new float[sizeC]();
    //h_C_ref = new float[sizeC]();
    cudaMalloc(&d_A, sF * sizeA);
    cudaMalloc(&d_B, sF * sizeB);
    cudaMalloc(&d_C, sF * sizeC);

    //float valuesA[] = {1.0f, 2.0f, 5.0f, 4.0f, 6.0f, 8.0f, 2.0f, 3.0f, 2.0f};
    populate_array(h_A, sizeA, gen, dis);
    //std::copy(valuesA,valuesA+9,h_A);
    //float valuesB[] = {1.0f, 0.0f, 2.0f, 2.0f, 1.0f, 1.0f, 8.0f, 2.0f, 4.0f};
    populate_array(h_B, sizeB, gen, dis);
    //std::copy(valuesB,valuesB+9,h_B);
    cudaMemcpy(d_A, h_A, sF*sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sF*sizeB, cudaMemcpyHostToDevice);
    //naive_mat_mul(h_A, h_B, h_C_ref, m, k, n);
    cudaEventCreate(&beg);
    cudaEventCreate(&end);
    cudaEventCreate(&cublasBeg);
    cudaEventCreate(&cublasEnd);
    

    cublasHandle_t handle;
    cublasCreate(&handle);
    
    invoke_cublas_kernel(d_A, d_B, d_C, m, k, n, handle);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C_cublas, d_C, sF*sizeC, cudaMemcpyDeviceToHost);

    float naive_time = run_kernel("naive", invoke_naive_matmul, d_A, d_B, d_C, m, k, n, h_C, h_C_cublas, gen, warmup_runs, measurement_runs);
    float row_coalesce_time = run_kernel("row_coalesce", invoke_rowmajor_matmul, d_A, d_B, d_C, m, k, n, h_C, h_C_cublas, gen, warmup_runs, measurement_runs);
    float shared_memory_time = run_kernel("shared_memory", invoke_shared_memory_matmul, d_A, d_B, d_C, m, k, n, h_C, h_C_cublas, gen, warmup_runs, measurement_runs);
    float oned_tiled_time = run_kernel("1_d_tiled", invoke_1D_tiled_matmul, d_A, d_B, d_C, m, k, n, h_C, h_C_cublas, gen, warmup_runs, measurement_runs);

    cudaEventRecord(cublasBeg);
    for(int i=0; i<measurement_runs; i++){
        invoke_cublas_kernel(d_A, d_B, d_C, m, k, n,handle);
	cudaDeviceSynchronize();
    }
    cudaEventRecord(cublasEnd);
    cudaEventSynchronize(cublasBeg);
    cudaEventSynchronize(cublasEnd);
    cudaEventElapsedTime(&cublas_elapsed_time, cublasBeg, cublasEnd);

    std::cout << std::fixed << std::setprecision(5);
    std::cout<<"Number of operations: "<<numoperations<<std::endl;
    std::cout<<"Time taken by cublas kernel: "<<cublas_elapsed_time/measurement_runs<<" ms"<<std::endl;
    std::cout<<"Cublas GFLOPS: "<<(numoperations / ((cublas_elapsed_time / measurement_runs) / 1000)) / 1e9<<std::endl;
    std::cout<<"Time taken by naive kernel: "<<naive_time/measurement_runs<<" ms"<<std::endl;
    std::cout<<"Naive Kernel GFLOPS: "<<(numoperations / ((naive_time/measurement_runs) / 1000)) / 1e9<<std::endl;
    std::cout<<"Time taken by row coalesce kernel: "<<row_coalesce_time/measurement_runs<<" ms"<<std::endl;
    std::cout<<"Row coalesce Kernel GFLOPS: "<<(numoperations / ((row_coalesce_time/measurement_runs) / 1000)) / 1e9<<std::endl;
    std::cout<<"Time taken by shared memory kernel: "<<shared_memory_time/measurement_runs<<" ms"<<std::endl;
    std::cout<<"Shared Memory Kernel GFLOPS: "<<(numoperations / ((shared_memory_time/measurement_runs) / 1000)) / 1e9<<std::endl;
    std::cout<<"Time taken by 1-d tiled kernel: "<<oned_tiled_time/measurement_runs<<" ms"<<std::endl;
    std::cout<<"1D Tiled Kernel GFLOPS: "<<(numoperations / ((oned_tiled_time/measurement_runs) / 1000)) / 1e9<<std::endl;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_cublas;
    cublasDestroy(handle);
}
