#include "kernels/mykernels.cuh"
#include "helpers/myhelpers.h"
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

/*
* Invocation starts here.
*/
int main(){
    int m = MMM;
    int n = MMN;
    int k = MMK;
    int sizeA = m * k;
    int sizeB = k * n;
    int sizeC = m * n;
    int sF = sizeof(float);
    int warmup_runs = 5;
    int measurement_runs = 50;
    int numoperations = m * n * 2 * k;
    float *h_A, *h_B, *h_C, *h_C_cublas;
    float *d_A, *d_B, *d_C;
    float elapsed_time, cublas_elapsed_time;
    cudaEvent_t beg, end, cublasBeg, cublasEnd;
    void (*invoke_kernel)(float *, float *, float *, int, int, int);


    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(50.0, 25.0);
    
    h_A = new float[sizeA];
    h_B = new float[sizeB];
    h_C = new float[sizeC];
    h_C_cublas = new float[sizeC];
    cudaMalloc(&d_A, sF * sizeA);
    cudaMalloc(&d_B, sF * sizeB);
    cudaMalloc(&d_C, sF * sizeC);

    populate_array(h_A, sizeA, gen, dis);
    populate_array(h_B, sizeB, gen, dis);
    cudaMemcpy(d_A, h_A, sF*sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sF*sizeB, cudaMemcpyHostToDevice);

    cudaEventCreate(&beg);
    cudaEventCreate(&end);
    cudaEventCreate(&cublasBeg);
    cudaEventCreate(&cublasEnd);
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    cudaEventRecord(cublasBeg);
    invoke_cublas_kernel(d_A, d_B, d_C, m, k, n, handle);
    cudaEventRecord(cublasEnd);
    cudaEventSynchronize(cublasBeg);
    cudaEventSynchronize(cublasEnd);
    cudaEventElapsedTime(&cublas_elapsed_time, cublasBeg, cublasEnd);
    cudaMemcpy(h_C_cublas, d_C, sF*sizeC, cudaMemcpyDeviceToHost);

    invoke_kernel = invoke_naive_matmul;

    // Startup check
    invoke_kernel(d_A, d_B, d_C, m, k, n);
    cudaMemcpy(h_C, d_C, sF*sizeC, cudaMemcpyDeviceToHost);
    int randomRow = gen() % m;
    int randomCol = gen() % n;
    if(h_C[randomRow * n + randomCol] != h_C_cublas[randomCol * m + randomRow]){
        std::cout << "Error: Cublas and my kernel results do not match" << std::endl;
        return 1;
    }
    for(int i=0; i<warmup_runs-1; i++){
        invoke_kernel(d_A, d_B, d_C, m, k, n);
    }

    cudaEventRecord(beg);
    for(int i=0; i<measurement_runs; i++){
        invoke_kernel(d_A, d_B, d_C, m, k, n);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, beg, end);

    std::cout << std::fixed << std::setprecision(5);

    std::cout<<"Time taken by cublas kernel: "<<cublas_elapsed_time<<" ms"<<std::endl;
    std::cout<<"Cublas GFLOPS: "<<(numoperations / (cublas_elapsed_time / 1000)) / 1e9<<std::endl;
    std::cout<<"Time taken by my kernel: "<<elapsed_time/measurement_runs<<" ms"<<std::endl;
    std::cout<<"Kernel GFLOPS: "<<(numoperations / ((elapsed_time/measurement_runs) / 1000)) / 1e9<<std::endl;
    std::cout<<"Relative performance: "<<cublas_elapsed_time / (elapsed_time/measurement_runs)<<std::endl;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_cublas;
    cublasDestroy(handle);
}
