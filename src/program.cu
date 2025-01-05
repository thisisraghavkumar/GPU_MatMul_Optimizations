#include "kernels/mykernels.cuh"
#include "helpers/myhelpers.h"
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <random>
#include <algorithm>

/*
void naive_mat_mul(float *A, float *B, float *C, int m, int k, int n){
	for(int i=0; i<m; i++){
		for(int j=0; j<n; j++){
			float sum = 0.0f;
			for(int l=0; l<k; l++){
				sum += A[i * k + l] * B[l * n + j];
			}
			C[i * n + j] = sum;
		}
	}
}
*/

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
    float *h_A, *h_B, *h_C, *h_C_cublas;//, *h_C_ref;
    float *d_A, *d_B, *d_C;
    float elapsed_time, cublas_elapsed_time;
    cudaEvent_t beg, end, cublasBeg, cublasEnd;
    void (*invoke_kernel)(float *, float *, float *, int, int, int);


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

    invoke_kernel = invoke_rowmajor_matmul;

    // Startup check
    invoke_kernel(d_A, d_B, d_C, m, k, n);
    cudaMemcpy(h_C, d_C, sF*sizeC, cudaMemcpyDeviceToHost);
    int randomRow = gen() % m;
    int randomCol = gen() % n;
    float tolerance = 1;
    if(fabs(h_C[randomRow * n + randomCol] - h_C_cublas[randomRow * n + randomCol]) > tolerance){
        std::cout << "Error: Cublas and my kernel results do not match at "<<randomRow<<", "<<randomCol << std::endl;
	std::cout <<"Content of h_C = "<<std::setprecision(32)<<h_C[randomRow * n + randomCol]<<std::endl;
	std::cout <<"Content of h_C_cublas = "<<std::setprecision(32)<<h_C_cublas[randomRow * n + randomCol]<<std::endl;

        return 1;
    }
    for(int i=0; i<warmup_runs-1; i++){
        invoke_kernel(d_A, d_B, d_C, m, k, n);
    }
    cudaEventRecord(cublasBeg);
    for(int i=0; i<measurement_runs; i++){
        invoke_cublas_kernel(d_A, d_B, d_C, m, k, n,handle);
	cudaDeviceSynchronize();
    }
    cudaEventRecord(cublasEnd);
    cudaEventSynchronize(cublasBeg);
    cudaEventSynchronize(cublasEnd);
    cudaEventElapsedTime(&cublas_elapsed_time, cublasBeg, cublasEnd);
 
    cudaEventRecord(beg);
    for(int i=0; i<measurement_runs; i++){
        invoke_kernel(d_A, d_B, d_C, m, k, n);
	cudaDeviceSynchronize();
    }
    cudaEventRecord(end);
    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, beg, end);

    std::cout << std::fixed << std::setprecision(5);
    std::cout<<"Number of operations: "<<numoperations<<std::endl;
    std::cout<<"Time taken by cublas kernel: "<<cublas_elapsed_time/measurement_runs<<" ms"<<std::endl;
    std::cout<<"Cublas GFLOPS: "<<(numoperations / ((cublas_elapsed_time / measurement_runs) / 1000)) / 1e9<<std::endl;
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
