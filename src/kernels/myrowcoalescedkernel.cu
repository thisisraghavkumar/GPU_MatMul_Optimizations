#include "mykernels.cuh"
#include "../helpers/myhelpers.h"
#include <cuda_runtime.h>

__global__ void myRowCoalesceKernel(float *A, float *B, float *C, int m, int k, int n){
    int firstRow = blockIdx.x * BLOCK_SIZE + threadIdx.x/blockDim.x;
    int secondCol = blockIdx.y * BLOCK_SIZE + threadIdx.x%blockDim.x;

    if(firstRow < m && secondCol < n){
        float sum = 0.0f;
        for(int i=0; i<k; i++){
            sum += A[firstRow * k + i] * B[i * n + secondCol];
        }
        C[firstRow * n + secondCol] = sum;
    }
}

void invoke_rowmajor_matmul(float *A, float *B, float *C, int m, int k, int n){
    dim3 blockSize(BLOCK_SIZE*BLOCK_SIZE);
    dim3 gridSize(CEILDIV(m, BLOCK_SIZE), CEILDIV(n, BLOCK_SIZE));

    myRowCoalesceKernel<<<gridSize, blockSize>>>(A, B, C, m, k, n);
}