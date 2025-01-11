#include "mykernels.cuh"

__global__ void mynaivekernel(float *A, float *B, float *C, int m, int k, int n){
    int firstrow = blockIdx.y * blockDim.y + threadIdx.y;
    int secondcol = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(firstrow < m && secondcol < n){
        float sum = 0.0f;
        for(int i=0; i<k; i++){
            sum += A[firstrow * k + i] * B[i*n + secondcol];
        }
        C[firstrow * n + secondcol] = sum;
    }
}

void invoke_naive_matmul(float *A, float *B, float *C, int m, int k, int n){
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(CEILDIV(n, BLOCK_SIZE), CEILDIV(m, BLOCK_SIZE));

    mynaivekernel<<<gridSize, blockSize>>>(A, B, C, m, k, n);
}
