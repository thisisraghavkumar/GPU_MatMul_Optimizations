#include "mykernels.cuh"

__global__ void shared_memory_kernel(float *A, float *B, float *C, int m, int k, int n){
    int frameRow = blockIdx.x;
    int frameCol = blockIdx.y;
    
    __shared__ float As[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE * BLOCK_SIZE];

    int threadRow = threadIdx.x / BLOCK_SIZE;
    int threadCol = threadIdx.x % BLOCK_SIZE;

    // move A to the first row of the tile this thread's block is processing
    A += (frameRow * BLOCK_SIZE) * k;
    // move B to the first column of the tile this thread's block is processing
    B += (frameCol * BLOCK_SIZE);
    // move C to the first row and first column of the tile this thread's block is processing
    C += (frameRow * BLOCK_SIZE) * n + (frameCol * BLOCK_SIZE);

    float sum = 0.0f;
    for(int idx=0; idx<k; idx+=BLOCK_SIZE){
        // load the cell corresponding to this thread into shared memory
        // the shared memory is BLOCK_SIZE x BLOCK_SIZE and indexed using threadRow and threadCol
        // A is already moved to the first row of the tile
        As[threadRow * BLOCK_SIZE + threadCol] = A[threadRow * k + threadCol];
        Bs[threadRow * BLOCK_SIZE + threadCol] = B[threadRow * n + threadCol];
        __syncthreads();
        // move A BLOCK_SIZE columns to the right
        A += BLOCK_SIZE;
        // move B BLOCK_SIZE rows down
        B += (BLOCK_SIZE * n);
        for(int i=0; i<BLOCK_SIZE; i++){
            sum += As[threadRow * BLOCK_SIZE + i] * Bs[i * BLOCK_SIZE + threadCol];
        }
        __syncthreads();
    }
    C[threadRow * n + threadCol] = sum;
}

void invoke_shared_memory_matmul(float *A, float *B, float *C, int m, int k, int n){
    dim3 gridSize(CEILDIV(m, BLOCK_SIZE), CEILDIV(n, BLOCK_SIZE));
    dim3 blockSize(BLOCK_SIZE * BLOCK_SIZE);

    shared_memory_kernel<<<gridSize, blockSize>>>(A,B,C,m,k,n);
}