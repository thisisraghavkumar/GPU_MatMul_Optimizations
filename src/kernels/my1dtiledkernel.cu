#include "mykernels.cuh"
#include "../helpers/myhelpers.h"

template <const int BM, const int BK, const int BN, const int TM> 
__global__ void my1dtiledkernel(float *A, float *B, float *C, int m, int k, int n){
    int frameRow = blockIdx.y;
    int frameCol = blockIdx.x;

    __shared__ float As[BM*BK];
    __shared__ float Bs[BK*BN];

    int threadRow = threadIdx.x / BN;
    int threadCol = threadIdx.x % BN;
    int ARow = threadIdx.x / BK;
    int ACol = threadIdx.x % BK;
    int BRow = threadIdx.x / BN;
    int BCol = threadIdx.x % BN;

    A += (frameRow * BM * k);
    B += (frameCol * BN);
    C += (frameRow * BM * n) + (frameCol * BN);

    float results[TM] = {0.0f};
    for(int idx=0;idx<k;idx+=BK){
        As[ARow * BK + ACol] = A[ARow * k + ACol];
        Bs[BRow * BN + BCol] = B[BRow * n + BCol];
        __syncthreads();

        A += BK;
        B += (BK * n);
        for(int l=0; l<BK; l++){
            float temp = Bs[l * BN + threadCol];
            for(int i=0; i<TM; i++){
                results[i] += As[(threadRow * TM + i) * BK + l] * temp;
            }
        }
        __syncthreads();
    }
    for(int i=0; i<TM;i++){
        C[(threadRow*TM+i)*BN + threadCol] = results[i];
    }
}

void invoke_1D_tiled_matmul(float *A, float *B, float *C, int m, int k, int n){
    dim3 gridDimension(CEILDIV(n, BN), CEILDIV(m, BM));
    dim3 blockDimension((BN * BM)/TM);
    my1dtiledkernel<64,8,64,8><<<gridDimension,blockDimension>>>(A, B, C, m, k, n);
}