#include "mykernels.cuh"
#include "../helpers/myhelpers.h"

template <const int BM, const int BN, const int BK, const int TM, const int TN> 
__global__ void my2dkernel(float *A, float *B, float *C, int m, int k, int n){
    int cRow = blockIdx.y;
    int cCol = blockIdx.x;

    const int totalCells = BM * BN;
    const int cellsPerThread = TM * TN;
    int threadsNeeded = totalCells/cellsPerThread;

    int threadCol = threadIdx.x % (BN/TN);
    int threadRow = threadIdx.x / (BN/TN);

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    int innerColA = threadIdx.x % BK;
    int innerRowA = threadIdx.x / BK;
    int rowsForA = threadsNeeded / BK;

    int innerColB = threadIdx.x % BN;
    int innerRowA = threadIdx.x / BK;
    int colsForB = threadsNeeded / BN;

    float results[TM * TN] = {0.0f};
    float regM[TM];
    float regN[TN];

    A += (cRow * BM * k);
    B += (cCol * BN);
    C += (cRow * BM * N) + (cCol * BN);

    for(int bkId = 0; bkId < k; bdId+=BK){
        for(int i=0; i<TM; i++){
            As[(innerRowA + i) * BK + innerColA] = A[(inerRowA + i) * k + innerColA];
        }
        for(int i=0; i<TN; i++){
            Bs[(innerRowB + i) * BN + innerColB] = B[(innerRowB + i) * n + innerColB];
        }
        __syncthreads();

        A += BK;
        B += BK * n;

        for(int l=0;l<BK;l++){
            for(int i=0; i<TM; i++){
                regM[i] = As[(threadRow * TM + i) * BK + l];
            }
            for(int i=0; i<TN; i++){
                regN[i] = Bs[l * BN + threadCol * TN + i];
            }
            for(int i=0; i<TM; i++){
                for(int j=0; j<TN; j++){
                    results[i * TN + j] += regM[i] * regN[j];
                }
            }
        }
        __syncthreads();
    }
    for(int i=0; i<TM; i++){
        for(int j=0; j<TN; j++){
            C[(threadRow * TM + i) * N + (threadCol * TN + j)] = results[i * TN + j];
        }
    }
}

void invoke_2D_tiled_matmul(float *A, float *B, float *C, int m, int k, int n){
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;

    dim3 gridSize(CEILDIV(N, BN),CEILDIV(M, BM));
    dim3 blockSize((BM * BN)/(TM * TN));
    my2dkernel<BM, BN, BK, TM, TN><<<gridSize, blockSize>>>(A, B, C, m, k, n);
}