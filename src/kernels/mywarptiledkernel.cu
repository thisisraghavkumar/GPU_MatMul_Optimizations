#include "mykernels.cuh"
#include "../helpers/myhelpers.h"

namespace wt
{
    template <const int BM, const int BN, const int BK, const int rowStrideA, const int rowStrideB>
    __device__ void loadInputMatrices(const float *A, const float *B, float *As, float *Bs, int k, int n, int innerRowA, int innerColA, int innerRowB, int innerColB){
        for(uint i=0; i<BM; i+=rowStrideA){
            const float4 tmp = reinterpret_cast<const float4 *>(&A[(innerRowA + i) * k + innerColA * 4])[0];
            As[(innerColA * 4 + 0) * BM + innerRowA + i] = tmp.x;
            As[(innerColA * 4 + 1) * BM + innerRowA + i] = tmp.y;
            As[(innerColA * 4 + 2) * BM + innerRowA + i] = tmp.z;
            As[(innerColA * 4 + 3) * BM + innerRowA + i] = tmp.w;
        }
        for(uint i=0; i<BK; i+=rowStrideB){
            reinterpret_cast<float4 *>(&Bs[(innerRowB + i) * BN + innerColB * 4])[0] =
                reinterpret_cast<const float4 *>(&B[(innerRowB + i) * n + innerColB * 4])[0];
        }
    }

    template <const int BM, const int BN, const int BK, const int WM, const int WN, const int WMITER, 
              const int WNITER, const int WSUBM, const int WSUBN, const int TM, const int TN>
    __device__ void computePartialProduct(
        float *regM, float *regN, float *threadResults, const float *As, const float *Bs,
        const uint warpRow, const uint warpCol, const uint threadRowInWarp, const uint threadColInWarp
    ){
        for(int l=0; l<BK; ++l){
            for(int i=0; i<WMITER; ++i){
                for(int j=0; j < TM; ++j){
                    regM[i * TM + j] = 
                        As[(l * BM) + warpRow * WM + i * WSUBM + threadRowInWarp * TM + j];
                }
            }
            for(int i=0; i<WNITER; ++i){
                for(int j=0; j<TN; ++j){
                    regN[i * TN + j] = 
                        Bs[(l * BN) + warpCol * WN + i * WSUBN + threadColInWarp * TN + j];
                }
            }

            for(int i=0; i<WMITER; ++i){
                for(int j=0; j<WNITER; ++j){
                    for(int ii=0; ii<TM; ++ii){
                        for(int jj=0; jj<TN; ++jj){
                            threadResults[(i * TM + ii) * (WNITER * TN) + (j * TN) ++ jj] +=
                                regM[i * TM + ii] * regN[j * TN + jj];
                        }
                    }
                }
            }
        }
    }
} // namespace wt

template <const int BM, const int BN, const int BK, const int WM, const int WN, const int WNITER
            const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS) mywarptilingkernel(float *A, float *B, float *C, int m, int k, int n){
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;
    const uint wardpIdx = threadIdx.x / WARP_SIZE;
    const uint warpCol = warpIdx % (BN/WN);
    const uint warpRow = warpIdx / (BN/WN);

    constexpr int WMITER = (WM * WN) / (WARP_SIZE * TM * TN * WNITER);
    constexpr int WSUBM = WM / WMITER;
    constexpr int WSUBN = WN / WNITER;

    const uint threadIdxInWarp = threadIdx.x % WARP_SIZE;
    const uint threadColInWarp = threadIdxInWarp % (WSUBN/TN);
    const uint threadRowInWarp = threadIdxInWarp / (WSUBN/TN);

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    A += cRow * BM * k;
    B += cCol * BN;
    C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

    const uint innerRowA = threadIdx.x / (BK/4);
    const uint innerColA = threadIdx.x % (BK/4);
    const uint strideA = (NUM_THREADS * 4) / BK;
    const uint innerRowB = threadIdx.x / (BN/4);
    const uint innerColB = threadIdx.x % (BN/4);
    const uint strideB = (NUM_THREADS * 4) / BN;

    float threadResults[WMITER * WNITER * TM * TN] = {0.0};
    float regM[WMITER * TM] = {0.0};
    float regN[WNITER * TN] = {0.0};

    for(int bkId = 0; bkId < k; bkIdx += BK){
        wt::loadInputMatrices<BM, BN, BK, strideA, strideB>(A,B,As,Bs,k,n,innerRowA,innerColA,innerRowB,innerColB);
        __syncthreads();
        wt::computePartialProduct<BM,BN,BK,WM,WN,WMITER,WNITER,WSUBM,WSUBN,TM,TN>(regM,regN,threadResults,As,Bs,warpRow,warpCol,threadRowInWarp,threadColInWarp);
        __syncthreads();
    }
    for(uint i=0; i<WMITER; ++i){
        for(uint j=0; j<WNITER; ++j){
            float *C_interim = C + (i * WSUBM) * n + j * WSUBN;
            for(uint ii=0; ii<TM; ++ii){
                for(uint jj=0; jj<TN; jj+=4){
                    float4 tmp;
                    const int idx = (i * TM + ii) * (WNITER *TN) + j * TN + jj;
                    tmp.x = threadResults[idx+0];
                    tmp.y = threadResults[idx+1];
                    tmp.z = threadResults[idx+2];
                    tmp.w = threadResults[idx+3];
                    reinterpret_cast<float4 *>(
                        &C_interim[(threadRowInWarp * TM + ii) * n + threadColInWarp * TN + jj]
                    )[0] = tmp;
                }
            }
        }
    }
}

void invoke_warptiled_matmul(float *A, float *B, float *C, int m, int k, int n){
    const uint NUM_THREADS = 128;
    const uint BN = 128;
    const uint BM = 128;
    const uint BK = 16;
    const uint WN = 64;
    const uint WM = 64;
    const uint WNITER = 4;
    const uint TN = 4;
    const uint TM = 8;
    bim3 blockDim(NUM_THREADS);

    constexpr NUM_WARPS = NUM_THREADS / 32;
    constexpr WMITER = (WM * WN)/(32 * TM * TN * WNITER);
    dim3 gridDim(CEILDIV(n,BN),CEILDIV(m,BM));
    mywarptilingkernel<BM,BN,BK,WM,WN,WNITER,TM,TN,NUM_THREADS><<<gridDim,blockDim>>>(A,B,C,m,k,n);
}