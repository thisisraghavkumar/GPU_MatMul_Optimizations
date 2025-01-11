#include "mykernels.cuh"

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void myvectorizedkernel(float *A, float *B, float *C, int m, int k, int n){
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    const int threadCol = threadIdx.x % (BN/TN);
    const int threadRow = threadIdx.x / (BN/TN);

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    A += cRow * BM * k;
    B += cCol * BN;
    C += cRow * BM * n + cCol * BN;

    const uint innerRowA = threadIdx.x / (BK/4);
    const uint innerColA = threadIdx.x % (BK/4);
    const uint innerRowB = threadIdx.x / (BN/4);
    const uint innerColB = threadIdx.x % (BN/4);

    float threadResults[TM * TN] = {0.0};
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};

    for(uint bkId = 0; bkId < k; bkId += BK){
        float4 tmp = reinterpret_cast<float4 *>(&A[innerRowA * k + innerColA * 4])[0];
        As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
        As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
        As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
        As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;

        reinterpret_cast<float4 *>(&Bs[innerRowB * BN + innerColB * 4])[0] 
          = reinterpret_cast<float4 *>(&B[innerRowB * n + innerColB * 4])[0];
        __syncthreads();

        A += BK;
        B += BK * n;

        for(int l=0; l<BK; ++l){
            for(int i=0; i<TM; ++i){
                regM[i] = As[l * BM + threadRow * TM + i];
            }
            for(int i=0; i<TN; ++i){
                regN[i] = Bs[l * BN + threadCol * TN + i];
            }
            for(int i=0; i<TM; ++i){
                for(int j=0; j<TN; ++j){
                    threadResults[i * TN + j] += regM[i] * regN[j];
                }
            }
        }
        __syncthreads();
    }
    for(int i=0; i<TM; ++i){
        for(int j=0; j<TN; j += 4){
            float4 tmp;/* = reinterpret_cast<float4 *>(
                &C[(threadRow * TM + i) * n + threadCol * TN + j]
            )[0];*/
            tmp.x = threadResults[i * TN + j];
            tmp.y = threadResults[i * TN + j + 1];
            tmp.z = threadResults[i * TN + j + 2];
            tmp.w = threadResults[i * TN + j + 3];

            reinterpret_cast<float4 *>(
                &C[(threadRow * TM + i) * n + threadCol * TN + j]
            )[0] = tmp;
        }
    }
}

void invoke_vectorized_matmul(float *A, float *B, float *C, int m, int k, int n){
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;

    dim3 gridDim(CEILDIV(n,BN),CEILDIV(m,BM));
    dim3 blockDim((BM * BN)/(TM * TN));

    myvectorizedkernel<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(A, B, C, m, k, n);
}

template <const int BM, const int BN, const int BK, const int TM, const int TN> void invoke_parameterized_vectorized_matmul(float *A, float *B, float *C, int m, int k, int n){
    dim3 gridDim(CEILDIV(n, BN), CEILDIV(m, BM));
    dim3 blockDim((BM * BN)/(TM * TN));

    myvectorizedkernel<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(A, B, C, m, k, n);
}