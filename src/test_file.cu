#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <random>



#define BLOCK_SIZE 32
#define MMM 1024LL
#define MMK 1024LL
#define MMN 1024LL
#define sF sizeof(float)

#define CEIL_DIV(dividend, divisor) ((dividend + divisor - 1) / divisor)

void populate_array(float *arr, int size, std::mt19937 &gen, std::uniform_real_distribution<float> &dis)
{
    for (int i = 0; i < size; i++)
    {
        arr[i] = dis(gen);
    }
}

// Kernel 2
template <const uint BLOCKSIZE>
__global__ void sgemm_global_mem_coalesce(int M, int N, int K, float alpha,
                                          const float *A, const float *B,
                                          float beta, float *C) {
  const int cRow = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  const int cCol = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  // if statement is necessary to make things work under tile quantization
  if (cRow < M && cCol < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[cRow * K + i] * B[i * N + cCol];
    }
    C[cRow * N + cCol] = alpha * tmp + beta * C[cRow * N + cCol];
  }
}

void run_sgemm_coalesce(int M, int N, int K, float alpha, float *A, float *B,
                        float beta, float *C) {
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  dim3 blockDim(32 * 32);
  sgemm_global_mem_coalesce<32>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

template <const uint BLOCKSIZE>
__global__ void myRowCoalesceKernel(float *A, float *B, float *C, int m, int k, int n){
    int firstRow = blockIdx.x * BLOCKSIZE + threadIdx.x/BLOCKSIZE;
    int secondCol = blockIdx.y * BLOCKSIZE + threadIdx.x%BLOCKSIZE;

    if(firstRow < m && secondCol < n){
        float sum = 0.0f;
        for(int i=0; i<k; ++i){
            sum += A[firstRow * k + i] * B[i * n + secondCol];
        }
        C[firstRow * n + secondCol] = sum;
    }
}

void invoke_rowmajor_matmul(float *A, float *B, float *C, int m, int k, int n){
    dim3 blockSize(BLOCK_SIZE*BLOCK_SIZE);
    dim3 gridSize(CEIL_DIV(m, BLOCK_SIZE), CEIL_DIV(n, BLOCK_SIZE));

    myRowCoalesceKernel<32><<<gridSize, blockSize>>>(A, B, C, m, k, n);
}

// Kernel 4
template <const int BM, const int BK, const int BN, const int TM> 
__global__ void myonedtiledkernel(float *A, float *B, float *C, int m, int k, int n){
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
        C[(threadRow*TM+i)*n + threadCol] = results[i];
    }
}

void invoke_oned_tiled_matmul(float *A, float *B, float *C, int m, int k, int n){
    const int BM = 64;
    const int BN = 64;
    const int BK = 8;
    const int TM = 8;
    dim3 gridDimension(CEILDIV(n, BN), CEILDIV(m, BM));
    dim3 blockDimension((BN * BM)/TM);
    myonedtiledkernel<BM,BK,BN,TM><<<gridDimension,blockDimension>>>(A, B, C, m, k, n);
}

template <const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm1DBlocktiling(int M, int N, int K, float alpha,
                                   const float *A, const float *B, float beta,
                                   float *C) {
  // If we flip x and y here we get ~30% less performance for large matrices.
  // The current, 30% faster configuration ensures that blocks with sequential
  // blockIDs access columns of B sequentially, while sharing the same row of A.
  // The slower configuration would share columns of A, but access into B would
  // be non-sequential. So the faster configuration has better spatial locality
  // and hence a greater L2 hit rate.
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // each warp will calculate 32*TM elements, with 32 being the columnar dim.
  const int threadCol = threadIdx.x % BN;
  const int threadRow = threadIdx.x / BN;

  // allocate space for the current blocktile in SMEM
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // todo: adjust this to each thread to load multiple entries and
  // better exploit the cache sizes
  assert(BM * BK == blockDim.x);
  assert(BN * BK == blockDim.x);
  const uint innerColA = threadIdx.x % BK; // warp-level GMEM coalescing
  const uint innerRowA = threadIdx.x / BK;
  const uint innerColB = threadIdx.x % BN; // warp-level GMEM coalescing
  const uint innerRowB = threadIdx.x / BN;

  // allocate thread-local cache for results in registerfile
  float threadResults[TM] = {0.0};

  // outer loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // populate the SMEM caches
    As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
    Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
    __syncthreads();

    // advance blocktile
    A += BK;
    B += BK * N;

    // calculate per-thread results
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // we make the dotproduct loop the outside loop, which facilitates
      // reuse of the Bs entry, which we can cache in a tmp var.
      float tmpB = Bs[dotIdx * BN + threadCol];
      for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        threadResults[resIdx] +=
            As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
      }
    }
    __syncthreads();
  }

  // write out the results
  for (uint resIdx = 0; resIdx < TM; ++resIdx) {
    C[(threadRow * TM + resIdx) * N + threadCol] =
        alpha * threadResults[resIdx] +
        beta * C[(threadRow * TM + resIdx) * N + threadCol];
  }
}

void runSgemm1DBlocktiling(int M, int N, int K, float alpha, float *A, float *B,
                           float beta, float *C) {
  const uint BM = 64;
  const uint BN = 64;
  const uint BK = 8;
  const uint TM = 8;
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  dim3 blockDim((BM * BN) / TM);
  sgemm1DBlocktiling<BM, BN, BK, TM>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

int main(){
    int m = MMM;
    int n = MMN;
    int k = MMK;
    int sizeA = m * k;
    int sizeB = k * n;
    int sizeC = m * n;

    int warmup_runs = 5;
    int measurement_runs = 50;
    long long numoperations = 2LL * m * n * k;
    float *h_A, *h_B, *h_C, *h_C_cublas; //, *h_C_ref;
    float *d_A, *d_B, *d_C;
    float myelapsed_time, refelapsed_time;
    cudaEvent_t mybeg, myend, refbeg, refend;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(50.0, 25.0);

    h_A = new float[sizeA]();
    h_B = new float[sizeB]();
    h_C = new float[sizeC]();
    h_C_cublas = new float[sizeC]();
    // h_C_ref = new float[sizeC]();
    cudaMalloc(&d_A, sF * sizeA);
    cudaMalloc(&d_B, sF * sizeB);
    cudaMalloc(&d_C, sF * sizeC);

    // float valuesA[] = {1.0f, 2.0f, 5.0f, 4.0f, 6.0f, 8.0f, 2.0f, 3.0f, 2.0f};
    populate_array(h_A, sizeA, gen, dis);
    // std::copy(valuesA,valuesA+9,h_A);
    // float valuesB[] = {1.0f, 0.0f, 2.0f, 2.0f, 1.0f, 1.0f, 8.0f, 2.0f, 4.0f};
    populate_array(h_B, sizeB, gen, dis);
    // std::copy(valuesB,valuesB+9,h_B);
    cudaMemcpy(d_A, h_A, sF * sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sF * sizeB, cudaMemcpyHostToDevice);
    // naive_mat_mul(h_A, h_B, h_C_ref, m, k, n);
    cudaEventCreate(&mybeg);
    cudaEventCreate(&myend);
    cudaEventCreate(&refbeg);
    cudaEventCreate(&refend);

    // Calling kernel 2
    invoke_rowmajor_matmul(d_A, d_B, d_C, m, k, n);
    run_sgemm_coalesce(m,n,k,1.0f,d_A,d_B,0.0f,d_C);
    cout<<"Kernel 2 : Global coalesced memory\n";
    cudaEventRecord(refbeg);
    for(int i=0; i<measurement_runs; ++i){
        run_sgemm_coalesce(m,n,k,1.0f,d_A,d_B,0.0f,d_C);
    }
    cudaEventRecord(refend);
    cudaEventSynchronize(refbeg);
    cudaEventSynchronize(refend);
    cudaEventElapsedTime(&refelapsed_time, refbeg, refend);
    std::cout<<"Ref implementation: "<<refelapsed_time<<" / "<<measurement_runs<<" = "<<refelapsed_time/measurement_runs<<"\n";

    cudaEventRecord(mybeg);
    for(int i=0; i<measurement_runs; ++i){
        invoke_rowmajor_matmul(d_A, d_B, d_C, m, k, n);
    }
    cudaEventRecord(myend);
    cudaEventSynchronize(mybeg);
    cudaEventSynchronize(myend);
    cudaEventElapsedTime(&myelapsed_time, mybeg, myend);
    std::cout<<"My implementation: "<<myelapsed_time<<" / "<<measurement_runs<<" = "<<myelapsed_time/measurement_runs<<"\n";
    
    // Calling kernel 4
    runSgemm1DBlocktiling(m,n,k,1.0f,d_A,d_B,0.0f,d_C);
    invoke_oned_tiled_matmul(d_A, d_B, d_C, m, k, n);
    cout<<"Kernel 4 : 1D Block tiling\n";
    cudaEventRecord(refbeg);
    for(int i=0; i<measurement_runs; ++i){
        runSgemm1DBlocktiling(m,n,k,1.0f,d_A,d_B,0.0f,d_C);
    }
    cudaEventRecord(refend);
    cudaEventSynchronize(refbeg);
    cudaEventSynchronize(refend);
    cudaEventElapsedTime(&refelapsed_time, refbeg, refend);
    std::cout<<"Ref implementation: "<<refelapsed_time<<" / "<<measurement_runs<<" = "<<refelapsed_time/measurement_runs<<"\n";

    cudaEventRecord(mybeg);
    for(int i=0; i<measurement_runs; ++i){
        invoke_oned_tiled_matmul(d_A, d_B, d_C, m, k, n);
    }
    cudaEventRecord(myend);
    cudaEventSynchronize(mybeg);
    cudaEventSynchronize(myend);
    cudaEventElapsedTime(&myelapsed_time, mybeg, myend);
    std::cout<<"My implementation: "<<myelapsed_time<<" / "<<measurement_runs<<" = "<<myelapsed_time/measurement_runs<<"\n";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_cublas;
}