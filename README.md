# Matrix Mutliplication CUDA Kernel Optimization

This project is my first attempt at learning a. how to write CUDA kernels and b. how to optimize them. I'm following the guidance available in these [work notes of Simon Boehm](https://siboehm.com/articles/22/CUDA-MMM), [his github repository](https://github.com/siboehm/SGEMM_CUDA/blob/60cba6f9b20a198116c76f18de8047f44df8c8b8/sgemm.cu#L11) and [this YouTube tutorial by Elliot Alridge](https://www.youtube.com/watch?v=86FAWCzIe_4)

I'm using the [NVIDIA RTX A4500](https://resources.nvidia.com/en-us-briefcase-for-datasheets/nvidia-rtx-a4500-dat?ncid=no-ncid) GPU which has a peak single precision performance of 23.7 TFLOPS, which is what I'll try to be achieving with Matrix Multiplication.

![GPUSpecs](/Images/RTX_A4500_specs.png)

Matrix size: m = n = k = 2048

Total number of operations: 17,179,869,184
(Calculated as `2 * m * n * k`, because each of the `m x n` elements in the result matrix requires multiplying `k` elements in a row of first matrix with the corresponding element in a column of the second matrix and then summing the `k` products which takes `k-1` addition operations. Therefore, each element of the result matrix requires `k + k -1 = 2k - 1` operations, or roughly `2k` operations.)

Warm up iterations: 5

Measurement interations: 50


## Kernels

|Kernel|Description|Time per iteration (in ms)|GFLOP/S|
|------|-----------|------------|-----|
|CuBLAS|cublasSGMEM library function|1.34|12789.50|
|Naive|Each thread produces one output cell by reading the input matrices from global memory|12.38575|1387.06|
|Row coalesce|Same as naive but consecutive threads use the same row of the first matrix if possible|16.11|1066.65|
|Shared memory|Each thread loads two operands in the shared memory and then produces a partial product. The final value of a cell is obtained by a thread by summing the partial product. Each block of threads produce BM x BN cells.|9.59|1791.49|
|1D Tiled|Each thread loads 2 operands in the shared memory and then produces TM partial products for cells in TM rows. The final TM results are obtained by a thread by summing the partial products over K/BK tiles.|3.09|5547.98|
|2D Tiled|Each thread loads 2 operands in the shared memory and then produces TM x TN partial products in TM rows and TN columns. The final TM * TN results are obtained by a thread by summing the partial products over K/BK tiles.|1.80|9514.67|
|Vectorized|Each threads loads 8 operands using 128 bytes load instruction instead of vanilla load instruction which only loads 32 bytes and then produces TM x TN parial products in TM rows and TN columns. The final TM * TN results are obtained by a thread by summing the partial products over K/BK tiles.|1.50|11414.32|
|Warp tiled|Each thread loads 8 operands using 128 bytes load instructions same as above and produces TM x TN x WMITER x WNITER partial products. The final results are obtained by a thread by summing the partial products over K/BK tiles.|1.63|10532.23|

## Observations

1. The largest jump in performance is obtained when a thread produces multiple cells of the output matrix. This can be attributed to high usage of a byte once it is loaded, this is called arithmetic intensity. Increasing the number of cells produced from shared memory kernel (1) to 1D tiled kernel (TM) to 2D tiled kernel (TM * TN) to warp tiled kernel (TM * TN * WMITER * WNITER) increased performance albeit the gains diminished i.e. the jump from 2D tiled kernel to warped tiled kernel was lower than the jump from 1D tiled kernel to 2D tiled kernel which was in turn less than the jump in performance from shared memory kernel to 1D tiled kernel.
2. The second remarkable performance improvement was obtained by using 128 byte loads instead of 32 byte loads.
# Appendix: Illustrations

## Shared Memory Kernel Visualization

![image](/Images/SharedMemKernelViz.png)
A block in the grid produces the cells in the output that lie in a frame of size BLOCK_SIZE x BLOCK_SIZE which starts at blockIdx.x and blockIdx.y (remember the block is 1D, the grid is still 2D). It starts with a window of same size as the frame and computes a matrix mutliplication on the area exposed when the window is superimposed on the input matrices. Once computes the window is slid to the right on the first matrix and downwards for the second matrix.