# Matrix Mutliplication CUDA Kernel Optimization

This project is my first attempt at learning a. how to write CUDA kernels and b. how to optimize them. I'm following the guidance available in these [work notes of Simon Boehm](https://siboehm.com/articles/22/CUDA-MMM), [his github repository](https://github.com/siboehm/SGEMM_CUDA/blob/60cba6f9b20a198116c76f18de8047f44df8c8b8/sgemm.cu#L11) and [this YouTube tutorial by Elliot Alridge](https://www.youtube.com/watch?v=86FAWCzIe_4)

I'm using the [NVIDIA RTX A4500](https://resources.nvidia.com/en-us-briefcase-for-datasheets/nvidia-rtx-a4500-dat?ncid=no-ncid) GPU which has a peak single precision performance of 23.7 TFLOPS, which is what I'll try to be achieving with Matrix Multiplication.

![GPUSpecs](/Images/RTX_A4500_specs.png)

Matrix size: m = n = k = 1024

Total number of operations: 2,147,483,648
(Calculated as `2 * m * n * k`, because each of the `m x n` elements in the result matrix requires multiplying `k` elements in a row of first matrix with the corresponding element in a column of the second matrix and then summing the `k` products which takes `k-1` addition operations. Therefore, each element of the result matrix requires `k + k -1 = 2k - 1` operations, or roughly `2k` operations.)

Warm up iterations: 5

Measurement interations: 50


## Kernels

|Kernel|Description|Time elapsed (in ms)|GFLOPS|
|------|-----------|------------|-----|
|CuBLAS|cublasSGMEM library function|0.17196|12488.62|
|Naive|Every thread identifies the first table's row and the second table's column it has to multiply using `threadIdx.x` and `threadIdx.y` respectively and the grid is two-dimensional of size 32 x 32|1.5337|1400.20|
|Row coalesce|Threads in the same warp process the same row of the first matrix improving reuse. Block is one dimensional.|1.24748|1721.46|
|Shared memory|Final result is calulcated by computing partial products of smaller tiles and summing them. For calculating partial products the data needed by each tile is first cached into the shared memory and then threads in the block can resue that data. Uses thread synchroinzation. One dimensional block.|1.19962|1790.14|

# Appendix: Illustrations

## Shared Memory Kernel Visualization

![image](/Images/SharedMemKernelViz.png)
A block in the grid produces the cells in the output that lie in a frame of size BLOCK_SIZE x BLOCK_SIZE which starts at blockIdx.x and blockIdx.y (remember the block is 1D, the grid is still 2D). It starts with a window of same size as the frame and computes a matrix mutliplication on the area exposed when the window is superimposed on the input matrices. Once computes the window is slid to the right on the first matrix and downwards for the second matrix.