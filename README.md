# Matrix Mutliplication CUDA Kernel Optimization

This project is my first attempt at learning a. how to write CUDA kernels and b. how to optimize them. I'm following the guidance available in these [work notes of Simon Boehm](https://siboehm.com/articles/22/CUDA-MMM), [his github repository](https://github.com/siboehm/SGEMM_CUDA/blob/60cba6f9b20a198116c76f18de8047f44df8c8b8/sgemm.cu#L11) and [this YouTube tutorial by Elliot Alridge](https://www.youtube.com/watch?v=86FAWCzIe_4)

I'm using the [NVIDIA RTX A4500](https://resources.nvidia.com/en-us-briefcase-for-datasheets/nvidia-rtx-a4500-dat?ncid=no-ncid) GPU which has a peak single precision performance of 23.7 TFLOPS, which is what I'll try to be achieving with Matrix Multiplication.

![GPUSpecs](/Images/RTX_A4500_specs.png)

Matrix size used:

Warm up iterations:

Measurement interations:

Total number of operations:
## Kernels

|Kernel|Description|Time elapsed|FLOPS|
|------|-----------|------------|-----|
|Naive|Every thread identifies the first table's row and the second table's column it has to multiply using `threadIdx.x` and `threadIdx.y` respectively and the grid is two-dimensional of size 32 x 32|to be found|to be found|