# parallelize-attention
Project for CSCI-GA.3033 GPUs: Architecture and Programming. Parallization of self-attention with OpenMP and CUDA.

## Implementations
1. Sequential (<a href="src/attention_sequential.cpp">attention_sequential.cpp</a>)
2. OpenMP (<a href="src/attention_openmp.cpp">attention_openmp.cpp</a>)
3. CUDA (<a href="src/attention_cuda.cu">attention_cuda.cu</a>, <a href="src/attention_cuda_tile.cu">attention_cuda_tile.cu</a>, <a href="src/attention_flash.cu">attention_flash.cu</a>)

Specifically, for CUDA, 6 different versions were implemented:

1. simple
2. transposed // [coalesced transpose](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/)
3. tiled (thread) //  [tiling](https://nichijou.co/cuda7-tiling/) 
4. tiled (block) // tiling with [block reduction](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)
5. flash // [flash-attention-minimal](https://github.com/tspeterkim/flash-attention-minimal)
6. flash multi // extension of multi-device

Data
- random matrices with dimension of (num_batch, sequence_length, embed_dim)

Evaluation
- compute time of memory allocation, copying (between host and device), and kernel execution

Assumption (for more simplified formulation):
-  *square* matrices as input
-  num_head = 1

## Experiments

config: (num_batch, sequence_length, embed_dim)

![exp1](charts/test1.svg)
Steady trend of accleration observed, though offset by CUDA's memory overlead at small size

![exp2](charts/test2.svg)

Tranposed version generally took longer time than simple, possibly due to additional transposing overhead. Following variations (i.e. tiled) were then built upon non-transposed simple version

Tiled (block) outran tiled (thread) with larger reduction ratio;
yet are still on par with transposed version, hypothesized that requirement of tiling method on shared memory limited number of threads.

Flash attention scaled with larger overhead, investigated that tiling was coupled with embed dimension $d$, as oppsed to a pre-defined *tile_width*.

Flash multi gave halved speeds from being run on GPU server with 2 devices.

![exp4](charts/test4.svg)

Tiles versions were tested with various *tile_width*, 8x8 was the best on GeForce GTX TITAN X (Maxwell)

![exp5](charts/test5.svg)

Operations were timed separately, memory allocation and copying between host and device used most compute, aligning well with the IO-bound insight raised in [FlashAttention](https://github.com/Dao-AILab/flash-attention).

![exp3](charts/test3.svg)

Unified memory with cudaMallocManaged() as opposed to conventional cudaMalloc() and cudaMemcpy();dynamic memory management eliminated part of memory overload, run time consequently shrank into kernel execution only.

## Executions

```bash
> cd src
> make clean && make && make run > ../output/output_toys.txt # results of toy example with config (2,2,2)
> run.sh # results of experiments
```
