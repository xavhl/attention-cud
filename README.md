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

Assumption (for simplified formulation):
-  *square* matrices as input
-  num_head = 1

## Experiments

see <a href="results.md">results.md</a>

## Executions

```bash
> cd src
> make clean && make && make run > ../output/output_toys.txt # results of toy example with config (2,2,2)
> run.sh # results of experiments
```
