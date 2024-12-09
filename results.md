
## Experiments

### Exp 1: sequential vs openmp vs cuda simple

config: (num_batch, sequence_length, embed_dim)

![exp1](charts/test1.svg)

config       |sequential|openmp  |cuda simple
-------------|----------|--------|-----------
(1,8,8)      |0.000025  |0.002438|0.00036    
(8,32,32)    |0.002254  |0.003467|0.000493   
(16,64,64)   |0.032223  |0.006979|0.001053   
(32,128,128) |0.495145  |0.037993|0.004239   
(64,256,256) |7.642602  |0.427509|0.026593   
(128,512,512)|120.431175|6.071292|0.203355   

Steady trend of accleration observed, though offset by CUDA's memory overlead at small size

### Exp 2: simple vs transposed vs tiling (thread) vs tiling (block) vs flash vs flash multi

![exp2](charts/test2.svg)

config       |cuda simple|cuda transposed|cuda tiling (thread)|cuda tiling (block)|cuda flash|cuda flash multi
-------------|-----------|---------------|--------------------|-------------------|----------|----------------
(1,8,8)      |0.00036    |0.000392       |0.000489            |0.000493           |0.000231  |0.000342        
(8,32,32)    |0.000493   |0.000528       |0.000582            |0.000584           |0.000463  |0.000627        
(16,64,64)   |0.001053   |0.001171       |0.001083            |0.001073           |0.002453  |0.002834        
(32,128,128) |0.004239   |0.005131       |0.00511             |0.007109           |0.01837   |0.009202        
(64,256,256) |0.026593   |0.034069       |0.037402            |0.031003           |0.329536  |0.01943         
(128,512,512)|0.203355   |0.262681       |0.364801            |0.253763           |8.471316  |2.837042        

Tranposed version generally took longer time than simple, possibly due to additional transposing overhead. Following variations (i.e. tiled) were then built upon non-transposed simple version

Tiled (block) outran tiled (thread) with larger reduction ratio;
yet are still on par with transposed version, hypothesized that requirement of tiling method on shared memory limited number of threads.

Flash attention scaled with larger overhead, investigated that tiling was coupled with embed dimension $d$, as oppsed to a pre-defined *tile_width*.

Flash multi gave halved speeds from being run on GPU server with 2 devices.

### Exp 3: tile size

![exp3](charts/test3.svg)

tile size|tiled (thread)|tiled (block)
---------|--------------|-------------
1x1      |23.954065     |12.061552    
4x4      |0.670721      |0.426662     
8x8      |0.367755      |0.268041     
16x16    |0.584292      |0.365835     
32x32    |1.467721      |0.811791    

Tiles versions were tested with various *tile_width*, 8x8 was the best on GeForce GTX TITAN X (Maxwell)

### Exp 4: memory allocation vs copying vs kernel

![exp4](charts/test4.svg)

config       |malloc  |memcpy_hd|kernel  |memcpy_dh
-------------|--------|---------|--------|---------
(1,8,8)      |0.092309|0.000073 |0.000401|0.000013 
(8,32,32)    |0.078525|0.000163 |0.000404|0.000022 
(16,64,64)   |0.078762|0.000554 |0.000417|0.000108 
(32,128,128) |0.079988|0.003436 |0.00041 |0.0035   
(64,256,256) |0.08066 |0.022566 |0.000427|0.008929 
(128,512,512)|0.084273|0.175471 |0.00042 |0.091274  

**memcpy_hd: memory copy host to device*

Operations were timed separately, memory allocation and copying between host and device used most compute, aligning well with the IO-bound insight raised in [FlashAttention](https://github.com/Dao-AILab/flash-attention).

### Exp 5: tiled (block) vs flash vs flash multi with unified memory 

![exp5](charts/test5.svg)

config       |cuda tiling (block)|cuda flash|cuda flash multi
-------------|-------------------|----------|----------------
(1,8,8)      |0.000477           |0.000469  |0.00024         
(8,32,32)    |0.000455           |0.000463  |0.000277        
(16,64,64)   |0.000485           |0.000478  |0.000301        
(32,128,128) |0.0005             |0.000502  |0.000287        
(64,256,256) |0.000638           |0.000662  |0.000281        
(128,512,512)|0.001211           |0.001185  |0.000299        

Unified memory with cudaMallocManaged() as opposed to conventional cudaMalloc() and cudaMemcpy();dynamic memory management eliminated part of memory overload, run time consequently shrank into kernel execution only.
