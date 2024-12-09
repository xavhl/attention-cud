#include <unistd.h>
#include "../matrix_io.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_fp16.h>
using namespace std;

#define NEG_INF -1e9

bool arg_print = false;
bool arg_reduced = false;
bool arg_mask = false;
bool arg_test = false;
bool arg_unified = false;
bool arg_test_separate = false;
int arg_tile_width = -1;
int tile_width = 1;
int num_batch = 1;
int embed_dim = 1;
int seq_length = 1;
char mode = 's'; // single / reduced

void proc_arg(int argc, char **argv) {
    int c;
    while ((c = getopt(argc, argv, "ruktpm:b:e:s:w:")) != -1) {
        switch (c) {
            case 'm':
                sscanf(optarg, " %c", &mode);
                break;
            case 'k':
                arg_mask = true;
                break;
            case 'u':
                arg_unified = true;
                break;
            case 'p':
                arg_test_separate = true;
                break;
            case 'r':
                arg_reduced = true;
                break;
            case 't':
                arg_test = true;
                arg_print = true;
                break;
            case 'b':
                sscanf(optarg, " %d", &num_batch);
                break;
            case 'e':
                sscanf(optarg, " %d", &embed_dim);
                break;
            case 's':
                sscanf(optarg, " %d", &seq_length);
                break;
            case 'w':
                sscanf(optarg, " %d", &arg_tile_width);
                break;
            default: // bad or unknown option
                exit(1);
                break;
        }
    }
}

__global__ void normalize_kernel(float* matrix, int b, int m, int n, float norm_const) {
    matrix[(blockIdx.z * blockDim.z + threadIdx.z)*m*n+(blockIdx.x*blockDim.x+threadIdx.x)*n+(blockIdx.y*blockDim.y+threadIdx.y)] /= norm_const;
}

__global__ void normalize_kernel_rowwise(float* matrix, int b, int m, int n, float* row_norm_const) {
    int batch = blockIdx.z * blockDim.z + threadIdx.z;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    matrix[batch*m*n+row*n+col] /= row_norm_const[batch*m+row]; 
}

__global__ void normalize_kernel_rowwise_reduced(float* matrix, int b, int m, int n, float* row_norm_const) {
    int batch = blockIdx.z * blockDim.z + threadIdx.z;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    matrix[batch*m*n+row*n+col] /= row_norm_const[batch*m*n+row*n]; 
}

__global__ void get_max_rowwise(float* matrix, int b, int m, int n, float* rowwise_buffer) {
    int batch = blockIdx.z * blockDim.z + threadIdx.z;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int j;
    
    float max_val = NEG_INF;

    // aggregation by thread 0
    if (col==0) {
        for (j = 0; j < n; j++) { max_val = fmaxf(max_val, matrix[batch*m*n+row*n+j]); }
        rowwise_buffer[batch*m+row] = max_val;
    }
}

__global__ void exp_sum(float* matrix, int b, int m, int n, float* rowwise_buffer) {
    int batch = blockIdx.z * blockDim.z + threadIdx.z;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int j;
    
    float sum = 0.0f;

    // aggregation by thread 0
    if (col==0) {
        for (j = 0; j < n; j++) { 
            matrix[batch*m*n+row*n+j] = exp(matrix[batch*m*n+row*n+j] - rowwise_buffer[batch*m+row]); 
            sum += matrix[batch*m*n+row*n+j]; 
        }
        rowwise_buffer[batch*m+row] = sum;
    }
}

__global__ void matrix_transpose_kernel(float* a, float* aT, int b, int m, int n) {
    int batch = blockIdx.z * blockDim.z + threadIdx.z;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    aT[batch*m*n+col*m+row] = a[batch*m*n+row*n+col];
}

__global__ void matrix_multiply_kernel(float* a, float* b, float* c, int nb, int m, int r, int n, int tile_width) {
    int batch = blockIdx.z * blockDim.z + threadIdx.z;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    extern __shared__ float shared[];
    float sum = 0.0f;
    
    for (int t=0; t < r/tile_width; t++) {
        shared[threadIdx.x*tile_width+threadIdx.y] = a[batch*m*n+row*n+t*tile_width+threadIdx.x];
        shared [tile_width*tile_width+threadIdx.x*tile_width+threadIdx.y] = b[batch*m*n+(t*tile_width+threadIdx.y)*m+col];
        __syncthreads();
    
        #pragma unroll
        for (int k = 0; k < tile_width; k++) { sum += shared[threadIdx.x*tile_width+k] * shared[tile_width*tile_width+k*tile_width+threadIdx.y]; }
        __syncthreads();
    }
    c[batch*m*n+row*n+col] = sum;
}

__global__ void matrix_mask_kernel(float* matrix, int* mask, int b, int m, int n) {
    int batch = blockIdx.z * blockDim.z + threadIdx.z;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (mask[batch*m*n+row*n+col] == 0.0f) { matrix[batch*m*n+row*n+col] = NEG_INF; }
}

void scaled_dot_product_attention_kernel_single(dim3 dimBlock, dim3 dimGrid,
    float* query, float* key, float* value, float* keyT, 
    float* mask, float* score, int b, int m, int n, float* output, float* rowwise_buffer, int tile_width
) {
    /* 
        row-wise aggregation (e.g. max, sum) assign to thread 0 
            - get_max_rowwise()
            - exp_sum()
    */
    size_t shared_mem_size = tile_width * tile_width * sizeof(float) * 2;
    matrix_transpose_kernel<<<dimGrid, dimBlock>>>(key, keyT, b, m, n);
    matrix_multiply_kernel<<<dimGrid, dimBlock, shared_mem_size>>>(query, keyT, score, b, m, n, m, tile_width);
    normalize_kernel<<<dimGrid, dimBlock>>>(score, b, m, m, sqrt((float) n));
    if (mask) matrix_mask_kernel<<<dimGrid, dimBlock>>>(score, (int*)mask, b, m, m);

    get_max_rowwise<<<dimGrid, dimBlock>>>(score, b, m, n, rowwise_buffer);
    exp_sum<<<dimGrid, dimBlock>>>(score, b, m, n, rowwise_buffer);
    normalize_kernel_rowwise<<<dimGrid, dimBlock>>>(score, b, m, n, rowwise_buffer);

    matrix_multiply_kernel<<<dimGrid, dimBlock, shared_mem_size>>>(score, value, output, b, m, m, n, tile_width);
}

__global__ void get_max_rowwise_reduced(float* matrix, int b, int m, int n, float* rowwise_block_buffer) {
    int batch = blockIdx.z * blockDim.z + threadIdx.z;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int j;
    
    float max_val = NEG_INF;

    if (col < blockDim.y) {
        for (j=col; j<n; j+=blockDim.y) { max_val = fmaxf(max_val, matrix[batch*m*n+row*n+j]); }
        rowwise_block_buffer[batch*m*n+row*n+col] = max_val;
    }
    
    __syncthreads();

    if (col == 0) {
        max_val = rowwise_block_buffer[batch*m*n+row*n+col];
        for (j=0; j<blockDim.y; j++) { max_val = fmaxf(max_val, matrix[batch*m*n+row*n+j]); }
        rowwise_block_buffer[batch*m*n+row*n] = max_val;
    }
}

__global__ void exp_sum_reduced(float* matrix, int b, int m, int n, float* rowwise_block_buffer) {
    int batch = blockIdx.z * blockDim.z + threadIdx.z;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int j;
    
    float sum;

    if (col < blockDim.y) {
        sum = 0.0f;
        for (j=col; j<n; j+=blockDim.y) {
            matrix[batch*m*n+row*n+j] = exp(matrix[batch*m*n+row*n+j] - rowwise_block_buffer[batch*m*n+row*n]); 
            sum += matrix[batch*m*n+row*n+j]; 
        }
        rowwise_block_buffer[batch*m*n+row*n+col] = sum;
    }

    __syncthreads();
    
    if (col == 0) {
        sum = 0.0f;
        for (j=0; j<blockDim.y; j++) { sum += rowwise_block_buffer[batch*m*n+row*n+col]; }
        rowwise_block_buffer[batch*m*n+row*n] = sum;
    }
}

void scaled_dot_product_attention_kernel_reduced(dim3 dimBlock, dim3 dimGrid,
    float* query, float* key, float* value, float* keyT, 
    float* mask, float* score, int b, int m, int n, float* output, float* rowwise_block_buffer, int tile_width
) {
    /*
        row-wise aggregation assign to block 0 
            - get_max_rowwise_reduced()
            - exp_sum_reduced()
    */
    size_t shared_mem_size = tile_width * tile_width * sizeof(float) * 2;
    matrix_transpose_kernel<<<dimGrid, dimBlock>>>(key, keyT, b, m, n);
    matrix_multiply_kernel<<<dimGrid, dimBlock, shared_mem_size>>>(query, keyT, score, b, m, n, m, tile_width);
    normalize_kernel<<<dimGrid, dimBlock>>>(score, b, m, m, sqrt((float) n));
    if (mask) matrix_mask_kernel<<<dimGrid, dimBlock>>>(score, (int*)mask, b, m, m);

    get_max_rowwise_reduced<<<dimGrid, dimBlock>>>(score, b, m, n, rowwise_block_buffer);
    exp_sum_reduced<<<dimGrid, dimBlock>>>(score, b, m, n, rowwise_block_buffer);
    normalize_kernel_rowwise_reduced<<<dimGrid, dimBlock>>>(score, b, m, n, rowwise_block_buffer);

    matrix_multiply_kernel<<<dimGrid, dimBlock, shared_mem_size>>>(score, value, output, b, m, m, n, tile_width);
}

void test() {

    int num_thread, grid_size_x, grid_size_y;

    if (arg_test) {
        num_thread = 1;
        tile_width = 1;
    }
    else {
        num_thread = 8;
        tile_width = 8;
    }
    if (arg_tile_width > 0) {
        num_thread = arg_tile_width;
        tile_width = arg_tile_width;
    }

    grid_size_x = (embed_dim + num_thread - 1) / num_thread;
    grid_size_y = (seq_length + num_thread - 1) / num_thread;

	struct timeval start, end;
    float *query, *key, *value, *mask, *score, *output;
    float *tranpose_buffer, *rowwise_buffer, *rowwise_block_buffer;

    int num_dim = 3;
    int dims[] = {num_batch, seq_length, embed_dim};
    int dims_mask[] = {num_batch, seq_length, seq_length};

    int dims_buff[] = {num_batch, seq_length, 1};
    int dims_buff_block[] = {num_batch, seq_length, num_thread};

    query = generate_matrix(dims, num_dim);
    key = generate_matrix(dims, num_dim);
    value = generate_matrix(dims, num_dim);
    tranpose_buffer = generate_matrix(dims, num_dim, 'z'); // random
    mask = arg_mask ? generate_matrix(dims_mask, num_dim, 'b') : nullptr; // boolean
    score = generate_matrix(dims_mask, num_dim, 'z'); // zero
    output = generate_matrix(dims, num_dim, 'z');
    rowwise_buffer = generate_matrix(dims_buff, num_dim, 'z');
    rowwise_block_buffer = generate_matrix(dims_buff_block, num_dim, 'z');

    dim3 dimBlock(num_thread, num_thread, 1);
    dim3 dimGrid(grid_size_x, grid_size_y, num_batch);

    float *query_device, *key_device, *value_device, *tranpose_buffer_device, *mask_device, *score_device, *output_device;
    float *rowwise_buffer_device, *rowwise_block_buffer_device;

    int size = num_batch*embed_dim*seq_length*sizeof(float);
    int size_mask = num_batch*seq_length*seq_length*sizeof(float);
    int size_buff = num_batch*seq_length*sizeof(float);

	cudaMalloc((void**)&query_device, size);
	cudaMalloc((void**)&key_device, size);
	cudaMalloc((void**)&value_device, size);
	cudaMalloc((void**)&tranpose_buffer_device, size);
	if (arg_mask) cudaMalloc((void**)&mask_device, size_mask);
	cudaMalloc((void**)&score_device, size_mask);
	cudaMalloc((void**)&output_device, size);

    if (mode == 's') { cudaMalloc((void**)&rowwise_buffer_device, size_buff); }
    else if (mode == 'r') { cudaMalloc((void**)&rowwise_block_buffer_device, size_buff); }
    else { printf("mode %c not defined\n", mode); return; }

    gettimeofday(&start, nullptr);

	cudaMemcpy(query_device, query, size, cudaMemcpyHostToDevice);
	cudaMemcpy(key_device, key, size, cudaMemcpyHostToDevice);
	cudaMemcpy(value_device, value, size, cudaMemcpyHostToDevice);
	cudaMemcpy(tranpose_buffer_device, tranpose_buffer, size, cudaMemcpyHostToDevice);
	if (arg_mask) cudaMemcpy(mask_device, mask, size_mask, cudaMemcpyHostToDevice); else mask_device = nullptr;
	cudaMemcpy(score_device, score, size_mask, cudaMemcpyHostToDevice);
	cudaMemcpy(output_device, output, size, cudaMemcpyHostToDevice);
	
    if (mode == 's') { 
        cudaMemcpy(rowwise_buffer_device, rowwise_buffer, size_buff, cudaMemcpyHostToDevice); 

        scaled_dot_product_attention_kernel_single(dimBlock, dimGrid,
            query_device, key_device, value_device, tranpose_buffer_device, 
            mask_device, score_device, 
            num_batch, seq_length, embed_dim, output_device, rowwise_buffer_device, tile_width
        );
    }
	else { 
        cudaMemcpy(rowwise_block_buffer_device, rowwise_block_buffer, size_buff, cudaMemcpyHostToDevice); 

        scaled_dot_product_attention_kernel_reduced(dimBlock, dimGrid,
            query_device, key_device, value_device, tranpose_buffer_device, 
            mask_device, score_device, 
            num_batch, seq_length, embed_dim, output_device, rowwise_block_buffer_device, tile_width
        );
    }
    
	cudaMemcpy(output, output_device, size, cudaMemcpyDeviceToHost);
    gettimeofday(&end, nullptr);

    if (arg_test) {
        cudaMemcpy(score, score_device, size_mask, cudaMemcpyDeviceToHost);
        cudaMemcpy(tranpose_buffer, tranpose_buffer_device, size, cudaMemcpyDeviceToHost);
    }

	float elapsedTime = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
	printf("%lf\n", elapsedTime); // printf("time: %lf\n", elapsedTime);

    cudaFree(query_device); 
    cudaFree(key_device); 
    cudaFree(value_device);
    cudaFree(tranpose_buffer_device);
    if (arg_mask) cudaFree(mask_device);
    cudaFree(score_device);
    cudaFree(output_device);
    
    if (arg_test) { 
        // print_result(query, key, value, mask, output, num_batch, seq_length, embed_dim); 
        printf("o=\n"); print_3d(output, num_batch, seq_length, embed_dim);
    }
}

void test_unified() {

    int num_thread, grid_size_x, grid_size_y;

    if (arg_test) {
        num_thread = 1;
        tile_width = 1;
    }
    else {
        num_thread = 8;
        tile_width = 8;
    }
    if (arg_tile_width > 0) {
        num_thread = arg_tile_width;
        tile_width = arg_tile_width;
    }

    grid_size_x = (embed_dim + num_thread - 1) / num_thread;
    grid_size_y = (seq_length + num_thread - 1) / num_thread;

	struct timeval start, end;

    int num_dim = 3;
    int dims[] = {num_batch, seq_length, embed_dim};
    int dims_mask[] = {num_batch, seq_length, seq_length};

    int dims_buff[] = {num_batch, seq_length, 1};
    int dims_buff_block[] = {num_batch, seq_length, num_thread};

    dim3 dimBlock(num_thread, num_thread, 1);
    dim3 dimGrid(grid_size_x, grid_size_y, num_batch);

    float *query_device, *key_device, *value_device, *tranpose_buffer_device, *mask_device, *score_device, *output_device;
    float *rowwise_buffer_device, *rowwise_block_buffer_device;

    int size = num_batch*embed_dim*seq_length*sizeof(float);
    int size_mask = num_batch*seq_length*seq_length*sizeof(float);
    int size_buff = num_batch*seq_length*sizeof(float);

    cudaMallocManaged((void**)&query_device, size);
	cudaMallocManaged((void**)&key_device, size);
	cudaMallocManaged((void**)&value_device, size);
	cudaMallocManaged((void**)&tranpose_buffer_device, size);
	if (arg_mask) cudaMallocManaged((void**)&mask_device, size_mask);
	cudaMallocManaged((void**)&score_device, size_mask);
	cudaMallocManaged((void**)&output_device, size);

    if (mode == 's') { cudaMallocManaged((void**)&rowwise_buffer_device, size_buff); }
    else if (mode == 'r') { cudaMallocManaged((void**)&rowwise_block_buffer_device, size_buff); }
    else { printf("mode %c not defined\n", mode); return; }

    query_device = generate_matrix(dims, num_dim);
    key_device = generate_matrix(dims, num_dim);
    value_device = generate_matrix(dims, num_dim);
    tranpose_buffer_device = generate_matrix(dims, num_dim, 'z'); // random
    mask_device = arg_mask ? generate_matrix(dims_mask, num_dim, 'b') : nullptr; // boolean
    score_device = generate_matrix(dims_mask, num_dim, 'z'); // zero
    output_device = generate_matrix(dims, num_dim, 'z');
    rowwise_buffer_device = generate_matrix(dims_buff, num_dim, 'z');
    rowwise_block_buffer_device = generate_matrix(dims_buff_block, num_dim, 'z');

    gettimeofday(&start, nullptr);

    if (mode == 's') { 
        scaled_dot_product_attention_kernel_single(dimBlock, dimGrid,
            query_device, key_device, value_device, tranpose_buffer_device, 
            mask_device, score_device, 
            num_batch, seq_length, embed_dim, output_device, rowwise_buffer_device, tile_width
        );
    }
	else { 
        scaled_dot_product_attention_kernel_reduced(dimBlock, dimGrid,
            query_device, key_device, value_device, tranpose_buffer_device, 
            mask_device, score_device, 
            num_batch, seq_length, embed_dim, output_device, rowwise_block_buffer_device, tile_width
        );
    }
    
    gettimeofday(&end, nullptr);

	float elapsedTime = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
	printf("%lf\n", elapsedTime); // printf("time: %lf\n", elapsedTime);

    cudaFree(query_device); 
    cudaFree(key_device); 
    cudaFree(value_device);
    cudaFree(tranpose_buffer_device);
    if (arg_mask) cudaFree(mask_device);
    cudaFree(score_device);
    cudaFree(output_device);
    
    if (arg_test) { 
        // print_result(query, key, value, mask, output, num_batch, seq_length, embed_dim); 
        printf("o=\n"); print_3d(output_device, num_batch, seq_length, embed_dim);
    }
}

void test_separate() {

    int num_thread, grid_size_x, grid_size_y;

    if (arg_test) {
        num_thread = 1;
        tile_width = 1;
    }
    else {
        num_thread = 8;
        tile_width = 8;
    }
    if (arg_tile_width > 0) {
        num_thread = arg_tile_width;
        tile_width = arg_tile_width;
    }

    grid_size_x = (embed_dim + num_thread - 1) / num_thread;
    grid_size_y = (seq_length + num_thread - 1) / num_thread;

	struct timeval start, end;
	struct timeval malloc_start, malloc_end, memcpyhd_start, memcpyhd_end, kernel_start, kernel_end, memcpydh_start, memcpydh_end;

    float *query, *key, *value, *mask, *score, *output;
    float *tranpose_buffer, *rowwise_buffer, *rowwise_block_buffer;

    int num_dim = 3;
    int dims[] = {num_batch, seq_length, embed_dim};
    int dims_mask[] = {num_batch, seq_length, seq_length};

    int dims_buff[] = {num_batch, seq_length, 1};
    int dims_buff_block[] = {num_batch, seq_length, num_thread};

    query = generate_matrix(dims, num_dim);
    key = generate_matrix(dims, num_dim);
    value = generate_matrix(dims, num_dim);
    tranpose_buffer = generate_matrix(dims, num_dim, 'z'); // random
    mask = arg_mask ? generate_matrix(dims_mask, num_dim, 'b') : nullptr; // boolean
    score = generate_matrix(dims_mask, num_dim, 'z'); // zero
    output = generate_matrix(dims, num_dim, 'z');
    rowwise_buffer = generate_matrix(dims_buff, num_dim, 'z');
    rowwise_block_buffer = generate_matrix(dims_buff_block, num_dim, 'z');

    dim3 dimBlock(num_thread, num_thread, 1);
    dim3 dimGrid(grid_size_x, grid_size_y, num_batch);

    float *query_device, *key_device, *value_device, *tranpose_buffer_device, *mask_device, *score_device, *output_device;
    float *rowwise_buffer_device, *rowwise_block_buffer_device;

    int size = num_batch*embed_dim*seq_length*sizeof(float);
    int size_mask = num_batch*seq_length*seq_length*sizeof(float);
    int size_buff = num_batch*seq_length*sizeof(float);

    gettimeofday(&malloc_start, nullptr);

	cudaMalloc((void**)&query_device, size);
	cudaMalloc((void**)&key_device, size);
	cudaMalloc((void**)&value_device, size);
	cudaMalloc((void**)&tranpose_buffer_device, size);
	if (arg_mask) cudaMalloc((void**)&mask_device, size_mask);
	cudaMalloc((void**)&score_device, size_mask);
	cudaMalloc((void**)&output_device, size);

    if (mode == 's') { cudaMalloc((void**)&rowwise_buffer_device, size_buff); }
    else if (mode == 'r') { cudaMalloc((void**)&rowwise_block_buffer_device, size_buff); }
    else { printf("mode %c not defined\n", mode); return; }

    gettimeofday(&malloc_end, nullptr);
    gettimeofday(&start, nullptr);
    gettimeofday(&memcpyhd_start, nullptr);

	cudaMemcpy(query_device, query, size, cudaMemcpyHostToDevice);
	cudaMemcpy(key_device, key, size, cudaMemcpyHostToDevice);
	cudaMemcpy(value_device, value, size, cudaMemcpyHostToDevice);
	cudaMemcpy(tranpose_buffer_device, tranpose_buffer, size, cudaMemcpyHostToDevice);
	if (arg_mask) cudaMemcpy(mask_device, mask, size_mask, cudaMemcpyHostToDevice); else mask_device = nullptr;
	cudaMemcpy(score_device, score, size_mask, cudaMemcpyHostToDevice);
	cudaMemcpy(output_device, output, size, cudaMemcpyHostToDevice);
    
    if (mode == 's')
        cudaMemcpy(rowwise_buffer_device, rowwise_buffer, size_buff, cudaMemcpyHostToDevice); 
	else
        cudaMemcpy(rowwise_block_buffer_device, rowwise_block_buffer, size_buff, cudaMemcpyHostToDevice); 

    gettimeofday(&memcpyhd_end, nullptr);
    gettimeofday(&kernel_start, nullptr);
	
    if (mode == 's') { 
        scaled_dot_product_attention_kernel_single(dimBlock, dimGrid,
            query_device, key_device, value_device, tranpose_buffer_device, 
            mask_device, score_device, 
            num_batch, seq_length, embed_dim, output_device, rowwise_buffer_device, tile_width
        );
    }
	else { 
        scaled_dot_product_attention_kernel_reduced(dimBlock, dimGrid,
            query_device, key_device, value_device, tranpose_buffer_device, 
            mask_device, score_device, 
            num_batch, seq_length, embed_dim, output_device, rowwise_block_buffer_device, tile_width
        );
    }

    gettimeofday(&kernel_end, nullptr);
    gettimeofday(&memcpydh_start, nullptr);
    
	cudaMemcpy(output, output_device, size, cudaMemcpyDeviceToHost);

    gettimeofday(&memcpydh_end, nullptr);
    gettimeofday(&end, nullptr);

    if (arg_test) {
        cudaMemcpy(score, score_device, size_mask, cudaMemcpyDeviceToHost);
        cudaMemcpy(tranpose_buffer, tranpose_buffer_device, size, cudaMemcpyDeviceToHost);
    }

    float malloc_time = (malloc_end.tv_sec - malloc_start.tv_sec) + (malloc_end.tv_usec - malloc_start.tv_usec) / 1e6;
	float memcpyhd_time = (memcpyhd_end.tv_sec - memcpyhd_start.tv_sec) + (memcpyhd_end.tv_usec - memcpyhd_start.tv_usec) / 1e6;
	float kernel_time = (kernel_end.tv_sec - kernel_start.tv_sec) + (kernel_end.tv_usec - kernel_start.tv_usec) / 1e6;
	float memcpydh_time = (memcpydh_end.tv_sec - memcpydh_start.tv_sec) + (memcpydh_end.tv_usec - memcpydh_start.tv_usec) / 1e6;
	float elapsedTime = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
	// printf("%lf\n", elapsedTime); // printf("time: %lf\n", elapsedTime);
	printf("%lf\t%lf\t%lf\t%lf\n", malloc_time, memcpyhd_time, kernel_time, memcpydh_time);

    cudaFree(query_device); 
    cudaFree(key_device); 
    cudaFree(value_device);
    cudaFree(tranpose_buffer_device);
    if (arg_mask) cudaFree(mask_device);
    cudaFree(score_device);
    cudaFree(output_device);
    
    if (arg_test) { 
        // print_result(query, key, value, mask, output, num_batch, seq_length, embed_dim); 
        printf("o=\n"); print_3d(output, num_batch, seq_length, embed_dim);
    }
}

int main(int argc, char* argv[]) {
    proc_arg(argc, argv);
    if (arg_unified) 
        test_unified();
    else {
        if (arg_test_separate)
            test_separate();
        else
            test();
    }
    return 0;
}

// nvcc ./src/attention_cuda_tile.cu matrix_io.cpp -lm -o ./src/attention_cuda_tile
// ./src/attention_cuda_tile -b 2 -s 2 -e 2 -t
// ./src/attention_cuda_tile -b 128 -s 512 -e 512
