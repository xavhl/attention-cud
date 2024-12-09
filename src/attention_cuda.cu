#include <unistd.h>
#include "../matrix_io.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <sys/time.h>
#include <cuda.h>
using namespace std;

#define NEG_INF -1e9

bool arg_reduced = false;
bool arg_test = false;
bool arg_mask = false;
int num_batch = 1;
int embed_dim = 1;
int seq_length = 1;
char mode = 's'; // simple / transposed

void proc_arg(int argc, char **argv) {
    int c;
    while ((c = getopt(argc, argv, "rtkm:b:e:s:")) != -1) {
        switch (c) {
            case 'm':
                sscanf(optarg, " %c", &mode);
                break;
            case 'k':
                arg_mask = true;
                break;
            case 'r':
                arg_reduced = true;
                break;
            case 't':
                arg_test = true;
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
            default: // bad or unknown option
                exit(1);
                break;
        }
    }
}

__global__ void softmax_kernel(float* matrix, int b, int m, int n) {
    int batch = blockIdx.z * blockDim.z + threadIdx.z;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    float max_val = NEG_INF, sum = 0.0f;

    if (j==0) {
        for (j = 0; j < n; j++) { max_val = fmaxf(max_val, matrix[batch*m*n+row*n+j]); }
        for (j = 0; j < n; j++) { matrix[batch*m*n+row*n+j] = exp(matrix[batch*m*n+row*n+j] - max_val); sum += matrix[batch*m*n+row*n+j]; }
        for (j = 0; j < n; j++) { matrix[batch*m*n+row*n+j] /= sum; }
    }
}

__global__ void matrix_transpose_kernel(float* a, float* aT, int b, int m, int n) {
    int batch = blockIdx.z * blockDim.z + threadIdx.z;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < m && col < n) aT[batch*m*n+col*m+row] = a[batch*m*n+row*n+col];
}

__global__ void matrix_multiply_kernel(float* a, float* b, float* c, int nb, int m, int r, int n) {
    int batch = blockIdx.z * blockDim.z + threadIdx.z;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    float sum = 0.0f;
    for (int k = 0; k < r; k++) { sum += a[batch*m*r+row*r+k] * b[batch*r*n+k*n+col]; }
    c[batch*m*n+row*n+col] = sum;
}

__global__ void matrix_multiply_transpose_kernel(float* a, float* b, float* c, int nb, int m, int r, int n) {
    int batch = blockIdx.z * blockDim.z + threadIdx.z;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    float sum = 0.0f;
    for (int k = 0; k < r; k++) { sum += a[batch*m*r+k*r+row] * b[batch*r*n+k*r+col]; }
    c[batch*m*n+row*n+col] = sum;
}

__global__ void normalize_kernel(float* matrix, int b, int m, int n, float norm_const) {
    matrix[(blockIdx.z * blockDim.z + threadIdx.z)*m*n+(blockIdx.x*blockDim.x+threadIdx.x)*n+(blockIdx.y*blockDim.y+threadIdx.y)] /= norm_const;
}

__global__ void matrix_mask_kernel(float* matrix, int* mask, int b, int m, int n) {
    int batch = blockIdx.z * blockDim.z + threadIdx.z;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (mask[batch*m*n+row*n+col] == 0.0f) { matrix[batch*m*n+row*n+col] = NEG_INF; }
}

void scaled_dot_product_attention_kernel_simple(dim3 dimBlock, dim3 dimGrid,
    float* query, float* key, float* value, float* mask, float* score, 
    float* keyT, 
    int b, int m, int n, float* output
) {
    matrix_transpose_kernel<<<dimGrid, dimBlock>>>(key, keyT, b, m, n);
    matrix_multiply_kernel<<<dimGrid, dimBlock>>>(query, keyT, score, b, m, n, m);

    normalize_kernel<<<dimGrid, dimBlock>>>(score, b, m, m, sqrt((float) n));
    if (mask) matrix_mask_kernel<<<dimGrid, dimBlock>>>(score, (int*)mask, b, m, m);

    softmax_kernel<<<dimGrid, dimBlock>>>(score, b, m, m);

    matrix_multiply_kernel<<<dimGrid, dimBlock>>>(score, value, output, b, m, m, n);
}

void scaled_dot_product_attention_kernel_transposed(dim3 dimBlock, dim3 dimGrid,
    float* query, float* key, float* value, float* mask, float* score,
    float* queryT, float* keyT, float *scoreT,
    int b, int m, int n, float* output
) {

    matrix_transpose_kernel<<<dimGrid, dimBlock>>>(query, queryT, b, m, n);
    matrix_transpose_kernel<<<dimGrid, dimBlock>>>(key, keyT, b, m, n);
    matrix_multiply_transpose_kernel<<<dimGrid, dimBlock>>>(queryT, keyT, score, b, m, n, m);

    normalize_kernel<<<dimGrid, dimBlock>>>(score, b, m, m, sqrt((float) n));

    if (mask) matrix_mask_kernel<<<dimGrid, dimBlock>>>(score, (int*)mask, b, m, m);

    softmax_kernel<<<dimGrid, dimBlock>>>(score, b, m, m);

    matrix_transpose_kernel<<<dimGrid, dimBlock>>>(score, scoreT, b, m, n); // borrow qkv_T
    matrix_multiply_transpose_kernel<<<dimGrid, dimBlock>>>(scoreT, value, output, b, m, m, n);
}

void test() {
	struct timeval start, end;
    float *query, *key, *value, *mask, *score, *output;

    int num_dim = 3;
    int dims[] = {num_batch, seq_length, embed_dim};
    int dims_mask[] = {num_batch, seq_length, seq_length};

    query = generate_matrix(dims, num_dim);
    key = generate_matrix(dims, num_dim);
    value = generate_matrix(dims, num_dim); // random
    mask = arg_mask ? generate_matrix(dims_mask, num_dim, 'b') : nullptr; // boolean
    score = generate_matrix(dims_mask, num_dim, 'z'); // zero
    output = generate_matrix(dims, num_dim, 'z');

    float *qkv_transpose, *qkv_transpose_, *score_transpose;

    qkv_transpose = generate_matrix(dims, num_dim, 'z');
    qkv_transpose_ = generate_matrix(dims, num_dim, 'z');
    score_transpose = generate_matrix(dims_mask, num_dim, 'z');

    int num_thread, grid_size_x, grid_size_y;

    num_thread = arg_test ? 1 : 16;
    grid_size_x = (embed_dim + num_thread - 1) / num_thread;
    grid_size_y = (seq_length + num_thread - 1) / num_thread;

    dim3 dimBlock(num_thread, num_thread, 1);
    dim3 dimGrid(grid_size_x, grid_size_y, num_batch);

    float *query_device, *key_device, *value_device, *mask_device, *score_device, *output_device;
    int size = num_batch*embed_dim*seq_length*sizeof(float);
    int size_mask = num_batch*seq_length*seq_length*sizeof(float);

    float *qkv_transpose_device, *qkv_transpose__device, *score_transpose_device;

	cudaMalloc((void**)&query_device, size);
	cudaMalloc((void**)&key_device, size);
	cudaMalloc((void**)&value_device, size);
	if (arg_mask) cudaMalloc((void**)&mask_device, size_mask);
	cudaMalloc((void**)&score_device, size_mask);
	cudaMalloc((void**)&output_device, size);

    cudaMalloc((void**)&qkv_transpose_device, size);

    if (mode == 't') {
        cudaMalloc((void**)&qkv_transpose__device, size);
        cudaMalloc((void**)&score_transpose_device, size);
    }

    gettimeofday(&start, nullptr);

	cudaMemcpy(query_device, query, size, cudaMemcpyHostToDevice);
	cudaMemcpy(key_device, key, size, cudaMemcpyHostToDevice);
	cudaMemcpy(value_device, value, size, cudaMemcpyHostToDevice);
	if (arg_mask) cudaMemcpy(mask_device, mask, size_mask, cudaMemcpyHostToDevice); else mask_device = nullptr;
	cudaMemcpy(score_device, score, size_mask, cudaMemcpyHostToDevice);
	cudaMemcpy(output_device, output, size, cudaMemcpyHostToDevice);
	
    cudaMemcpy(qkv_transpose_device, qkv_transpose, size, cudaMemcpyHostToDevice);

    if (mode == 't') {
        cudaMemcpy(qkv_transpose__device, qkv_transpose_, size, cudaMemcpyHostToDevice);
        cudaMemcpy(score_transpose_device, score_transpose, size_mask, cudaMemcpyHostToDevice);
    }

    
    if (mode == 's') { // simple
        scaled_dot_product_attention_kernel_simple(dimGrid, dimBlock,
            query_device, key_device, value_device, mask_device, score_device, 
            qkv_transpose_device,
            num_batch, seq_length, embed_dim, output_device
        );
    }
    else if (mode == 't') { // transposed
        scaled_dot_product_attention_kernel_transposed(dimGrid, dimBlock,
            query_device, key_device, value_device, mask_device, score_device, 
            qkv_transpose_device, qkv_transpose__device, score_transpose_device,
            num_batch, seq_length, embed_dim, output_device
        );
    }
    else {
        printf("mode %c not defined", mode);
    }

	cudaMemcpy(output, output_device, size, cudaMemcpyDeviceToHost);

    gettimeofday(&end, nullptr);

    if (arg_test) { 
	    cudaMemcpy(score, score_device, size_mask, cudaMemcpyDeviceToHost);
	    cudaMemcpy(qkv_transpose, qkv_transpose_device, size, cudaMemcpyDeviceToHost);
    }

	float elapsedTime = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
	printf("%lf", elapsedTime); // printf("time: %lf\n", elapsedTime);

    cudaFree(query_device); 
    cudaFree(key_device); 
    cudaFree(value_device);
    if (arg_mask) cudaFree(mask_device);
    cudaFree(score_device);
    cudaFree(output_device);
    
    cudaFree(qkv_transpose_device);

    if (mode == 't') {
        cudaFree(qkv_transpose__device);
        cudaFree(score_transpose_device);
    }
    
    if (arg_test) { 
        // print_result(query, key, value, mask, output, num_batch, seq_length, embed_dim); 
        printf("o=\n"); print_3d(output, num_batch, seq_length, embed_dim);
    }
}

int main(int argc, char* argv[]) {
    proc_arg(argc, argv);
    test();
    return 0;
}

// nvcc ./src/attention_cuda.cu matrix_io.cpp -lm -o ./src/attention_cuda
// ./src/attention_cuda -b 2 -s 2 -e 2 -t
