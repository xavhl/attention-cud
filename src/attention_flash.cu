
#include <unistd.h>
#include "../matrix_io.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <sys/time.h>
#include <cuda.h>
using namespace std;
#define NEG_INF -1e9

bool arg_test = false;
bool arg_unified = false; // use unified memory
int num_batch = 1;
int embed_dim = 1;
int seq_length = 1;
char mode = 's'; // single 's' / multi 'm'

void proc_arg(int argc, char **argv) {
    int c;
    while ((c = getopt(argc, argv, "rtum:b:e:s:")) != -1) {
        switch (c) {
            case 'm':
                sscanf(optarg, " %c", &mode);
                break;
            case 'u': 
                arg_unified = true;
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
            default:
                exit(1);
                break;
        }
    }
}

// credit: https://github.com/tspeterkim/flash-attention-minimal/blob/main/flash.cu
__global__ void forward_kernel(const float* Q, const float* K, const float* V, const int N, const int d,
                    const int Tc, const int Tr, const int Bc, const int Br, const float softmax_scale,
                    float* l, float *m, float* O) {
    int tx = threadIdx.x;
    int bx = blockIdx.x; int by = blockIdx.y;  // batch and head index

    // Offset into Q,K,V,O,l,m - different for each batch and head
    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);  // gridDim.y = nh
    int lm_offset = (bx * gridDim.y * N) + (by * N);  // offset for l and m

    // Define SRAM for Q,K,V,S
    extern __shared__ float sram[];
    int tile_size = Bc * d;  // size of Qi, Kj, Vj
    float* Qi = sram;
    float* Kj = &sram[tile_size];
    float* Vj = &sram[tile_size * 2];
    float* S = &sram[tile_size * 3];
    for (int j = 0; j < Tc; j++) {

        // Load Kj, Vj to SRAM
        for (int x = 0; x < d; x++) {
            Kj[(tx * d) + x] = K[qkv_offset + (tile_size * j) + (tx * d) + x];
            Vj[(tx * d) + x] = V[qkv_offset + (tile_size * j) + (tx * d) + x];
        }
        __syncthreads();  // such that the inner loop can use the correct Kj, Vj

        for (int i = 0; i < Tr; i++)  {

            // Load Qi to SRAM, l and m to registers
            for (int x = 0; x < d; x++) {
                Qi[(tx * d) + x] = Q[qkv_offset + (tile_size * i) + (tx * d) + x];
            }
            float row_m_prev = m[lm_offset + (Br * i) + tx];
            float row_l_prev = l[lm_offset + (Br * i) + tx];

            // S = QK^T, row_m = rowmax(S)
            float row_m = -INFINITY;
            for (int y = 0; y < Bc; y++) {
                float sum = 0;
                for (int x = 0; x < d; x++) {
                    sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
                }
                sum *= softmax_scale;
                S[(Bc * tx) + y] = sum;

                if (sum > row_m)
                    row_m = sum;
            }

            // P = exp(S - row_m), row_l = rowsum(P)
            float row_l = 0;
            for (int y = 0; y < Bc; y++) {
                S[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - row_m);
                row_l += S[(Bc * tx) + y];
            }

            // Compute new m and l
            float row_m_new = max(row_m_prev, row_m);
            float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + (__expf(row_m - row_m_new) * row_l);

            // Write O, l, m to HBM
            for (int x = 0; x < d; x++) {
                float pv = 0;  // Pij * Vj
                for (int y = 0; y < Bc; y++) {
                    pv += S[(Bc * tx) + y] * Vj[(y * d) + x];
                }
                O[qkv_offset + (tile_size * i) + (tx * d) + x] = (1 / row_l_new) \
                    * ((row_l_prev * __expf(row_m_prev - row_m_new) * O[qkv_offset + (tile_size * i) + (tx * d) + x]) \
                    + (__expf(row_m - row_m_new) * pv));
            }
            m[lm_offset + (Br * i) + tx] = row_m_new;
            l[lm_offset + (Br * i) + tx] = row_l_new;
        }
        __syncthreads();  // otherwise, thread can use the wrong Kj, Vj in inner loop
    }
    __syncthreads();
}

void test() {

    struct timeval start, end;
    float *query, *key, *value, *output, *row_max, *row_sum;

    int num_dim = 3;
    int dims[] = {num_batch, seq_length, embed_dim};
    int num_dim_row = 2;
    int dims_row[] = {num_batch, seq_length};

    query = generate_matrix(dims, num_dim);
    key = generate_matrix(dims, num_dim);
    value = generate_matrix(dims, num_dim);
    output = generate_matrix(dims, num_dim, 'z');
    row_max = generate_matrix(dims_row, num_dim_row, 'n');
    row_sum = generate_matrix(dims_row, num_dim_row, 'z');

    float *query_device, *key_device, *value_device, *output_device, *row_max_device, *row_sum_device;
    int size = num_batch*seq_length*embed_dim*sizeof(float);
    int size_row = num_batch*seq_length*sizeof(float);

    const int B = num_batch;
    const int N = seq_length;
    const int nh = 1;
    const int d = embed_dim;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    const int Bc = min(ceil(prop.sharedMemPerBlock/sizeof(float)/(4*d)), (float)N);
    const int Br = min(Bc,d);

    const int Tc = ceil((float) N / Bc); const int Tr = ceil((float) N / Br);
    const float softmax_scale = 1.0 / sqrt(d);

    // Calculate SRAM size needed per block
    const int sram_size = (3 * Bc * d * sizeof(float)) + (Bc * Br * sizeof(float));
    int max_sram_size = prop.sharedMemPerBlock;
    // cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    if (arg_test) printf("Max shared memory: %d, requested shared memory: %d \n", max_sram_size, sram_size);

    cudaMalloc((void**)&query_device, size);
	cudaMalloc((void**)&key_device, size);
	cudaMalloc((void**)&value_device, size);
	cudaMalloc((void**)&output_device, size);
	cudaMalloc((void**)&row_max_device, size_row);
	cudaMalloc((void**)&row_sum_device, size_row);

    gettimeofday(&start, nullptr);

    cudaMemcpy(query_device, query, size, cudaMemcpyHostToDevice);
	cudaMemcpy(key_device, key, size, cudaMemcpyHostToDevice);
	cudaMemcpy(value_device, value, size, cudaMemcpyHostToDevice);
	cudaMemcpy(output_device, output, size, cudaMemcpyHostToDevice);
	cudaMemcpy(row_max_device, row_max, size_row, cudaMemcpyHostToDevice);
	cudaMemcpy(row_sum_device, row_sum, size_row, cudaMemcpyHostToDevice);

    dim3 grid_dim(B, nh);  // batch_size x num_heads
    dim3 block_dim(Bc);  // Bc threads per block

    forward_kernel<<<grid_dim, block_dim, sram_size>>>(
        query_device, key_device, value_device,
        N, d, Tc, Tr, Bc, Br, softmax_scale,
        row_sum_device, row_max_device, output_device
    );

	cudaMemcpy(output, output_device, size, cudaMemcpyDeviceToHost);
    gettimeofday(&end, nullptr);

    float elapsedTime = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
	printf("%lf\n", elapsedTime);

    cudaFree(query_device); 
    cudaFree(key_device); 
    cudaFree(value_device);
    cudaFree(output_device);

    if (arg_test) { 
        // print_result(query, key, value, mask, output, num_batch, seq_length, embed_dim); 
        printf("o=\n"); print_3d(output, num_batch, seq_length, embed_dim);
    }

    delete[] query;
    delete[] key;
    delete[] value;
    delete[] output;
    delete[] row_max;
    delete[] row_sum;
}

void test_multi() {

    struct timeval start, end;
    float *query, *key, *value, *output, *row_max, *row_sum;

    int num_dim = 3;
    int dims[] = {num_batch, seq_length, embed_dim};
    int num_dim_row = 2;
    int dims_row[] = {num_batch, seq_length};
    int num_elements = num_batch * seq_length * embed_dim;
    int num_elements_row = num_batch * seq_length;

    query = generate_matrix(dims, num_dim);
    key = generate_matrix(dims, num_dim);
    value = generate_matrix(dims, num_dim);
    output = generate_matrix(dims, num_dim, 'z');
    row_max = generate_matrix(dims_row, num_dim_row, 'n');
    row_sum = generate_matrix(dims_row, num_dim_row, 'z');

    int size = num_batch*seq_length*embed_dim*sizeof(float);
    int size_row = num_batch*seq_length*sizeof(float);

    const int B = num_batch;
    const int N = seq_length;
    const int nh = 1;
    const int d = embed_dim;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    const int Bc = min(ceil(prop.sharedMemPerBlock/sizeof(float)/(4*d)), (float)N);
    const int Br = min(Bc,d);
    
    const int Tc = ceil((float) N / Bc); const int Tr = ceil((float) N / Br);
    const float softmax_scale = 1.0 / sqrt(d);

    // Calculate SRAM size needed per block
    const int sram_size = (3 * Bc * d * sizeof(float)) + (Bc * Br * sizeof(float));
    if (arg_test) {
        printf(
            "device[0] max shared memory: %d, requested shared memory: %d \n", 
            prop.sharedMemPerBlock, 
            sram_size
        );
    }

    // device 1
    cudaGetDeviceProperties(&prop, 1);
    if (arg_test) {
        printf(
            "device[1] max shared memory: %d, requested shared memory: %d \n", 
            prop.sharedMemPerBlock, 
            sram_size
        );
    }

    int num_devices = 2;
    int size_per_device = size / num_devices;
    int size_row_per_device = size_row / num_devices;
    int num_element_per_device = num_elements / num_devices;
    int num_element_row_per_device = num_elements_row / num_devices;

    cudaStream_t stream0, stream1;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);

    cudaSetDevice(0);
    float *q_d0, *k_d0, *v_d0, *o_d0, *rm_d0, *rs_d0;
	cudaMalloc((void**)&q_d0, size_per_device);
	cudaMalloc((void**)&k_d0, size_per_device);
	cudaMalloc((void**)&v_d0, size_per_device);
	cudaMalloc((void**)&o_d0, size_per_device);
	cudaMalloc((void**)&rm_d0, size_row_per_device);
	cudaMalloc((void**)&rs_d0, size_row_per_device);

    cudaSetDevice(1);
    float *q_d1, *k_d1, *v_d1, *o_d1, *rm_d1, *rs_d1;
	cudaMalloc((void**)&q_d1, size_per_device);
	cudaMalloc((void**)&k_d1, size_per_device);
	cudaMalloc((void**)&v_d1, size_per_device);
	cudaMalloc((void**)&o_d1, size_per_device);
	cudaMalloc((void**)&rm_d1, size_row_per_device);
	cudaMalloc((void**)&rs_d1, size_row_per_device);

    gettimeofday(&start, nullptr); // timing starts from cudaMemcpy() 

    // device 1
    cudaMemcpy(q_d1, &query[num_element_per_device], size_per_device, cudaMemcpyHostToDevice);
	cudaMemcpy(k_d1, &key[num_element_per_device], size_per_device, cudaMemcpyHostToDevice);
	cudaMemcpy(v_d1, &value[num_element_per_device], size_per_device, cudaMemcpyHostToDevice);
	cudaMemcpy(o_d1, &output[num_element_per_device], size_per_device, cudaMemcpyHostToDevice);
	cudaMemcpy(rm_d1, &row_max[num_element_row_per_device], size_row_per_device, cudaMemcpyHostToDevice);
	cudaMemcpy(rs_d1, &row_sum[num_element_row_per_device], size_row_per_device, cudaMemcpyHostToDevice);

    cudaSetDevice(0);
    cudaMemcpy(q_d0, query, size_per_device, cudaMemcpyHostToDevice);
	cudaMemcpy(k_d0, key, size_per_device, cudaMemcpyHostToDevice);
	cudaMemcpy(v_d0, value, size_per_device, cudaMemcpyHostToDevice);
	cudaMemcpy(o_d0, output, size_per_device, cudaMemcpyHostToDevice);
	cudaMemcpy(rm_d0, row_max, size_row_per_device, cudaMemcpyHostToDevice);
	cudaMemcpy(rs_d0, row_sum, size_row_per_device, cudaMemcpyHostToDevice);

    dim3 grid_dim(B, nh);  // batch_size x num_heads
    dim3 block_dim(Bc);  // Bc threads per block

    // device 0
    forward_kernel<<<grid_dim, block_dim, sram_size>>>(
        q_d0, k_d0, v_d0,
        N, d, Tc, Tr, Bc, Br, softmax_scale,
        rs_d0, rm_d0, o_d0
    );

    cudaSetDevice(1);
    forward_kernel<<<grid_dim, block_dim, sram_size>>>(
        q_d1, k_d1, v_d1,
        N, d, Tc, Tr, Bc, Br, softmax_scale,
        rs_d1, rm_d1, o_d1
    );

	cudaMemcpy(&output[num_element_per_device], o_d1, size_per_device, cudaMemcpyDeviceToHost);
    cudaSetDevice(0); cudaMemcpy(output, o_d0, size_per_device, cudaMemcpyDeviceToHost);

    gettimeofday(&end, nullptr);

    float elapsedTime = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
	printf("%lf\n", elapsedTime); // printf("time: %lf\n", elapsedTime);

    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);

    cudaFree(q_d0); cudaFree(k_d0); cudaFree(v_d0); cudaFree(o_d0);
    cudaFree(q_d1); cudaFree(k_d1); cudaFree(v_d1); cudaFree(o_d1);

    if (arg_test) { 
        // print_result(query, key, value, mask, output, num_batch, seq_length, embed_dim); 
        printf("o=\n"); print_3d(output, num_batch, seq_length, embed_dim);
    }

    delete[] query;
    delete[] key;
    delete[] value;
    delete[] output;
    delete[] row_max;
    delete[] row_sum;
}

void test_multi_unified() {

    struct timeval start, end;
    float *query, *key, *value, *output, *row_max, *row_sum;

    int num_dim = 3;
    int dims[] = {num_batch, seq_length, embed_dim};
    int num_dim_row = 2;
    int dims_row[] = {num_batch, seq_length};
    int num_elements = num_batch * seq_length * embed_dim;
    int num_elements_row = num_batch * seq_length;

    query = generate_matrix(dims, num_dim);
    key = generate_matrix(dims, num_dim);
    value = generate_matrix(dims, num_dim);
    output = generate_matrix(dims, num_dim, 'z');
    row_max = generate_matrix(dims_row, num_dim_row, 'n');
    row_sum = generate_matrix(dims_row, num_dim_row, 'z');

    int size = num_batch*seq_length*embed_dim*sizeof(float);
    int size_row = num_batch*seq_length*sizeof(float);

    const int B = num_batch;
    const int N = seq_length;
    const int nh = 1;
    const int d = embed_dim;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    const int Bc = min(ceil(prop.sharedMemPerBlock/sizeof(float)/(4*d)), (float)N);
    const int Br = min(Bc,d);
    
    const int Tc = ceil((float) N / Bc); const int Tr = ceil((float) N / Br);
    const float softmax_scale = 1.0 / sqrt(d);

    // Calculate SRAM size needed per block
    const int sram_size = (3 * Bc * d * sizeof(float)) + (Bc * Br * sizeof(float));
    if (arg_test) {
        printf(
            "device[0] max shared memory: %d, requested shared memory: %d \n", 
            prop.sharedMemPerBlock, 
            sram_size
        );
    }

    // device 1
    cudaGetDeviceProperties(&prop, 1);
    if (arg_test) {
        printf(
            "device[1] max shared memory: %d, requested shared memory: %d \n", 
            prop.sharedMemPerBlock, 
            sram_size
        );
    }

    int num_devices = 2;
    int size_per_device = size / num_devices;
    int size_row_per_device = size_row / num_devices;
    int num_element_per_device = num_elements / num_devices;
    int num_element_row_per_device = num_elements_row / num_devices;

    cudaStream_t stream0, stream1;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);

    cudaSetDevice(0);
    float *q_d0, *k_d0, *v_d0, *o_d0, *rm_d0, *rs_d0;

	cudaMallocManaged((void**)&q_d0, size_per_device);
	cudaMallocManaged((void**)&k_d0, size_per_device);
	cudaMallocManaged((void**)&v_d0, size_per_device);
	cudaMallocManaged((void**)&o_d0, size_per_device);
	cudaMallocManaged((void**)&rm_d0, size_row_per_device);
	cudaMallocManaged((void**)&rs_d0, size_row_per_device);

    cudaMemcpy(q_d0, query, size_per_device, cudaMemcpyHostToDevice);
	cudaMemcpy(k_d0, key, size_per_device, cudaMemcpyHostToDevice);
	cudaMemcpy(v_d0, value, size_per_device, cudaMemcpyHostToDevice);
	cudaMemcpy(o_d0, output, size_per_device, cudaMemcpyHostToDevice);
	cudaMemcpy(rm_d0, row_max, size_row_per_device, cudaMemcpyHostToDevice);
	cudaMemcpy(rs_d0, row_sum, size_row_per_device, cudaMemcpyHostToDevice);

    cudaSetDevice(1);
    float *q_d1, *k_d1, *v_d1, *o_d1, *rm_d1, *rs_d1;

	cudaMallocManaged((void**)&q_d1, size_per_device);
	cudaMallocManaged((void**)&k_d1, size_per_device);
	cudaMallocManaged((void**)&v_d1, size_per_device);
	cudaMallocManaged((void**)&o_d1, size_per_device);
	cudaMallocManaged((void**)&rm_d1, size_row_per_device);
	cudaMallocManaged((void**)&rs_d1, size_row_per_device);

    cudaMemcpy(q_d1, &query[num_element_per_device], size_per_device, cudaMemcpyHostToDevice);
	cudaMemcpy(k_d1, &key[num_element_per_device], size_per_device, cudaMemcpyHostToDevice);
	cudaMemcpy(v_d1, &value[num_element_per_device], size_per_device, cudaMemcpyHostToDevice);
	cudaMemcpy(o_d1, &output[num_element_per_device], size_per_device, cudaMemcpyHostToDevice);
	cudaMemcpy(rm_d1, &row_max[num_element_row_per_device], size_row_per_device, cudaMemcpyHostToDevice);
	cudaMemcpy(rs_d1, &row_sum[num_element_row_per_device], size_row_per_device, cudaMemcpyHostToDevice);
    
    dim3 grid_dim(B, nh);  // batch_size x num_heads
    dim3 block_dim(Bc);  // Bc threads per block

    gettimeofday(&start, nullptr);

    // device 1
    forward_kernel<<<grid_dim, block_dim, sram_size>>>(
        q_d0, k_d0, v_d0,
        N, d, Tc, Tr, Bc, Br, softmax_scale,
        rs_d0, rm_d0, o_d0
    );

    cudaSetDevice(0);
    forward_kernel<<<grid_dim, block_dim, sram_size>>>(
        q_d1, k_d1, v_d1,
        N, d, Tc, Tr, Bc, Br, softmax_scale,
        rs_d1, rm_d1, o_d1
    );

    gettimeofday(&end, nullptr);

	cudaMemcpy(&output[num_element_per_device], o_d1, size_per_device, cudaMemcpyDeviceToHost);
    cudaSetDevice(1); 
    cudaMemcpy(output, o_d0, size_per_device, cudaMemcpyDeviceToHost);


    float elapsedTime = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
	printf("%lf\n", elapsedTime);

    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);

    cudaFree(q_d0); cudaFree(k_d0); cudaFree(v_d0); cudaFree(o_d0);
    cudaFree(q_d1); cudaFree(k_d1); cudaFree(v_d1); cudaFree(o_d1);

    if (arg_test) { 
        // print_result(query, key, value, mask, output, num_batch, seq_length, embed_dim); 
        printf("o=\n"); print_3d(output, num_batch, seq_length, embed_dim);
    }

    delete[] query;
    delete[] key;
    delete[] value;
    delete[] output;
    delete[] row_max;
    delete[] row_sum;
}

int main(int argc, char* argv[]) {
    proc_arg(argc, argv);
    if (mode == 's') 
        test(); 
    else if (mode == 'm') { 
        if (arg_unified)
            test_multi_unified();
        else
            test_multi(); 
    }
    else { printf("mode %c not defined\n", mode); }
    return 0;
}

// nvcc ./src/attention_flash.cu matrix_io.cpp -lm -o ./src/attention_flash
// ./src/attention_flash -b 2 -s 2 -e 2 -t
