#include <unistd.h>
#include "matrix_io.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <sys/time.h>
#include <omp.h>
using namespace std;

bool arg_test = false;
bool arg_mask = false;
int num_batch = 1;
int embed_dim = 1;
int seq_length = 1;

void proc_arg(int argc, char **argv) {
    int c;
    while ((c = getopt(argc, argv, "tkb:e:s:")) != -1) {
        switch (c) {
            case 't':
                arg_test = true;
                break;
            case 'k':
                arg_mask = true;
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

void softmax(float* matrix, int b, int m, int n) {
    for (int h = 0; h < b; ++h) {
        #pragma omp parallel for
        for (int i = 0; i < m; ++i) {
            float max_val = -1e9;
            float sum_exp = 0.0f;

            for (int j = 0; j < n; ++j) // find maximum value in row
                max_val = max(max_val, matrix[h*m*n+i*n+j]);

            for (int j = 0; j < n; ++j) { // compute exponentials and sum 
                matrix[h*m*n+i*n+j] = exp(matrix[h*m*n+i*n+j] - max_val); // subtract max for stability
                sum_exp += matrix[h*m*n+i*n+j];
            }

            for (int j = 0; j < n; ++j) // normalize to get probabilities
                matrix[h*m*n+i*n+j] /= sum_exp;
        }
    }
} 

void matrix_multiply(float* a, float* b, float* c, int nb, int m, int r, int n) {
    int h, i, j, k;

    #pragma omp parallel for private(h,i,j,k) shared(a,b,c,m,r,n,nb) 
    for (h = 0; h < nb; ++h) {
        int batch_length_a = h*m*r;
        int batch_length_b = h*r*n;
        int batch_length_c = h*m*n;
        // #pragma omp parallel for private(i,j,k,sum) shared(a,b,c,m,r,n,batch_length) 
        // #pragma omp for
        for (i = 0; i < m; i++) {
            for (j = 0; j < n; j++) {
                float sum = 0.0f;
                for (k = 0; k < r; k++)
                    sum += a[batch_length_a+i*r+k] * b[batch_length_b+k*n+j];
                c[batch_length_c+i*n+j] = sum;
            }
        }
    }
}

void matrix_multiply_transpose(float* a, float* b, float* c, int nb, int m, int r, int n) {
    for (int h = 0; h < nb; ++h) {
        int batch_len_a = h*m*r;
        int batch_len_b = h*r*n;
        int batch_len_c = h*m*n;
        #pragma omp parallel for shared(a,b,c,m,r,n,nb) 
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                float sum = 0.0f;
                for (int k = 0; k < r; k++)
                    sum += a[batch_len_a+i*r+k] * b[batch_len_b+j*r+k];
                c[batch_len_c+i*n+j] = sum;
            }
        }
    }
}

void matrix_transpose(float* a, float* aT, int b, int m, int n) {
    for (int h = 0; h < b; ++h) {
        #pragma omp parallel for
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++)
                aT[h*m*n+j*m+i] = a[h*m*n+i*n+j];
        }
    }
}

void normalize(float* matrix, int b, int m, int n, float norm_const) {
    for (int h = 0; h < b; ++h) {
        #pragma omp parallel for
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++)
                matrix[h*m*n+i*n+j] /= norm_const;
        }
    }
}

void matrix_mask(float* matrix, int* mask, int b, int m, int n) {
    for (int h = 0; h < b; ++h) {
        #pragma omp parallel for
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (mask[h*m*n+i*n+j] == 0)
                    matrix[h*m*n+i*n+j] = -1e9;
            }
        }
    }
}

void scaled_dot_product_attention(
    float* query, float* key, float* value, float* mask, int b, int m, int n, float* output
) {
    int dims_score[] = {b, m, m}; 
    float* scores = generate_matrix(dims_score, 3, 'z');
    
    int dims_valueT[] = {b, m, n}; 
    float* valueT = generate_matrix(dims_valueT, 3, 'z');
    
    float sum;
    float dk = sqrt((float) n);

    matrix_multiply_transpose(query, key, scores, b, m, n, m);

    normalize(scores, b, m, m, dk);

    if (mask) matrix_mask(scores, (int*)mask, b, m, m);

    softmax(scores, b, m, m);
    
    matrix_transpose(value, valueT, b, m, n);
    matrix_multiply_transpose(scores, valueT, output, b, m, m, n);

    delete[] scores;
    delete[] valueT;
}

void test() {
	struct timeval start, end;
    float *query, *key, *value, *mask, *output;

    int num_dim = 3;
    int dims[] = {num_batch, seq_length, embed_dim};
    int dims_mask[] = {num_batch, seq_length, seq_length};

    query = generate_matrix(dims, num_dim);
    key = generate_matrix(dims, num_dim);
    value = generate_matrix(dims, num_dim); // random
    mask = arg_mask ? generate_matrix(dims_mask, num_dim, 'b') : nullptr; // boolean
    output = generate_matrix(dims, num_dim, 'z'); // zero

    gettimeofday(&start, nullptr);
    scaled_dot_product_attention(query, key, value, mask, num_batch, seq_length, embed_dim, output);
    gettimeofday(&end, nullptr);
	float elapsedTime = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
	printf("%lf\n", elapsedTime); // printf("time: %lf\n", elapsedTime);

    if (arg_test) { 
        // print_result(query, key, value, mask, output, num_batch, seq_length, embed_dim); 
        printf("o=\n"); print_3d(output, num_batch, seq_length, embed_dim);
    }

    delete[] query, delete[] key, delete[] value, delete[] mask, delete[] output;
}

int main(int argc, char* argv[]) {
    proc_arg(argc, argv);
    test();
    return 0;
}

// g++ -fopenmp -DENABLE_OPENMP ./src/attention_openmp.cpp matrix_io.cpp -o ./src/attention_openmp -std=c++11
// ./src/attention_openmp -b 2 -s 2 -e 2
