#include <iostream>
#include <random>
#include <vector>
#include <cstdlib>
using namespace std;
#define NEG_INF -1e9

default_random_engine generator(123);
uniform_real_distribution<float> distribution(0.0, 1.0);

float generate_element(char mode) {

    float gen_element;
    switch (mode) {
        case 'n': // negative infinity
            gen_element = NEG_INF;
            break;
        case 'r': // random
            gen_element = distribution(generator);
            break;
        case 'z': // zero
            gen_element = 0.0f;
            break;
        case 'b': // boolean
            gen_element = (distribution(generator) * 100) > 40;
            break;
    }
    return gen_element;
}

float* generate_matrix(const int* dims, int num_dim, char mode='r') {
    if (num_dim <= 0)
        throw std::invalid_argument("Input array must have at least one element.");

    // Calculate the product of the input array elements
    unsigned int product = 1; // Use long long to handle large products
    for (int i = 0; i < num_dim; ++i) { product *= dims[i]; }

    // Allocate a float array of size equal to the product
    float* outputArray = new float[product];

    // Initialize the output array (optional)
    for (int i = 0; i < product; ++i)
        outputArray[i] = generate_element(mode); // or any other initialization

    return outputArray;
}

void print_3d(float* matrix, int b, int m, int n) {
    for (int h = 0; h < b; ++h) {
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j)
                printf("%.4f ", matrix[h*m*n+i*n+j]);
            printf("\n");
        }
    }
}

void print_result(
    float* query, float* key, float* value, float* mask, float* output,
    int num_batch, int seq_length, int embed_dim
) {
    printf("q=\n"); 
    print_3d(query, num_batch, seq_length, embed_dim);
    printf("\nk=\n"); 
    print_3d(key, num_batch, seq_length, embed_dim);
    printf("\nv=\n"); 
    print_3d(value, num_batch, seq_length, embed_dim);
    if (mask) {
        printf("\nm=\n"); 
        print_3d(mask, num_batch, seq_length, seq_length);
    }
    printf("\no=\n"); 
    print_3d(output, num_batch, seq_length, embed_dim);
    printf("\n");
}