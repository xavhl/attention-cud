#include <iostream>
#include <random>
#include <vector>
using namespace std;

float* generate_matrix(const int* dims, int num_dim, char mode='r');

void print_3d(float* matrix, int b, int h, int w);

void print_result(
    float* query, float* key, float* value, float* mask, float* output,
    int num_batch, int seq_length, int embed_dim
);