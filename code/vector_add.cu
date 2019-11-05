#include <stdio.h>
#include <assert.h>


/*
 * Vector addition on GPU
 * compile and run with nvcc -arch=sm_70 -o vector_add vector_add.cu -run
 *
 * written by Eric Bonnet 09/2019
 */


// cuda error handling function
cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

// initialize vectors
void initWith(float num, float *a, int N) {
    for(int i = 0; i < N; ++i)
        a[i] = num;
}

// GPU kernel function to add vectors a and b and put result into result
__global__ void addVectorsInto(float *result, float *a, float *b, int N) {

    // the vectors are larger than the grid so we use a gride-stride loop approach to manipulate the arrays
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i < N; i += stride)
        result[i] = a[i] + b[i];
}

// just check that the results are correct
void checkElementsAre(float target, float *array, int N) {
    for(int i = 0; i < N; i++) {
        if(array[i] != target) {
            printf("FAIL: array[%d] - %0.0f does not equal %0.0f\n", i, array[i], target);
            exit(1);
        }
    }
    printf("SUCCESS! All values added correctly.\n");
}

int main() {

    // vector size, here it is 2,097,152 
    const int N = 2<<20;
    size_t size = N * sizeof(float);

    // array pointers
    float *a;
    float *b;
    float *c;

    // allocate memory (unified memory on host and GPU) and check allocation is ok
    checkCuda(cudaMallocManaged(&a, size));
    checkCuda(cudaMallocManaged(&b, size));
    checkCuda(cudaMallocManaged(&c, size));

    // initialize all vectors
    initWith(3, a, N);
    initWith(4, b, N);
    initWith(0, c, N);

    // declare and set the number of blocks and the number of threads per block
    size_t threadsPerBlock;
    size_t numberOfBlocks;

    threadsPerBlock = 256;
    numberOfBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // add vectors on GPU
    addVectorsInto<<<numberOfBlocks, threadsPerBlock>>>(c, a, b, N);

    // check for errors and synchronize results
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());

    // check results
    checkElementsAre(7, c, N);

    // free memory
    checkCuda(cudaFree(a));
    checkCuda(cudaFree(b));
    checkCuda(cudaFree(c));
}

