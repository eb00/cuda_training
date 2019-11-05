#include <stdio.h>

#define N  64

/*
 *
 * GPU accelerated matrix multiplication
 * Compile & run with: nvcc -o -arch=sm_70 -o matrix_multiply_2d matrix_multiply_2d.cu -run
 *
 * written by Eric Bonnet 09/2019
 */

// GPU kernel function version for matrix multiplication
__global__ void matrixMulGPU( int * a, int * b, int * c ) {

      int val = 0;

      // calculate 2D indices depending on block and thread Ids
      int row = blockIdx.x * blockDim.x + threadIdx.x;
      int col = blockIdx.y * blockDim.y + threadIdx.y;

      // compute matrix multiplication for a given row and column in the matrix
      if (row < N && col < N) {
            for ( int k = 0; k < N; ++k )
                val += a[row * N + k] * b[k * N + col];
            c[row * N + col] = val;
      }
}

// CPU function for matrix multiplication
void matrixMulCPU( int * a, int * b, int * c ) {

      int val = 0;

      for( int row = 0; row < N; ++row )
            for( int col = 0; col < N; ++col ) {
                  val = 0;
                  for ( int k = 0; k < N; ++k )
                      val += a[row * N + k] * b[k * N + col];
                  c[row * N + col] = val;
            }
}

int main() {
      
      // Allocate a solution matrix for both the CPU and the GPU operations
      int *a, *b, *c_cpu, *c_gpu;
      
      // Number of bytes of an N x N matrix
      int size = N * N * sizeof (int);

      // Allocate memory (unified memory system)
      cudaMallocManaged (&a, size);
      cudaMallocManaged (&b, size);
      cudaMallocManaged (&c_cpu, size);
      cudaMallocManaged (&c_gpu, size);

      // Initialize memory; create 2D matrices
      for( int row = 0; row < N; ++row )
            for( int col = 0; col < N; ++col ) {
                  a[row*N + col] = row;
                  b[row*N + col] = col+2;
                  c_cpu[row*N + col] = 0;
                  c_gpu[row*N + col] = 0;
            }

      /*
       * Assign `threads_per_block` and `number_of_blocks` 2D values that can be used in matrixMulGPU above.
       */

      dim3 threads_per_block(16, 16, 1);
      dim3 number_of_blocks ((N / threads_per_block.x) + 1, (N / threads_per_block.y) + 1, 1);

      // call GPU matrix multiplication kernel function
      matrixMulGPU <<< number_of_blocks, threads_per_block >>> ( a, b, c_gpu );

      // synchronize results
      cudaDeviceSynchronize();

      // Call the CPU version for comparison 
      matrixMulCPU( a, b, c_cpu );

      // Compare CPU and GPU results 
      bool error = false;
      for( int row = 0; row < N && !error; ++row )
        for( int col = 0; col < N && !error; ++col )
          if (c_cpu[row * N + col] != c_gpu[row * N + col])
          {
            printf("FOUND ERROR at c[%d][%d]\n", row, col);
            error = true;
            break;
          }
      if (!error)
        printf("Success!\n");

      // Free all our allocated memory
      cudaFree(a);
      cudaFree(b);
      cudaFree( c_cpu ); 
      cudaFree( c_gpu );
}

