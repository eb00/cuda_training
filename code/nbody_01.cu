#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"
#include "check.h"

#define SOFTENING 1e-9f

/*
 * Implementation of a n-body simulator with GPU acceleration.
 * This simulator predicts the individual motion of a group of objects 
 * interacting gravitationally. 
 * 
 * compilation: nvcc -arch=sm_70 -o nbody nbody_01.cu
 * run with ./nbody 11
 * optional: profile with 'nvprof ./nbody'
 *
 * performance: average of 35.099 Billion Interactions / second for 4096 bodies (NVIDIA V100) 
 * CPU: 30 Million Interactions / sec.
 *
 * written by Eric Bonnet 09/2019
 */


/*
 * Each body contains x, y, and z coordinate positions,
 * as well as velocities in the x, y, and z directions.
 */

typedef struct { float x, y, z, vx, vy, vz; } Body;


void randomizeBodies(float *data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    }
}

/*
 * This function calculates the gravitational impact of all bodies in the system
 * on all others, but does not update their positions.
 *
 * kernel version of the function for execution on GPU
 */

__global__ void bodyForce(Body *p, float dt, int n) {

    // calculate index in function of cuda block and thread id 
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < n) {

        float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

        for (int j = 0; j < n; j++) {
            float dx = p[j].x - p[i].x;
            float dy = p[j].y - p[i].y;
            float dz = p[j].z - p[i].z;
            float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
            float invDist = rsqrtf(distSqr);
            float invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
        }

        p[i].vx += dt*Fx; p[i].vy += dt*Fy; p[i].vz += dt*Fz;
    }
}

/*
 * kernel function for integration
 */

__global__ void integrate(Body *p, float dt, int n) {

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
      p[i].x += p[i].vx*dt;
      p[i].y += p[i].vy*dt;
      p[i].z += p[i].vz*dt;
    }
}

int main(const int argc, const char** argv) {

      int nBodies = 2<<11;
      int salt = 0;
      if (argc > 1) nBodies = 2<<atoi(argv[1]);

      if (argc > 2) salt = atoi(argv[2]);

      const float dt = 0.01f; // time step
      const int nIters = 10;  // simulation iterations

      int bytes = nBodies * sizeof(Body);
      float *buf;

      // get device Id
      int deviceId;
      cudaGetDevice(&deviceId);

      // Allocate managed memory
      cudaMallocManaged(&buf, bytes);

      Body *p = (Body*)buf;

      size_t threadsPerBlock;
      size_t numberOfBlocks;

      threadsPerBlock = 256;
      numberOfBlocks = (nBodies + threadsPerBlock - 1) / threadsPerBlock;
      printf("Number of blocks = %i\n", (int)numberOfBlocks);

      randomizeBodies(buf, 6 * nBodies); // Init pos / vel data

      double totalTime = 0.0;
      cudaError_t asyncErr, integrateErr, bodyForceErr;

      /*
       * This simulation will run for 10 cycles of time, calculating gravitational
       * interaction amongst bodies, and adjusting their positions to reflect.
       */

      // Allocate memory on GPU card
      cudaMemPrefetchAsync(buf, bytes, deviceId);

      for (int iter = 0; iter < nIters; iter++) {

          StartTimer();

          // call kernel function bodyForce
          bodyForce<<<numberOfBlocks, threadsPerBlock>>>(p, dt, nBodies); // compute interbody forces

          bodyForceErr = cudaGetLastError();
          if (bodyForceErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(bodyForceErr));

          // synchronize results
          asyncErr = cudaDeviceSynchronize();
          if (asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));

          // call kernel function for integration
          integrate<<<numberOfBlocks, threadsPerBlock>>>(p, dt, nBodies); // compute interbody forces

          // check if anything weird happened
          integrateErr = cudaGetLastError();
          if (integrateErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(integrateErr));

          // synchronize results
          asyncErr = cudaDeviceSynchronize();
          if(asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));

          const double tElapsed = GetTimer() / 1000.0;
          totalTime += tElapsed;
    }

    // get results back on device
    cudaMemPrefetchAsync(buf, bytes, cudaCpuDeviceId);

    double avgTime = totalTime / (double)(nIters);
    float billionsOfOpsPerSecond = 1e-9 * nBodies * nBodies / avgTime;

    #ifdef ASSESS
        checkPerformance(buf, billionsOfOpsPerSecond, salt);
    #else
        checkAccuracy(buf, nBodies);
        printf("%d Bodies: average %0.3f Billion Interactions / second\n", nBodies, billionsOfOpsPerSecond);
        salt += 1;
    #endif

    cudaFree(buf);
}
