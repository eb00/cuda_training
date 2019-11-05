#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"
#include "check.h"

#define SOFTENING 1e-9f


/*
 * Implementation of the n-body problem in pure c language.
 * compilation with 'gcc nbody.c -o nbody' 
 *
 */



/*
 * Each body contains x, y, and z coordinate positions,
 * as well as velocities in the x, y, and z directions.
 */

typedef struct { float x, y, z, vx, vy, vz; } Body;

/*
 * Random initialization of all elements (positions and velocities) 
 */

void randomizeBodies(float *data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    }
}

/*
 * This function calculates the gravitational impact of all bodies in the system
 * on all others, but does not update their positions.
 */

void bodyForce(Body *p, float dt, int n) {
    for (int i = 0; i < n; ++i) {
        float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

        for (int j = 0; j < n; j++) {
            float dx = p[j].x - p[i].x;
            float dy = p[j].y - p[i].y;
            float dz = p[j].z - p[i].z;
            float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
            //float invDist = rsqrtf(distSqr);
            float invDist = sqrt(distSqr);
            float invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
        }

        p[i].vx += dt*Fx; p[i].vy += dt*Fy; p[i].vz += dt*Fz;
    }
}

int main(const int argc, const char** argv) {

      /*
       * Total number of bodies considered in this example 
       * This number can be changed by passing values into the command line.
       */

      int nBodies = 2<<11;
      int salt = 0;
      if (argc > 1) nBodies = 2<<atoi(argv[1]);

      /*
       * This salt is for assessment reasons. Tampering with it will result in automatic failure.
       */

      if (argc > 2) salt = atoi(argv[2]);

      const float dt = 0.01f; // time step
      const int nIters = 10;  // simulation iterations

      int bytes = nBodies * sizeof(Body);
      float *buf;

      buf = (float *)malloc(bytes);

      Body *p = (Body*)buf;

      randomizeBodies(buf, 6 * nBodies); // Init pos / vel data

      double totalTime = 0.0;

      /*
       * This simulation will run for 10 cycles of time, calculating gravitational
       * interaction amongst bodies, and adjusting their positions to reflect.
       */

      for (int iter = 0; iter < nIters; iter++) {
            StartTimer();

            // compute interbody forces
            bodyForce(p, dt, nBodies);

            // integrate position
            for (int i = 0 ; i < nBodies; i++) {
                p[i].x += p[i].vx*dt;
                p[i].y += p[i].vy*dt;
                p[i].z += p[i].vz*dt;
            }

            const double tElapsed = GetTimer() / 1000.0;
            totalTime += tElapsed;
      }

      double avgTime = totalTime / (double)(nIters);
      float billionsOfOpsPerSecond = 1e-9 * nBodies * nBodies / avgTime;

    #ifdef ASSESS
      checkPerformance(buf, billionsOfOpsPerSecond, salt);
    #else
      checkAccuracy(buf, nBodies);
      printf("%d Bodies: average %0.3f Billion Interactions / second\n", nBodies, billionsOfOpsPerSecond);
      salt += 1;
    #endif

      free(buf);
}

