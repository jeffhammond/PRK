#pragma omp requires unified_shared_memory

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

int main(int argc, char * argv[])
{
  printf("Parallel Research Kernels version %d\n", PRKVERSION );
  printf("C11/OpenMP TARGET STREAM triad: A = B + scalar * C\n");

  //////////////////////////////////////////////////////////////////////
  /// Read and test input parameters
  //////////////////////////////////////////////////////////////////////

  if (argc < 3) {
    printf("Usage: <# iterations> <vector length>\n");
    return 1;
  }

  int iterations = atoi(argv[1]);
  if (iterations < 1) {
    printf("ERROR: iterations must be >= 1\n");
    return 1;
  }

  // length of a the vector
  size_t length = atol(argv[2]);
  if (length <= 0) {
    printf("ERROR: Vector length must be greater than 0\n");
    return 1;
  }

  int device = (argc > 3) ? atol(argv[3]) : omp_get_default_device();
  if ( (device < 0 || omp_get_num_devices() <= device ) && (device != omp_get_default_device()) ) {
    printf("ERROR: device number %d is not valid.\n", device);
    return 1;
  }

  printf("Number of iterations = %d\n", iterations);
  printf("Vector length        = %zu\n", length);
  printf("OpenMP Device        = %d\n", device);

  //////////////////////////////////////////////////////////////////////
  // Allocate space and perform the computation
  //////////////////////////////////////////////////////////////////////

  double nstream_time = 0.0;

  size_t bytes = length*sizeof(double);
  double * restrict A = omp_target_alloc(bytes, device);
  double * restrict B = omp_target_alloc(bytes, device);
  double * restrict C = omp_target_alloc(bytes, device);

  double scalar = 3.0;

  #pragma omp target teams distribute parallel for simd schedule(static) device(device)
  for (size_t i=0; i<length; i++) {
      A[i] = 0.0;
      B[i] = 2.0;
      C[i] = 2.0;
  }

  {
    for (int iter = 0; iter<=iterations; iter++) {

      if (iter==1) nstream_time = omp_get_wtime();

      #pragma omp target teams distribute parallel for simd schedule(static) device(device)
      for (size_t i=0; i<length; i++) {
          A[i] += B[i] + scalar * C[i];
      }
    }
    nstream_time = omp_get_wtime() - nstream_time;
  }

  omp_target_free(C, device);
  omp_target_free(B, device);

  //////////////////////////////////////////////////////////////////////
  /// Analyze and output results
  //////////////////////////////////////////////////////////////////////

  double ar = 0.0;
  double br = 2.0;
  double cr = 2.0;
  for (int i=0; i<=iterations; i++) {
      ar += br + scalar * cr;
  }

  ar *= length;

  double asum = 0.0;
  #pragma omp target teams distribute parallel for reduction(+:asum) device(device)
  for (size_t i=0; i<length; i++) {
      asum += fabs(A[i]);
  }

  omp_target_free(A, device);

  double epsilon=1.e-8;
  if (fabs(ar-asum)/asum > epsilon) {
      printf("Failed Validation on output array\n"
             "       Expected checksum: %lf\n"
             "       Observed checksum: %lf\n"
             "ERROR: solution did not validate\n", ar, asum);
      return 1;
  } else {
      printf("Solution validates\n");
      double avgtime = nstream_time/iterations;
      double nbytes = 4.0 * length * sizeof(double);
      printf("Rate (MB/s): %lf Avg time (s): %lf\n", 1.e-6*nbytes/avgtime, avgtime);
  }

  return 0;
}


