///
/// Copyright (c) 2013, Intel Corporation
///
/// Redistribution and use in source and binary forms, with or without
/// modification, are permitted provided that the following conditions
/// are met:
///
/// * Redistributions of source code must retain the above copyright
///       notice, this list of conditions and the following disclaimer.
/// * Redistributions in binary form must reproduce the above
///       copyright notice, this list of conditions and the following
///       disclaimer in the documentation and/or other materials provided
///       with the distribution.
/// * Neither the name of Intel Corporation nor the names of its
///       contributors may be used to endorse or promote products
///       derived from this software without specific prior written
///       permission.
///
/// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
/// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
/// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
/// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
/// COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
/// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
/// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
/// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
/// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
/// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
/// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
/// POSSIBILITY OF SUCH DAMAGE.

//////////////////////////////////////////////////////////////////////
///
/// NAME:    transpose
///
/// PURPOSE: This program measures the time for the transpose of a
///          column-major stored matrix into a row-major stored matrix.
///
/// USAGE:   Program input is the matrix order and the number of times to
///          repeat the operation:
///
///          transpose <matrix_size> <# iterations> [tile size]
///
///          An optional parameter specifies the tile size used to divide the
///          individual matrix blocks for improved cache and TLB performance.
///
///          The output consists of diagnostics to make sure the
///          transpose worked and timing statistics.
///
/// HISTORY: Written by  Rob Van der Wijngaart, February 2009.
///          Converted to C++11 by Jeff Hammond, February 2016 and May 2017.
///          C11-ification by Jeff Hammond, June 2017.
///
//////////////////////////////////////////////////////////////////////

#include <mpi.h>
#include "prk_util.h"

int main(int argc, char * argv[])
{
  MPI_Init(&argc,&argv);

  int me, np;
  MPI_Comm_rank(MPI_COMM_WORLD, &me);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  if (me==0) {
    printf("Parallel Research Kernels version %d\n", PRKVERSION );
    printf("C11/MPI Matrix transpose: B = A^T\n");
  }

  //////////////////////////////////////////////////////////////////////
  /// Read and test input parameters
  //////////////////////////////////////////////////////////////////////

  int iterations;
  size_t order, block_order, tile_size;
  if (me==0) {
    if (argc < 3) {
      printf("Usage: <# iterations> <matrix order> [tile size]\n");
      MPI_Abort(MPI_COMM_WORLD,1);
    }

    // number of times to do the transpose
    iterations = atoi(argv[1]);
    if (iterations < 1) {
      printf("ERROR: iterations must be >= 1\n");
      MPI_Abort(MPI_COMM_WORLD,1);
    }

    // order of a the matrix
    order = atol(argv[2]);
    if (order <= 0) {
      printf("ERROR: Matrix Order must be greater than 0\n");
      MPI_Abort(MPI_COMM_WORLD,1);
    }
    if ((order % np) != 0) {
      printf("ERROR: Matrix Order should be divisible by # procs, %d\n", np);
      MPI_Abort(MPI_COMM_WORLD,1);
    } else {
        block_order = order / np;
    }

    // default tile size for tiling of local transpose
    tile_size = (argc>3) ? atoi(argv[3]) : 32;
    // a negative tile size means no tiling of the local transpose
    if (tile_size <= 0) tile_size = order;

    printf("Number of processes   = %d\n", np);
    printf("Number of iterations  = %d\n", iterations);
    printf("Matrix order          = %zu\n", order);
    printf("Tile size             = %zu\n", tile_size);
  }

  const MPI_Datatype PRK_SIZE_T = sizeof(size_t)==sizeof(int64_t) ? MPI_INT64_T : MPI_INT32_T;
  MPI_Bcast(&iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&order, 1, PRK_SIZE_T, 0, MPI_COMM_WORLD);
  MPI_Bcast(&tile_size, 1, PRK_SIZE_T, 0, MPI_COMM_WORLD);
  MPI_Bcast(&block_order, 1, PRK_SIZE_T, 0, MPI_COMM_WORLD);

  int count = 0;
  if (order*block_order > MPI_INT) {
    printf("ERROR: order*block_order (%zu) is too large for MPI_Alltoall\n", order*block_order);
    MPI_Abort(MPI_COMM_WORLD,2);
  } else {
    count = (int)(order*block_order);
  }

  //////////////////////////////////////////////////////////////////////
  /// Allocate space for the input and transpose matrix
  //////////////////////////////////////////////////////////////////////

  double trans_time = 0.0;

  size_t bytes = count*sizeof(double);
  double * restrict A = prk_malloc(bytes);
  double * restrict B = prk_malloc(bytes);
  double * restrict T = prk_malloc(bytes);

  for (size_t i=0; i<order; i++) {
    for (size_t j=0; j<block_order; j++) {
      const size_t offset = me * block_order;
      A[i*order+j] = (double)(i*order+offset+j);
      B[i*order+j] = 0.0;
      T[i*order+j] = 0.0;
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);

  for (int iter = 0; iter<=iterations; iter++) {

    if (iter==1) {
        MPI_Barrier(MPI_COMM_WORLD);
        trans_time = MPI_Wtime();
    }

    // this is designed to match the mpi4py implementation,
    // which uses ~50% more memory than the C89/MPI1 version.

    // printing only
    for (int r=0; r<np; r++) {
      if (me==r) {
        for (size_t i=0; i<order; i++) {
          for (size_t j=0; j<block_order; j++) {
            printf("%d: A(%zu,%zu)=%lf\n", me, i, j, A[i*order+j]);
          }
        }
        fflush(stdout);
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }

    // global transpose - change to large-count version some day
    MPI_Alltoall(A, count, MPI_DOUBLE,
                 T, count, MPI_DOUBLE,
                 MPI_COMM_WORLD);

    // printing only
    for (int r=0; r<np; r++) {
      if (me==r) {
        for (size_t i=0; i<order; i++) {
          for (size_t j=0; j<block_order; j++) {
            printf("%d: T(%zu,%zu)=%lf\n", me, i, j, T[i*order+j]);
          }
        }
        fflush(stdout);
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }

    // local transpose
    for (int r=0; r<np; r++) {
      const size_t lo = block_order * r;
      const size_t hi = block_order * (r+1);
      for (size_t i=lo; i<hi; i++) {
        for (size_t j=0; j<order; j++) {
          B[i*order+j] += T[j*order+i];
        }
      }
    }

    // A += 1
    for (size_t j=0;j<block_order; j++) {
      for (size_t i=0;i<order; i++) {
        A[j*order+i] += 1.0;
      }
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  trans_time = MPI_Wtime() - trans_time;

  //////////////////////////////////////////////////////////////////////
  // Analyze and output results
  //////////////////////////////////////////////////////////////////////

  double abserr = 0.0;
  const double addit = (iterations+1.) * (iterations/2.);
  for (size_t j=0; j<order; j++) {
    for (size_t i=0; i<order; i++) {
      const size_t ij = i*order+j;
      const size_t ji = j*order+i;
      const double reference = (double)(ij)*(1.+iterations)+addit;
      abserr += fabs(B[ji] - reference);
    }
  }

  //prk_free(A);
  //prk_free(B);
  //prk_free(T);

#ifdef VERBOSE
  printf("Sum of absolute differences: %lf\n", abserr);
#endif

  const double epsilon = 1.0e-8;
  if (abserr < epsilon) {
    printf("Solution validates\n");
    const double avgtime = trans_time/iterations;
    printf("Rate (MB/s): %lf Avg time (s): %lf\n", 2.0e-6 * bytes/avgtime, avgtime );
  } else {
    printf("ERROR: Aggregate squared error %e exceeds threshold %e\n", abserr, epsilon );
    return 1;
  }

  MPI_Finalize();

  return 0;
}


