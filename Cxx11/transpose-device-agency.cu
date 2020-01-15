///
/// Copyright (c) 2020, Intel Corporation
/// Copyright (c) 2014, NVIDIA CORPORATION. All rights reserved.
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
///          transpose <matrix_size> <# iterations>
///
///          The output consists of diagnostics to make sure the
///          transpose worked and timing statistics.
///
/// HISTORY: Written by  Rob Van der Wijngaart, February 2009.
///          Converted to C++11 by Jeff Hammond, February 2016 and May 2017.
///          Agency implementation of Jared Hoberock integrated by Jeff Hammond, January 2020
///
//////////////////////////////////////////////////////////////////////

#include "prk_util.h"
#include "prk_cuda.h"
#include "prk_thrust.h"

// The implementation of transpose has been copied from
// https://github.com/agency-library/agency/blob/master/testing/unorganized/cuda/transpose.cu
// with float types replaced with double types

#include <agency/agency.hpp>
#include <agency/cuda.hpp>

const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;

struct transpose_naive
{
  __device__ void operator()(agency::cuda::grid_agent_2d& self, double* odata, const double* idata)
  {
    auto idx = TILE_DIM * self.outer().index() + self.inner().index();
    int width = self.outer().group_shape()[0] * TILE_DIM;

    for(int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
    {
      odata[idx[0]*width + (idx[1]+j)] += idata[(idx[1]+j)*width + idx[0]];
      idata[(idx[1]+j)*width + idx[0]] += 1.0;
    }
  }
};

struct tile2d_t
{
  double data[TILE_DIM][TILE_DIM];
};


struct transpose_coalesced
{
  __device__ void operator()(agency::cuda::grid_agent_2d& self, double* odata, const double* idata, tile2d_t& tile)
  {
    int x = self.outer().index()[0] * TILE_DIM + self.inner().index()[0];
    int y = self.outer().index()[1] * TILE_DIM + self.inner().index()[1];
    int width = self.outer().group_shape()[0] * TILE_DIM;

    for(int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
      tile.data[self.inner().index()[1]+j][self.inner().index()[0]] = idata[(y+j)*width + x];
      idata[(y+j)*width + x] += 1.0; // is this the right way to do this?
    }

    self.inner().wait();

    x = self.outer().index()[1] * TILE_DIM + self.inner().index()[0];  // transpose block offset
    y = self.outer().index()[0] * TILE_DIM + self.inner().index()[1];

    for(int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
      odata[(y+j)*width + x] += tile.data[self.inner().index()[0]][self.inner().index()[1] + j];
    }
  }
};


struct conflict_free_tile2d_t
{
  double data[TILE_DIM][TILE_DIM+1];
};


struct transpose_no_bank_conflicts
{
  __device__ void operator()(agency::cuda::grid_agent_2d& self, double* odata, const double* idata, conflict_free_tile2d_t& tile)
  {
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for(int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
      tile.data[threadIdx.y+j][threadIdx.x] = idata[(y+j)*width + x];
      idata[(y+j)*width + x] += 1.0; // is this the right way to do this?
    }

    self.inner().wait();

    x = self.outer().index()[1] * TILE_DIM + self.inner().index()[0];  // transpose block offset
    y = self.outer().index()[0] * TILE_DIM + self.inner().index()[1];

    for(int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
      odata[(y+j)*width + x] += tile.data[threadIdx.x][threadIdx.y + j];
    }
  }
};


int main(int argc, char * argv[])
{
  std::cout << "Parallel Research Kernels version " << PRKVERSION << std::endl;
  std::cout << "C++11/Thrust+Agency Matrix transpose: B = A^T" << std::endl;

  //////////////////////////////////////////////////////////////////////
  /// Read and test input parameters
  //////////////////////////////////////////////////////////////////////

  int iterations;
  int order;
  int variant = 0;
  try {
      if (argc < 3) {
        throw "Usage: <# iterations> <matrix order> [<variant=0,1,2>]";
      }

      // number of times to do the transpose
      iterations  = std::atoi(argv[1]);
      if (iterations < 1) {
        throw "ERROR: iterations must be >= 1";
      }

      // order of a the matrix
      order = std::atoi(argv[2]);
      if (order <= 0) {
        throw "ERROR: Matrix Order must be greater than 0";
      } else if (order > std::floor(std::sqrt(INT_MAX))) {
        throw "ERROR: matrix dimension too large - overflow risk";
      }
      if (n % TILE_DIM != 0) {
        throw "ERROR: matrix dimension is not an integer multiple of TILE_DIM";
      }
      if (TILE_DIM % BLOCK_ROWS!= 0) {
        throw "ERROR: TILE_DIM is not an integer multiple of BLOCK_ROWS";
      }

      if (argc > 3) {
        variant = std::atoi(argv[3]);
        if ( (variant < 0) || (variant > 2) ) {
          throw "ERROR: Invalid variant selected! (Use 0, 1 or 2.)";
        }
      }
  }
  catch (const char * e) {
    std::cout << e << std::endl;
    return 1;
  }

  std::string variant_name;
  switch (variant) {
      case 0: variant_name = "naive";             break;
      case 1: variant_name = "coalesced";         break;
      case 2: variant_name = "no bank conflicts"; break;
  }

  std::cout << "Number of iterations  = " << iterations << std::endl;
  std::cout << "Matrix order          = " << order << std::endl;
  std::cout << "Tile size             = " << TILE_DIM << std::endl;
  std::cout << "Algorithm variant     = " << variant_name << std::endl;

  agency::size2 dim_grid{nx/TILE_DIM, ny/TILE_DIM};
  agency::size2 dim_block{TILE_DIM, BLOCK_ROWS};

  //////////////////////////////////////////////////////////////////////
  /// Allocate space for the input and transpose matrix
  //////////////////////////////////////////////////////////////////////

  thrust::device_vector<double> A(order*order);
  thrust::device_vector<double> B(order*order);
  thrust::host_vector<double>   h_B(order*order,0.0);
  // fill A with the sequence 0 to order^2-1 as doubles
  thrust::sequence(thrust::device, A.begin(), A.end() );
  thrust::fill(thrust::device, B.begin(), B.end(), 0.0);

  auto trans_time = 0.0;

  for (int iter = 0; iter<=iterations; iter++) {

    if (iter==1) trans_time = prk::wtime();

    switch (variant) {
        case 0:
            agency::bulk_async(agency::cuda::grid(dim_grid, dim_block),
                               transpose_naive{},
                               raw_pointer_cast(B.data()), raw_pointer_cast(A.data()));
            break;
        case 1:
            agency::bulk_async(agency::cuda::grid(dim_grid, dim_block),
                               transpose_coalesced{},
                               raw_pointer_cast(B.data()), raw_pointer_cast(A.data()),
                               agency::share_at_scope<1,tile2d_t>());
            break;
        case 2:
            agency::bulk_async(agency::cuda::grid(dim_grid, dim_block),
                               transpose_no_bank_conflicts{},
                               raw_pointer_cast(B.data()), raw_pointer_cast(A.data()),
                               agency::share_at_scope<1,conflict_free_tile2d_t>());
            break;

  }
  trans_time = prk::wtime() - trans_time;

  // copy device to host
  h_B = B;

  //////////////////////////////////////////////////////////////////////
  /// Analyze and output results
  //////////////////////////////////////////////////////////////////////

  const auto addit = (iterations+1.) * (iterations/2.);
  auto abserr = 0.0;
  for (int j=0; j<order; j++) {
    for (int i=0; i<order; i++) {
      const int ij = i*order+j;
      const int ji = j*order+i;
      const double reference = static_cast<double>(ij)*(1.+iterations)+addit;
      abserr += std::fabs(h_B[ji] - reference);
    }
  }

#ifdef VERBOSE
  std::cout << "Sum of absolute differences: " << abserr << std::endl;
#endif

  const auto epsilon = 1.0e-8;
  if (abserr < epsilon) {
    std::cout << "Solution validates" << std::endl;
    auto avgtime = trans_time/iterations;
    auto bytes = (size_t)order * (size_t)order * sizeof(double);
    std::cout << "Rate (MB/s): " << 1.0e-6 * (2L*bytes)/avgtime
              << " Avg time (s): " << avgtime << std::endl;
  } else {
    std::cout << "ERROR: Aggregate squared error " << abserr
              << " exceeds threshold " << epsilon << std::endl;
    return 1;
  }

  return 0;
}


