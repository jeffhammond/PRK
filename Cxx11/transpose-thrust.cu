///
/// Copyright (c) 2013, Intel Corporation
/// Copyright (c) 2021, NVIDIA
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
///
//////////////////////////////////////////////////////////////////////

#include "prk_util.h"
#include "prk_cuda.h"
#include "prk_thrust.h"

struct printf_functor
{
  __host__ __device__
  void operator()(int x)
  {
    // note that using printf in a __device__ function requires
    // code compiled for a GPU with compute capability 2.0 or
    // higher (nvcc --arch=sm_20)
    printf("%d\n", x);
  }
};

struct square : public thrust::unary_function<int,int>
{
  __host__ __device__
  int operator()(int x) const
  {
    return x * x;
  }
};

int main(int argc, char * argv[])
{
  std::cout << "Parallel Research Kernels version " << PRKVERSION << std::endl;
  std::cout << "C++11/Thrust Matrix transpose: B = A^T" << std::endl;

  //////////////////////////////////////////////////////////////////////
  /// Read and test input parameters
  //////////////////////////////////////////////////////////////////////

  int iterations;
  int order;
  try {
      if (argc < 3) {
        throw "Usage: <# iterations> <matrix order>";
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
      } else if (order > prk::get_max_matrix_size()) {
        throw "ERROR: matrix dimension too large - overflow risk";
      }
  }
  catch (const char * e) {
    std::cout << e << std::endl;
    return 1;
  }

  std::cout << "Number of iterations  = " << iterations << std::endl;
  std::cout << "Matrix order          = " << order << std::endl;

  //////////////////////////////////////////////////////////////////////
  /// Allocate space for the input and transpose matrix
  //////////////////////////////////////////////////////////////////////

  thrust::universal_vector<double> A(order*order);
  thrust::universal_vector<double> B(order*order);
  // fill A with the sequence 0 to order^2-1 as doubles
  thrust::sequence(thrust::device, A.begin(), A.end() );
  thrust::fill(thrust::device, B.begin(), B.end(), 0.0);

  thrust::counting_iterator<int> first(0);
  thrust::counting_iterator<int> last(order);
  thrust::for_each(thrust::device, first, last, printf_functor() );

  //thrust::transform_iterator<int> first2(0,[=](int x){ return x*x; });
  //thrust::transform_iterator<int> last2(order*order);
  //thrust::for_each(thrust::device, first2, last2, printf_functor() );

  //auto range = std::views::iota(0,order);
  thrust::universal_vector<int> range(order);
  thrust::sequence(range.begin(), range.end());

  double trans_time{0};

  for (int iter = 0; iter<=iterations; iter++) {

    if (iter==1) trans_time = prk::wtime();

    thrust::for_each( thrust::device,
                      std::begin(range), std::end(range),
                      [=] __device__ (int i) {
      //thrust::for_each( thrust::device, std::begin(range), std::end(range), [&] (int j) {
      //std::for_each( std::begin(range), std::end(range), [&] (int j) {
      for (int j=0; j<order; j++) {
          B[i*order+j] += A[j*order+i];
          A[j*order+i] += 1.0;
      };
      //});
    });
  }
  trans_time = prk::wtime() - trans_time;

  //////////////////////////////////////////////////////////////////////
  /// Analyze and output results
  //////////////////////////////////////////////////////////////////////

  const double addit = (iterations+1.) * (iterations/2.);
  double abserr(0);
  // TODO: replace with std::generate, std::accumulate, or similar
  for (int j=0; j<order; j++) {
    for (int i=0; i<order; i++) {
      const int ij = i*order+j;
      const int ji = j*order+i;
      const double reference = static_cast<double>(ij)*(1.+iterations)+addit;
      abserr += prk::abs(B[ji] - reference);
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


