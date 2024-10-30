///
/// Copyright (c) 2018, Intel Corporation
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
/// NAME:    sgemm
///
/// PURPOSE: This program tests the efficiency with which a dense matrix
///          dense multiplication is carried out
///
/// USAGE:   The program takes as input the matrix order,
///          the number of times the matrix-matrix multiplication
///          is carried out, and, optionally, a tile size for matrix
///          blocking
///
///          <progname> <# iterations> <matrix order> [<batches>]
///
///          The output consists of diagnostics to make sure the
///          algorithm worked, and of timing statistics.
///
/// FUNCTIONS CALLED:
///
///          Other than OpenMP or standard C functions, the following
///          functions are used in this program:
///
///          cblasSgemm()
///          cublasSgemmStridedBatched()
///
/// HISTORY: Written by Rob Van der Wijngaart, February 2009.
///          Converted to C++11 by Jeff Hammond, December, 2017.
///
//////////////////////////////////////////////////////////////////////

#include "prk_util.h"
#include "prk_cuda.h"

void prk_sgemm(const cublasHandle_t & h,
               const int order,
               float * A,
               float * B,
               float * C)
{
    const float alpha = 1.0;
    const float beta  = 1.0;

    prk::CUDA::check( cublasSgemm(h,
                                  CUBLAS_OP_N, CUBLAS_OP_N, // opA, opB
                                  order, order, order,      // m, n, k
                                  &alpha,                   // alpha
                                  A, order,                 // A, lda
                                  B, order,                 // B, ldb
                                  &beta,                    // beta
                                  C, order) );              // C, ldc
}

int main(int argc, char * argv[])
{
  std::cout << "Parallel Research Kernels version " << PRKVERSION << std::endl;
  std::cout << "C++11/CUBLAS Dense matrix-matrix multiplication: C += A x B" << std::endl;

  prk::CUDA::info info;
  //info.print();

  //////////////////////////////////////////////////////////////////////
  /// Read and test input parameters
  //////////////////////////////////////////////////////////////////////

  int iterations;
  int order;
  bool tf32{false};
  try {
      if (argc < 2) {
        throw "Usage: <# iterations> <matrix order> [<use TF32 [0/1]>]";
      }

      iterations  = std::atoi(argv[1]);
      if (iterations < 1) {
        throw "ERROR: iterations must be >= 1";
      }

      order = std::atoi(argv[2]);
      if (order <= 0) {
        throw "ERROR: Matrix Order must be greater than 0";
      } else if (order > prk::get_max_matrix_size()) {
        throw "ERROR: matrix dimension too large - overflow risk";
      }

      if (argc > 3) {
        tf32 = prk::parse_boolean(std::string(argv[3]));
      }
  }
  catch (const char * e) {
    std::cout << e << std::endl;
    return 1;
  }

  std::cout << "Number of iterations = " << iterations << std::endl;
  std::cout << "Matrix order         = " << order << std::endl;
  std::cout << "TF32                 = " << (tf32 ? "yes" : "no") << std::endl;

  cublasHandle_t h;
  prk::CUDA::check( cublasCreate(&h) );

  if (tf32) {
    cublasSetMathMode(h, CUBLAS_TF32_TENSOR_OP_MATH);
  }

  const int tile_size = 32;
  dim3 dimGrid(prk::divceil(order,tile_size),prk::divceil(order,tile_size),1);
  dim3 dimBlock(tile_size, tile_size, 1);

  info.checkDims(dimBlock, dimGrid);

  //////////////////////////////////////////////////////////////////////
  // Allocate space for matrices
  //////////////////////////////////////////////////////////////////////

  double gemm_time(0);

  const size_t nelems = (size_t)order * (size_t)order;

  // host buffers
  auto m_a = prk::CUDA::malloc_managed<float>(nelems);
  auto m_b = prk::CUDA::malloc_managed<float>(nelems);
  auto m_c = prk::CUDA::malloc_managed<float>(nelems);

  for (int i=0; i<order; ++i) {
    for (int j=0; j<order; ++j) {
       m_a[i*order+j] = i;
       m_b[i*order+j] = i;
       m_c[i*order+j] = 0;
    }
  }

  double xfer(0);
  double comp(0);
  {
    for (int iter = 0; iter<=iterations; iter++) {

      if (iter==1) {
          prk::CUDA::sync();
          gemm_time = prk::wtime();
      }

      {
        double t0 = prk::wtime();
        prk_sgemm(h, order, m_a, m_b, m_c);
        double t1 = prk::wtime();
        if (iter==0) xfer = (t1-t0);
        if (iter==1) comp = (t1-t0);
      }
    }
    prk::CUDA::sync();
    gemm_time = prk::wtime() - gemm_time;
  }
  xfer -= comp;
  std::cout << "xfer, comp = " << xfer << "," << comp << std::endl;

  prk::CUDA::free(m_a);
  prk::CUDA::free(m_b);

  prk::CUDA::check( cublasDestroy(h) );

  prk::CUDA::sync();

  //////////////////////////////////////////////////////////////////////
  /// Analyze and output results
  //////////////////////////////////////////////////////////////////////

  double epsilon;
  if(tf32) {
    epsilon = 1.0e-4;
  } else {
    epsilon = 1.0e-8;
  }
  const auto forder = static_cast<double>(order);
  const auto reference = 0.25 * prk::pow(forder,3) * prk::pow(forder-1.0,2) * (iterations+1);
  double residuum(0);
  const auto checksum = prk::reduce( &(m_c[0]), &(m_c[nelems]), 0.0);
  residuum += std::abs(checksum-reference)/reference;

  if (residuum < epsilon) {
#if VERBOSE
    std::cout << "Reference checksum = " << reference << "\n"
              << "Actual checksum = " << checksum << std::endl;
#endif
    std::cout << "Solution validates" << std::endl;
    auto avgtime = gemm_time/iterations;
    auto nflops = 2.0 * prk::pow(forder,3);
    std::cout << "Rate (MF/s): " << 1.0e-6 * nflops/avgtime
              << " Avg time (s): " << avgtime << std::endl;
  } else {
    std::cout << "Reference checksum = " << reference << "\n"
              << "Residuum           = " << residuum << std::endl;
    return 1;
  }

  prk::CUDA::free(m_c);

  return 0;
}


