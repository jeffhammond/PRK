///
/// Copyright (c) 2017, Intel Corporation
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
/// NAME:    nstream
///
/// PURPOSE: To compute memory bandwidth when adding a vector of a given
///          number of double precision values to the scalar multiple of
///          another vector of the same length, and storing the result in
///          a third vector.
///
/// USAGE:   The program takes as input the number
///          of iterations to loop over the triad vectors, the length of the
///          vectors, and the offset between vectors
///
///          <progname> <# iterations> <vector length> <offset>
///
///          The output consists of diagnostics to make sure the
///          algorithm worked, and of timing statistics.
///
/// NOTES:   Bandwidth is determined as the number of words read, plus the
///          number of words written, times the size of the words, divided
///          by the execution time. For a vector length of N, the total
///          number of words read and written is 4*N*sizeof(double).
///
/// HISTORY: This code is loosely based on the Stream benchmark by John
///          McCalpin, but does not follow all the Stream rules. Hence,
///          reported results should not be associated with Stream in
///          external publications
///
///          Converted to C++11 by Jeff Hammond, November 2017.
///
//////////////////////////////////////////////////////////////////////

#include "prk_util.h"
#include "prk_cuda.h"

__global__ void init(const unsigned n, double * A, double value)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        A[i] = value;
    }
}

__global__ void nstream(const unsigned n, const double scalar, double * A, const double * B, const double * C)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        A[i] += B[i] + scalar * C[i];
    }
}

int main(int argc, char * argv[])
{
  std::cout << "Parallel Research Kernels version " << PRKVERSION << std::endl;
  std::cout << "C++11/CUDA STREAM triad: A = B + scalar * C" << std::endl;

  prk::CUDA::info info;
  info.print();

  auto qs = prk::CUDA::queues();

  //////////////////////////////////////////////////////////////////////
  /// Read and test input parameters
  //////////////////////////////////////////////////////////////////////

  int iterations;
  int length, local_length;
  int use_ngpu = 1;
  try {
      if (argc < 3) {
        throw "Usage: <# iterations> <vector length> [<use_ngpu>]";
      }

      iterations  = std::atoi(argv[1]);
      if (iterations < 1) {
        throw "ERROR: iterations must be >= 1";
      }

      length = std::atoi(argv[2]);
      if (length <= 0) {
        throw "ERROR: vector length must be positive";
      }

      if (argc > 3) {
        use_ngpu = std::atoi(argv[3]);
      }
      if ( use_ngpu > qs.size() ) {
          std::string error = "You cannot use more devices ("
                            + std::to_string(use_ngpu)
                            + ") than you have ("
                            + std::to_string(qs.size()) + ")";
          throw error;
      }

      if (length % use_ngpu != 0) {
          std::string error = "ERROR: vector length ("
                            + std::to_string(length)
                            + ") should be divisible by # procs ("
                            + std::to_string(use_ngpu) + ")";
          throw error;
      }
      local_length = length / use_ngpu;
  }
  catch (const char * e) {
    std::cout << e << std::endl;
    return 1;
  }

  std::cout << "Number of devices     = " << use_ngpu << std::endl;
  std::cout << "Number of iterations  = " << iterations << std::endl;
  std::cout << "Vector length         = " << length << std::endl;
  std::cout << "Vector length (local) = " << local_length << std::endl;

  int np = use_ngpu;
  qs.resize(np);

  const int blockSize = 256;
  dim3 dimBlock(blockSize, 1, 1);
  dim3 dimGrid(prk::divceil(length,blockSize), 1, 1);

  info.checkDims(dimBlock, dimGrid);

  //////////////////////////////////////////////////////////////////////
  // Allocate space and perform the computation
  //////////////////////////////////////////////////////////////////////

  double nstream_time(0);

  auto h_A = prk::CUDA::vector<double>(length, 0);
  auto h_B = prk::CUDA::vector<double>(length, 2);
  auto h_C = prk::CUDA::vector<double>(length, 2);

  auto d_A = std::vector<double*>(np, nullptr);
  auto d_B = std::vector<double*>(np, nullptr);
  auto d_C = std::vector<double*>(np, nullptr);

  qs.allocate<double>(d_A, local_length);
  qs.allocate<double>(d_B, local_length);
  qs.allocate<double>(d_C, local_length);
  qs.waitall();

  qs.scatter<double>(d_A, h_A, local_length);
  qs.scatter<double>(d_B, h_B, local_length);
  qs.scatter<double>(d_C, h_C, local_length);
  qs.waitall();

  // overwrite host buffer with garbage to detect bugs
  h_A.fill(-77777777);

  const double scalar(3);
  {
      for (int iter = 0; iter<=iterations; iter++) {

        if (iter==1) nstream_time = prk::wtime();

        for (int i=0; i<np; ++i) {
            //std::cerr << "INFO: device: " << i << std::endl;
            prk::CUDA::check( cudaSetDevice(i) );
            //std::cerr << "INFO: nstream args: " << local_length << "," << scalar << "," << d_A.at(i) << "," << d_B.at(i) << "," << d_C.at(i) << std::endl;
            nstream<<<dimGrid, dimBlock>>>(local_length, scalar, d_A.at(i), d_B.at(i), d_C.at(i));
            //std::cerr << "INFO: nstream submitted" << std::endl;
            //qs.wait(i);
            //std::cerr << "INFO: nstream finished" << std::endl;
        }
        //std::cerr << "INFO: nstream finished" << std::endl;
        qs.waitall();
      }
      nstream_time = prk::wtime() - nstream_time;
  }

  qs.gather<double>(h_A, d_A, local_length);
  qs.waitall();

  qs.free(d_A);
  qs.free(d_B);
  qs.free(d_C);
  qs.waitall();

  //////////////////////////////////////////////////////////////////////
  /// Analyze and output results
  //////////////////////////////////////////////////////////////////////

  double ar(0);
  double br(2);
  double cr(2);
  for (int i=0; i<=iterations; i++) {
      ar += br + scalar * cr;
  }

  ar *= length;

  double asum(0);
  for (int i=0; i<length; i++) {
      asum += prk::abs(h_A[i]);
  }

  double epsilon=1.e-8;
  if (prk::abs(ar-asum)/asum > epsilon) {
      std::cout << "Failed Validation on output array\n"
                << std::setprecision(16)
                << "       Expected checksum: " << ar << "\n"
                << "       Observed checksum: " << asum << std::endl;
      std::cout << "ERROR: solution did not validate" << std::endl;
      return 1;
  } else {
      std::cout << "Solution validates" << std::endl;
      double avgtime = nstream_time/iterations;
      double nbytes = 4.0 * length * sizeof(double);
      std::cout << "Rate (MB/s): " << std::fixed << 1.e-6*nbytes/avgtime
                << " Avg time (s): " << std::scientific << avgtime << std::endl;
  }

  return 0;
}


