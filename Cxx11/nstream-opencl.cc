///
/// Copyright (c) 2020, Intel Corporation
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
///          of iterations to loop over the triad vectors and
///          the length of the vectors.
///
///          <progname> <# iterations> <vector length>
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
#include "prk_opencl.h"

template <typename T>
void run(cl::Context context, int iterations, size_t length)
{
  auto precision = (sizeof(T)==8) ? 64 : 32;

  auto kfile = "nstream"+std::to_string(precision)+".cl";
  cl::Program program(context, prk::opencl::loadProgram(kfile), true);

  cl_int err = CL_SUCCESS;
  try {
    program.build();
  }
  catch (...) {
    auto info = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(&err);
    for (auto &pair : info) {
      std::cout << pair.second << std::endl;
    }
  }
  auto function  = (precision==64) ? "nstream64" : "nstream32";
  auto kernel = cl::KernelFunctor<int, T, cl::Buffer, cl::Buffer, cl::Buffer>(program, function, &err);

  cl::CommandQueue queue(context);

  //////////////////////////////////////////////////////////////////////
  /// Allocate space for the input and nstream matrix
  //////////////////////////////////////////////////////////////////////

  double nstream_time{0};

  std::vector<T> h_a(length, T(0));
  std::vector<T> h_b(length, T(2));
  std::vector<T> h_c(length, T(2));

  // copy input from host to device
  cl::Buffer d_a = cl::Buffer(context, begin(h_a), end(h_a), false);
  cl::Buffer d_b = cl::Buffer(context, begin(h_b), end(h_b), true);
  cl::Buffer d_c = cl::Buffer(context, begin(h_c), end(h_c), true);

  double scalar = 3.0;

  for (int iter = 0; iter<=iterations; iter++) {

    if (iter==1) nstream_time = prk::wtime();

    kernel(cl::EnqueueArgs(queue, cl::NDRange(length)), length, scalar, d_a, d_b, d_c);
    queue.finish();

  }
  nstream_time = prk::wtime() - nstream_time;

  // copy output back to host
  cl::copy(queue, d_a, begin(h_a), end(h_a));

  //////////////////////////////////////////////////////////////////////
  /// Analyze and output results
  //////////////////////////////////////////////////////////////////////

  double ar(0);
  T br(2);
  T cr(2);
  for (int i=0; i<=iterations; i++) {
      ar += br + scalar * cr;
  }

  ar *= length;

  double asum(0);
  for (size_t i=0; i<length; i++) {
      asum += prk::abs(h_a[i]);
  }

  const double epsilon = (precision==64) ? 1.0e-8 : 1.0e-4;
  if (prk::abs(ar-asum)/asum > epsilon) {
      std::cout << "Failed Validation on output array\n"
                << std::setprecision(16)
                << "       Expected checksum: " << ar << "\n"
                << "       Observed checksum: " << asum << std::endl;
      std::cout << "ERROR: solution did not validate" << std::endl;
  } else {
      std::cout << "Solution validates" << std::endl;
      double avgtime = nstream_time/iterations;
      double nbytes = 4.0 * length * sizeof(T);
      std::cout << precision << "B "
                << "Rate (MB/s): " << 1.e-6*nbytes/avgtime
                << " Avg time (s): " << avgtime << std::endl;
  }
}

int main(int argc, char* argv[])
{
  prk::opencl::listPlatforms();

  std::cout << "Parallel Research Kernels" << std::endl;
  std::cout << "C++11/OpenCL STREAM triad: A = B + scalar * C" << std::endl;

  //////////////////////////////////////////////////////////////////////
  /// Read and test input parameters
  //////////////////////////////////////////////////////////////////////

  int iterations;
  size_t length;
  try {
      if (argc < 3) {
        throw "Usage: <# iterations> <vector length>";
      }

      iterations  = std::atoi(argv[1]);
      if (iterations < 1) {
        throw "ERROR: iterations must be >= 1";
      }

      length = std::atol(argv[2]);
      if (length <= 0) {
        throw "ERROR: vector length must be positive";
      }
  }
  catch (const char * e) {
    std::cout << e << std::endl;
    return 1;
  }

  std::cout << "Number of iterations = " << iterations << std::endl;
  std::cout << "Vector length        = " << length << std::endl;

  //////////////////////////////////////////////////////////////////////
  /// Setup OpenCL environment
  //////////////////////////////////////////////////////////////////////

  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  if ( platforms.size() == 0 ) {
    std::cout <<" No platforms found. Check OpenCL installation!\n";
    return 1;
  }
  for (auto plat : platforms) {
    std::cout << "====================================================\n"
              << "CL_PLATFORM_NAME=" << plat.getInfo<CL_PLATFORM_NAME>() << ", "
              << "CL_PLATFORM_VENDOR=" << plat.getInfo<CL_PLATFORM_VENDOR>() << std::endl;

    std::vector<cl::Device> devices;
    plat.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    for (auto dev : devices) {
      std::cout << "CL_DEVICE_NAME="   << dev.getInfo<CL_DEVICE_NAME>()   << ", "
                << "CL_DEVICE_VENDOR=" << dev.getInfo<CL_DEVICE_VENDOR>() << std::endl;

      cl_int err = CL_SUCCESS;
      cl::Context ctx(dev, NULL, NULL, NULL, &err);
      const int precision = prk::opencl::precision(ctx);
      //std::cout << "Device Precision        = " << precision << "-bit" << std::endl;
      if (precision==64) {
          run<double>(dev, iterations, length);
      }
      run<float>(dev, iterations, length);
    }
  }
  std::cout << "====================================================" << std::endl;

  return 0;
}
