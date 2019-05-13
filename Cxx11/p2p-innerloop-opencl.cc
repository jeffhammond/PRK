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
/// NAME:    Pipeline
///
/// PURPOSE: This program tests the efficiency with which point-to-point
///          synchronization can be carried out. It does so by executing
///          a pipelined algorithm on an n^2 grid. The first array dimension
///          is distributed among the threads (stripwise decomposition).
///
/// USAGE:   The program takes as input the
///          dimensions of the grid, and the number of iterations on the grid
///
///                <progname> <iterations> <n>
///
///          The output consists of diagnostics to make sure the
///          algorithm worked, and of timing statistics.
///
/// FUNCTIONS CALLED:
///
///          Other than standard C functions, the following
///          functions are used in this program:
///
///          wtime()
///
/// HISTORY: - Written by Rob Van der Wijngaart, February 2009.
///            C99-ification by Jeff Hammond, February 2016.
///            C++11-ification by Jeff Hammond, May 2017.
///
//////////////////////////////////////////////////////////////////////

#include "prk_util.h"
#include "prk_opencl.h"

template <typename T>
void run(cl::Context context, int iterations, int n, bool consolidated)
{
  const int precision = (sizeof(T)==8) ? 64 : 32;

  cl::Program program(context, prk::opencl::loadProgram("p2p.cl"), true);

  std::string function = (precision==64) ? "p2p64" : "p2p32";

  cl_int err;
  auto kernel = cl::make_kernel<int, cl::Buffer>(program, function, &err);
  if(err != CL_SUCCESS){
    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
  }

  auto kerneli = cl::make_kernel<int, int, cl::Buffer>(program, function+"i", &err);
  if(err != CL_SUCCESS){
    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
  }

  auto kernelf = cl::make_kernel<int, cl::Buffer>(program, function+"f", &err);
  if(err != CL_SUCCESS){
    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
  }

  auto info = cl::make_kernel<int, cl::Buffer>(program, "info", &err);
  if(err != CL_SUCCESS){
    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
  }

  cl::CommandQueue queue(context);

  //////////////////////////////////////////////////////////////////////
  /// Allocate space for the input and transpose matrix
  //////////////////////////////////////////////////////////////////////

  std::vector<T> h_grid(n*n, T(0));
  for (int j=0; j<n; j++) {
    h_grid[0*n+j] = static_cast<double>(j);
  }
  for (int i=0; i<n; i++) {
    h_grid[i*n+0] = static_cast<double>(i);
  }

  // copy input from host to device
  cl::Buffer d_grid = cl::Buffer(context, begin(h_grid), end(h_grid), false);

  double pipeline_time(0);

  size_t range = 2*n;
  info(cl::EnqueueArgs(queue, cl::NDRange(range)), n, d_grid);
  queue.finish();

  for (int iter = 0; iter<=iterations; iter++) {

    if (iter==1) pipeline_time = prk::wtime();

    if (consolidated) {
        kernel(cl::EnqueueArgs(queue, cl::NDRange(range)), n, d_grid);
    } else {
        for (int i=2; i<=2*n-2; i++) {
          kerneli(cl::EnqueueArgs(queue, cl::NDRange(range)), i, n, d_grid);
        }
        kernelf(cl::EnqueueArgs(queue, cl::NDRange(1)), n, d_grid);
    }
  }

  // from device to host
  cl::copy(queue,d_grid, begin(h_grid), end(h_grid));
  queue.finish();

  pipeline_time = prk::wtime() - pipeline_time;

  //cl::copy(d_grid, begin(h_grid), end(h_grid));

  //////////////////////////////////////////////////////////////////////
  // Analyze and output results.
  //////////////////////////////////////////////////////////////////////

  // error tolerance
  const T epsilon = (sizeof(T)==8) ? 1.e-8 : 1.e-4f;

  // verify correctness, using top right value
  T corner_val = ((iterations+1)*(2*n-2));
  if ( (std::fabs(h_grid[(n-1)*n+(n-1)] - corner_val)/corner_val) > epsilon) {
    std::cout << "ERROR: checksum " << h_grid[(n-1)*n+(n-1)]
              << " does not match verification value " << corner_val << std::endl;
    //return;
  }

#ifdef VERBOSE
  std::cout << "Solution validates; verification value = " << corner_val << std::endl;
#else
  std::cout << "Solution validates" << std::endl;
#endif
  auto avgtime = pipeline_time/iterations;
  std::cout << "Rate (MFlops/s): "
            << 2.0e-6 * ( (n-1.)*(n-1.) )/avgtime
            << " Avg time (s): " << avgtime << std::endl;
}

int main(int argc, char* argv[])
{
  std::cout << "Parallel Research Kernels version " << PRKVERSION << std::endl;
  std::cout << "C++11/OpenCL INNERLOOP pipeline execution on 2D grid" << std::endl;

  prk::opencl::listPlatforms();

  //////////////////////////////////////////////////////////////////////
  // Process and test input parameters
  //////////////////////////////////////////////////////////////////////

  int iterations;
  int n;
  bool consolidated;
  try {
      if (argc < 3) {
        throw " <# iterations> <array dimension> [<consolidated>]";
      }

      // number of times to run the pipeline algorithm
      iterations  = std::atoi(argv[1]);
      if (iterations < 1) {
        throw "ERROR: iterations must be >= 1";
      }

      // grid dimensions
      n = std::atoi(argv[2]);
      if (n < 1) {
        throw "ERROR: grid dimensions must be positive";
      } else if ( static_cast<size_t>(n)*static_cast<size_t>(n) > INT_MAX) {
        throw "ERROR: grid dimension too large - overflow risk";
      }

      // consolidate antidiagonal sweeps into one kernel
      if (argc > 3) {
          consolidated = std::atoi(argv[3]);
      } else {
          consolidated = true;
      }
  }
  catch (const char * e) {
    std::cout << e << std::endl;
    return 1;
  }

  std::cout << "Number of iterations = " << iterations << std::endl;
  std::cout << "Grid sizes           = " << n << ", " << n << std::endl;
  std::cout << "Consolidated kernel  = " << consolidated << std::endl;

  //////////////////////////////////////////////////////////////////////
  /// Setup OpenCL environment
  //////////////////////////////////////////////////////////////////////

  cl_int err = CL_SUCCESS;

  cl::Context cpu(CL_DEVICE_TYPE_CPU, NULL, NULL, NULL, &err);
  if ( err == CL_SUCCESS && prk::opencl::available(cpu) )
  {
    const int precision = prk::opencl::precision(cpu);

    std::cout << "CPU Precision        = " << precision << "-bit" << std::endl;

    if (0 && precision==64) {
        run<double>(cpu, iterations, n, consolidated);
    } else {
        run<float>(cpu, iterations, n, consolidated);
    }
  }

  cl::Context gpu(CL_DEVICE_TYPE_GPU, NULL, NULL, NULL, &err);
  if ( err == CL_SUCCESS && prk::opencl::available(gpu) )
  {
    const int precision = prk::opencl::precision(gpu);

    std::cout << "GPU Precision        = " << precision << "-bit" << std::endl;

    if (0 && precision==64) {
        run<double>(gpu, iterations, n, consolidated);
    } else {
        run<float>(gpu, iterations, n, consolidated);
    }
  }

  cl::Context acc(CL_DEVICE_TYPE_ACCELERATOR, NULL, NULL, NULL, &err);
  if ( err == CL_SUCCESS && prk::opencl::available(acc) )
  {

    const int precision = prk::opencl::precision(acc);

    std::cout << "ACC Precision        = " << precision << "-bit" << std::endl;

    if (precision==64) {
        run<double>(acc, iterations, n, consolidated);
    } else {
        run<float>(acc, iterations, n, consolidated);
    }
  }

  return 0;
}
