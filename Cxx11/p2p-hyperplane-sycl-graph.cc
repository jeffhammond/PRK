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
///	           SYCL Graph version by Pablo Reble, Dec 2024.
///
//////////////////////////////////////////////////////////////////////

#include "prk_sycl.h"
#include "prk_util.h"
#include "p2p-kernel.h"

#include <sycl/ext/oneapi/experimental/graph.hpp>

class sweep;
class corner;

int main(int argc, char* argv[])
{
  std::cout << "Parallel Research Kernels" << std::endl;
  std::cout << "C++11/SYCL HYPERPLANE pipeline execution on 2D grid" << std::endl;

  //////////////////////////////////////////////////////////////////////
  // Process and test input parameters
  //////////////////////////////////////////////////////////////////////

  int iterations;
  int n;
  try {
      if (argc < 3) {
        throw " <# iterations> <array dimension> [<chunk dimension>]";
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
      } else if ( n > prk::get_max_matrix_size() ) {
        throw "ERROR: grid dimension too large - overflow risk";
      }

  }
  catch (const char * e) {
    std::cout << e << std::endl;
    return 1;
  }

  std::cout << "Number of iterations = " << iterations << std::endl;
  std::cout << "Grid sizes           = " << n << ", " << n << std::endl;

  //////////////////////////////////////////////////////////////////////
  // Allocate space and perform the computation
  //////////////////////////////////////////////////////////////////////

  double pipeline_time{0}; // silence compiler warning

  std::vector<double> h_grid(n*n,0.0);
  for (int j=0; j<n; j++) {
    h_grid[0*n+j] = static_cast<double>(j);
    h_grid[j*n+0] = static_cast<double>(j);
  }

  sycl::queue q;

  sycl::buffer<double> d_grid { h_grid.data(), h_grid.size() };
  d_grid.set_write_back(false);
    
  {
    sycl::ext::oneapi::experimental::command_graph g{q, {sycl::ext::oneapi::experimental::property::graph::assume_buffer_outlives_graph{}}};
    // start recording into a graph
    g.begin_recording(q);
    {
      for (int i=2; i<=2*n-2; i++) {

        sycl::id<1> I{unsigned(i)};
        sycl::id<1> One{1};

        q.submit([&](sycl::handler& h) {

          sycl::accessor grid(d_grid, h);

          unsigned begin = std::max(2,i-n+2);
          unsigned end   = std::min(i,n)+1;
          unsigned range = end-begin;

          h.parallel_for<class sweep>(sycl::range<1>{range}, [=] (sycl::item<1> j) {
            auto J = begin + j.get_id();
            sycl::id<1> N{unsigned(n)};
            sycl::id<1> X{I-J+One};
            sycl::id<1> Y{J-One};
            sycl::id<1> Xold{X-One}; // x-1
            sycl::id<1> Yold{Y-One}; // y-1
            sycl::id<1> index0{X*N+Y};
            sycl::id<1> index1{Xold*N+Y};
            sycl::id<1> index2{X*N+Yold};
            sycl::id<1> index3{Xold*N+Yold};
            grid[index0] = grid[index1] + grid[index2] - grid[index3];
          });
        });
      }
      q.submit([&](sycl::handler& h) {

        sycl::accessor grid(d_grid, h);

        h.single_task<class corner>([=] {
            grid[0*n+0] = -grid[(n-1)*n+(n-1)];
        });
      });
    }
    g.end_recording();

  auto execGraph = g.finalize();

  for (int iter = 0; iter<=iterations; iter++) {
    if (iter==1) pipeline_time = prk::wtime();
    q.ext_oneapi_graph(execGraph).wait();
  }

  pipeline_time = prk::wtime() - pipeline_time;

#ifdef VERBOSE
  g.print_graph("p2p-hypergraph.dot");
#endif

  }


  sycl::host_accessor HostAccA(d_grid);

  //////////////////////////////////////////////////////////////////////
  // Analyze and output results.
  //////////////////////////////////////////////////////////////////////

  const double epsilon = 1.e-8;
  auto corner_val = ((iterations+1.)*(2.*n-2.));
  if ( (prk::abs(HostAccA[(n-1)*n+(n-1)] - corner_val)/corner_val) > epsilon) {
    std::cout << "ERROR: checksum " << HostAccA[(n-1)*n+(n-1)]
              << " does not match verification value " << corner_val << std::endl;
    return 1;
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

  return 0;
}
