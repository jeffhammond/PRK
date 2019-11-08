///
/// Copyright (c) 2018, Intel Corporation
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

#ifndef PRK_RAJA_H
#define PRK_RAJA_H

//#ifdef USE_RAJA
//# define RAJA_ENABLE_NESTED 1
# include "RAJA/RAJA.hpp"
//#endif

#if defined(RAJA_ENABLE_OPENMP)
  typedef RAJA::omp_parallel_for_exec thread_exec;
#elif defined(RAJA_ENABLE_OPENMP_TARGET)
  typedef RAJA::omp_target_parallel_for_exec<64> thread_exec;
#elif defined(RAJA_ENABLE_TBB)
  typedef RAJA::tbb_for_exec thread_exec;
#elif defined(RAJA_ENABLE_CUDA)
  const int CUDA_BLOCK_SIZE = 256;
  typedef RAJA::cuda_exec<CUDA_BLOCK_SIZE> thread_exec;
#else
#warning No OpenMP!
  typedef RAJA::seq_exec thread_exec;
#endif

#ifdef RAJA_ENABLE_OPENMP
  typedef RAJA::omp_reduce reduce_exec;
#else
  //#warning No RAJA support for OpenMP!
  typedef RAJA::seq_reduce reduce_exec;
#endif

#if defined(RAJA_ENABLE_CUDA)
#define RAJA_LAMBDA [=] RAJA_DEVICE
using regular_policy = RAJA::KernelPolicy< RAJA::statement::For<0, RAJA::cuda_exec<CUDA_BLOCK_SIZE>,
                                           RAJA::statement::For<1, RAJA::cuda_exec<CUDA_BLOCK_SIZE>,
                                           RAJA::statement::Lambda<0> > > >;
using permute_policy = RAJA::KernelPolicy< RAJA::statement::For<1, RAJA::cuda_exec<CUDA_BLOCK_SIZE>,
                                           RAJA::statement::For<0, RAJA::cuda_exec<CUDA_BLOCK_SIZE>,
                                           RAJA::statement::Lambda<0> > > >;
#else
#define RAJA_LAMBDA [=]
using regular_policy = RAJA::KernelPolicy< RAJA::statement::For<0, thread_exec,
                                           RAJA::statement::For<1, RAJA::simd_exec,
                                           RAJA::statement::Lambda<0> > > >;
using permute_policy = RAJA::KernelPolicy< RAJA::statement::For<1, thread_exec,
                                           RAJA::statement::For<0, RAJA::simd_exec,
                                           RAJA::statement::Lambda<0> > > >;
#endif

typedef RAJA::View<double, RAJA::Layout<2>> matrix;

#endif /* PRK_RAJA_H */
