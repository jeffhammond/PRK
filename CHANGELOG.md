# Changelog

## Unreleased

### Documentation

**README.md — feature table corrections and additions**

*Modern C++ table:*
- Fixed `OpenACC` row: stencil, transpose, and nstream were missing (`stencil-openacc.cc`, `transpose-openacc.cc`, `nstream-openacc.cc`)
- Fixed `MPI (RMA)` row: added stencil, transpose, and dgemm (`stencil-mpi.cc`, `transpose-{a2a,get,p2p}-mpi.cc`, `dgemm-mpi-{cblas,cublas}.cc`)
- Added rows for backends present in source but missing from the table: `stdpar`, `CUDAStF`, `C++ Ranges`, `NVSHMEM`, `NCCL`, `OCCA`, `HPX`, `oneDPL`, `UPC++`
- Added note that single-precision (`sgemm`) and mixed-precision (`xgemm`) GEMM variants exist for CBLAS, CUBLAS, HIPBLAS, and oneMKL backends

*Modern Fortran table:*
- Added `PIC` column (`pic.F90`, `pic-openmp.F90`, `pic_soa.F90`, `pic_soa-openmp.F90`)
- Fixed `Global Arrays` row: added dgemm (`dgemm-ga.F90`)
- Fixed `OpenMP tasks` row: added dgemm (`dgemm-taskloop-openmp.F90`)
- Fixed `OpenMP target` row: added dgemm (`dgemm-openmp-target.F90`)
- Fixed `OpenACC` row: added p2p (`p2p-openacc.F90`)
- Added rows for `stdpar` (stencil, transpose, nstream, dgemm), `CUDA Fortran` (transpose, nstream), and `MPI` (transpose, nstream)

*Other languages table:*
- Fixed `Python 3 w/ mpi4py` row: added p2p (`p2p-numpy-mpi.py`)
- Added `Python 3 w/ Numba` row (p2p, stencil, transpose, nstream)
- Added `Python 3 w/ CuPy` row (stencil, transpose, nstream)
- Added `Python 3 w/ OpenSHMEM` row (p2p, stencil, transpose, nstream)

**GettingStarted.md — accuracy and usability improvements**

- Replaced stale Travis CI reference with the `ci/` directory; fixed typo "undestand" → "understand"
- Added GPU toolchain guidance: `make.defs.nvhpc` (NVIDIA HPC SDK), `make.defs.cuda` (GCC+NVCC), `make.defs.hip` (AMD ROCm), `make.defs.oneapi`/`make.defs.llvm` (Intel GPU/SYCL)
- Fixed `make -j$(nproc)` to be cross-platform: `make -j$(nproc 2>/dev/null || sysctl -n hw.logicalcpu)` (works on Linux and macOS)
- Added explanation that the `-k` flag is intentional — not all backends are expected to be available
- Added `scripts/small/runall` and `scripts/wide/runall` examples to the quick-start section
