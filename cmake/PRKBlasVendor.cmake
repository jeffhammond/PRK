# Prefer MKL when one is active (MKLROOT set, e.g. by oneAPI's setvars.sh or
# cmake/toolchains/oneapi.cmake) and the user didn't choose a BLAS backend.
#
# Plain find_package(BLAS) is unreliable here: with ifx it settles on a system
# netlib libblas ahead of MKL, and setting BLA_VENDOR=Intel10_* makes FindBLAS
# fail outright for Fortran-only projects (MKLROOT is typically
# .../mkl/<version>, not .../mkl, so FindBLAS misses the compiler-dir
# libiomp5 and its probe fails). Resolve the MKL libraries ourselves and hand
# them to FindBLAS as BLAS_LIBRARIES, which it then just validates.
if(DEFINED ENV{MKLROOT} AND NOT BLAS_LIBRARIES AND NOT BLA_VENDOR)
  set(_prk_mkl_hints "$ENV{MKLROOT}/lib" "$ENV{MKLROOT}/lib/intel64")
  find_library(PRK_MKL_INTEL_LP64 mkl_intel_lp64 HINTS ${_prk_mkl_hints})
  find_library(PRK_MKL_INTEL_THREAD mkl_intel_thread HINTS ${_prk_mkl_hints})
  find_library(PRK_MKL_CORE mkl_core HINTS ${_prk_mkl_hints})
  set(_prk_omp_hints "${_prk_mkl_hints}")
  if(DEFINED ENV{CMPLR_ROOT})
    list(APPEND _prk_omp_hints "$ENV{CMPLR_ROOT}/lib")
  endif()
  file(GLOB _prk_oneapi_compiler_libs "$ENV{MKLROOT}/../../compiler/latest/lib")
  list(APPEND _prk_omp_hints ${_prk_oneapi_compiler_libs})
  find_library(PRK_MKL_IOMP5 iomp5 HINTS ${_prk_omp_hints})
  if(PRK_MKL_INTEL_LP64 AND PRK_MKL_INTEL_THREAD AND PRK_MKL_CORE AND PRK_MKL_IOMP5)
    set(BLAS_LIBRARIES "${PRK_MKL_INTEL_LP64};${PRK_MKL_INTEL_THREAD};${PRK_MKL_CORE};${PRK_MKL_IOMP5}" CACHE STRING "MKL (from MKLROOT)")
    # MKL also provides LAPACK; preset it too, or CMake 3.21's FindLAPACK
    # misparses the BLAS list ("-SSL2", trailing-whitespace link items).
    set(LAPACK_LIBRARIES "${BLAS_LIBRARIES}" CACHE STRING "MKL (from MKLROOT)")
    message(STATUS "BLAS: using MKL from MKLROOT=$ENV{MKLROOT}")
  endif()
endif()
