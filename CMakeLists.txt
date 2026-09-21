cmake_minimum_required(VERSION 3.16)

# No default languages: each PRK_ENABLE_<LANG> subdirectory enables only the
# language(s) it needs, so e.g. building just Fortran never requires a C++
# compiler to be available.
project(PRK NONE)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/modules")

enable_testing()

option(PRK_ENABLE_FORTRAN "Build FORTRAN/ kernels" ON)

if(PRK_ENABLE_FORTRAN)
  enable_language(Fortran)
  add_subdirectory(FORTRAN)
endif()

# Future languages follow the same pattern: add an option
# PRK_ENABLE_<LANG> (default ON), enable_language(...) if the subdirectory
# needs a language CMake doesn't already know about, then
# add_subdirectory(<DIR>), with a <DIR>/CMakeLists.txt structured like
# FORTRAN/CMakeLists.txt (dependency detection via find_package()/compiler
# ID checks, one target group per Makefile target group, CTest entries
# reproducing ci/build-run-prk.sh's run commands).
