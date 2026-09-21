cmake_minimum_required(VERSION 3.18)

# In-source builds (`cmake .`) are allowed: the legacy hand-written Makefile
# this would otherwise collide with has been moved to Makefile.legacy.

# Thin convenience wrapper only: every language under here is its own
# independent CMake project (own cmake_minimum_required()/project()/
# enable_testing(), own dependency detection) and can be configured and
# built standalone, e.g. `cd FORTRAN && cmake -S . -B build`. This root
# project just add_subdirectory()s the enabled ones so `cmake -S . -B build`
# from the repo root builds/tests everything in one combined build tree.
# CMake fully supports a nested project() under add_subdirectory().
project(PRK NONE)

enable_testing()

option(PRK_ENABLE_FORTRAN "Build FORTRAN/ kernels" ON)

if(PRK_ENABLE_FORTRAN)
  add_subdirectory(FORTRAN)
endif()

# Future languages follow the same pattern: add an option
# PRK_ENABLE_<LANG> (default ON) and add_subdirectory(<DIR>), with a
# <DIR>/CMakeLists.txt structured like FORTRAN/CMakeLists.txt -- its own
# cmake_minimum_required()/project()/enable_testing(), CMAKE_MODULE_PATH
# pointed at ../cmake/modules via CMAKE_CURRENT_LIST_DIR (not
# CMAKE_SOURCE_DIR, which only means "repo root" when standalone), and the
# same dependency-detection / target-group / CTest pattern.
