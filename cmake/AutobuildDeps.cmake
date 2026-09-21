#[=======================================================================[.rst:
AutobuildDeps
-------------

Optional dependencies that PRK's own CI builds from source when the system
doesn't have them (see ci/install-opencoarrays.sh, ci/install-armci-mpi.sh,
ci/install-ga.sh) get the same treatment here via ExternalProject_Add,
instead of just being silently skipped. Controlled by
PRK_FORTRAN_AUTOBUILD_DEPS (default ON); each prk_autobuild_*() function is a
no-op if the dependency was already found on the system, or if this option
is off.

This same ExternalProject_Add pattern (ci/install-*.sh script -> CMake
external project, with BUILD_BYPRODUCTS pointing at the not-yet-built
library/executable paths so the dependency graph is still correct) is the
one to reuse for Kokkos and RAJA once Cxx11 gets CMake support -- both are
CMake-native projects, so they can use a plain CMAKE_ARGS configure step
like prk_autobuild_opencoarrays() below, rather than the autotools
CONFIGURE_COMMAND dance prk_autobuild_global_arrays() needs.
#]=======================================================================]

option(PRK_FORTRAN_AUTOBUILD_DEPS
  "Fetch and build missing optional Fortran dependencies (OpenCoarrays, Global Arrays) from source"
  ON)

include(ExternalProject)

function(prk_autobuild_opencoarrays)
  if(OpenCoarrays_FOUND OR NOT PRK_FORTRAN_AUTOBUILD_DEPS)
    return()
  endif()
  if(NOT (MPI_C_FOUND AND MPI_Fortran_FOUND))
    message(STATUS "OpenCoarrays: not found, and cannot autobuild without MPI C/Fortran")
    return()
  endif()

  message(STATUS "OpenCoarrays: not found on system, fetching and building from source (GitHub: sourceryinstitute/opencoarrays)")

  set(_prefix "${CMAKE_BINARY_DIR}/_deps/opencoarrays")
  ExternalProject_Add(opencoarrays_external
    GIT_REPOSITORY    https://github.com/sourceryinstitute/opencoarrays.git
    GIT_TAG           2.10.2
    GIT_SHALLOW       TRUE
    PREFIX            "${_prefix}"
    CMAKE_ARGS        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
                       -DMPI_C_COMPILER=${MPI_C_COMPILER}
                       -DMPI_Fortran_COMPILER=${MPI_Fortran_COMPILER}
                       -DCMAKE_BUILD_TYPE=Release
    BUILD_BYPRODUCTS  <INSTALL_DIR>/bin/caf
                       <INSTALL_DIR>/bin/cafrun
                       <INSTALL_DIR>/lib${CMAKE_INSTALL_LIBDIR_SUFFIX}/libcaf_mpi${CMAKE_SHARED_LIBRARY_SUFFIX}
  )
  ExternalProject_Get_property(opencoarrays_external INSTALL_DIR)

  set(OpenCoarrays_CAF "${INSTALL_DIR}/bin/caf" CACHE FILEPATH "" FORCE)
  set(OpenCoarrays_CAFRUN "${INSTALL_DIR}/bin/cafrun" CACHE FILEPATH "" FORCE)
  set(OpenCoarrays_LIBRARY "${INSTALL_DIR}/lib/libcaf_mpi${CMAKE_SHARED_LIBRARY_SUFFIX}" CACHE FILEPATH "" FORCE)
  set(OpenCoarrays_FOUND TRUE PARENT_SCOPE)
  set(OpenCoarrays_AUTOBUILT TRUE PARENT_SCOPE)
  set(OpenCoarrays_EXTERNAL_TARGET opencoarrays_external PARENT_SCOPE)
endfunction()

function(prk_autobuild_global_arrays)
  if(GlobalArrays_FOUND OR NOT PRK_FORTRAN_AUTOBUILD_DEPS)
    return()
  endif()
  if(NOT MPI_Fortran_FOUND)
    message(STATUS "Global Arrays: not found, and cannot autobuild without MPI Fortran")
    return()
  endif()
  find_program(_prk_mpicc mpicc)
  find_program(_prk_mpif90 mpif90)
  if(NOT (_prk_mpicc AND _prk_mpif90))
    message(STATUS "Global Arrays: not found, and cannot autobuild without mpicc/mpif90 on PATH")
    return()
  endif()

  message(STATUS "Global Arrays: not found on system, fetching and building ARMCI-MPI + GA from source (GitHub: jeffhammond/armci-mpi, GlobalArrays/ga)")

  # ARMCI-MPI: GA's transport layer. Autotools, so this mirrors
  # ci/install-armci-mpi.sh's autogen.sh && configure && make install
  # exactly, just run inside the CMake build instead of a shell script.
  set(_armci_prefix "${CMAKE_BINARY_DIR}/_deps/armci-mpi")
  ExternalProject_Add(armci_mpi_external
    GIT_REPOSITORY    https://github.com/jeffhammond/armci-mpi.git
    GIT_TAG           master
    PREFIX            "${_armci_prefix}"
    BUILD_IN_SOURCE   TRUE
    CONFIGURE_COMMAND <SOURCE_DIR>/autogen.sh
              COMMAND <SOURCE_DIR>/configure CC=${_prk_mpicc} --prefix=<INSTALL_DIR>
    BUILD_COMMAND     make
    INSTALL_COMMAND   make install
    BUILD_BYPRODUCTS  <INSTALL_DIR>/lib/libarmci.a
  )
  ExternalProject_Get_property(armci_mpi_external INSTALL_DIR)
  set(_armci_install_dir "${INSTALL_DIR}")

  # GA itself, linked against the ARMCI-MPI we just built. Mirrors
  # ci/install-ga.sh: autogen.sh && configure --with-armci=... && make install.
  set(_ga_prefix "${CMAKE_BINARY_DIR}/_deps/ga")
  ExternalProject_Add(global_arrays_external
    DEPENDS           armci_mpi_external
    GIT_REPOSITORY    https://github.com/GlobalArrays/ga.git
    GIT_TAG           develop
    PREFIX            "${_ga_prefix}"
    BUILD_IN_SOURCE   TRUE
    CONFIGURE_COMMAND <SOURCE_DIR>/autogen.sh
              COMMAND <SOURCE_DIR>/configure --with-armci=${_armci_install_dir}
                        MPICC=${_prk_mpicc} MPIFC=${_prk_mpif90} MPIF77=${_prk_mpif90}
                        --prefix=<INSTALL_DIR>
    BUILD_COMMAND     make
    INSTALL_COMMAND   make install
    BUILD_BYPRODUCTS  <INSTALL_DIR>/lib/libga.a
  )
  ExternalProject_Get_property(global_arrays_external INSTALL_DIR)

  # CMake validates IMPORTED target INTERFACE_INCLUDE_DIRECTORIES exist at
  # generate time, even though this directory is only populated once
  # global_arrays_external actually builds -- so it must exist now.
  file(MAKE_DIRECTORY "${INSTALL_DIR}/include")

  set(GlobalArrays_INCLUDE_DIR "${INSTALL_DIR}/include" CACHE PATH "" FORCE)
  set(GlobalArrays_LIBRARY "${INSTALL_DIR}/lib/libga.a" CACHE FILEPATH "" FORCE)
  set(GlobalArrays_ARMCI_LIBRARY "${_armci_install_dir}/lib/libarmci.a" CACHE FILEPATH "" FORCE)
  set(GlobalArrays_FOUND TRUE PARENT_SCOPE)
  set(GlobalArrays_AUTOBUILT TRUE PARENT_SCOPE)
  set(GlobalArrays_EXTERNAL_TARGET global_arrays_external PARENT_SCOPE)
endfunction()
