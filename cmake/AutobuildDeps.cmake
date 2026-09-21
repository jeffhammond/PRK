#[=======================================================================[.rst:
AutobuildDeps
-------------

Optional dependencies get built from source via ExternalProject_Add instead
of just being silently skipped when the system doesn't have them.
OpenCoarrays/ARMCI-MPI/Global Arrays mirror what PRK's own CI already does
by hand (see ci/install-opencoarrays.sh, ci/install-armci-mpi.sh,
ci/install-ga.sh); PETSc has no such CI precedent but gets the same
treatment for consistency. Controlled by PRK_AUTOBUILD_DEPS (default ON);
each prk_autobuild_*() function is a no-op if the dependency was already
found on the system, or if this option is off.

This same ExternalProject_Add pattern (ci/install-*.sh script -> CMake
external project, with BUILD_BYPRODUCTS pointing at the not-yet-built
library/executable paths so the dependency graph is still correct) is the
one to reuse for Kokkos and RAJA once Cxx11 gets CMake support -- both are
CMake-native projects, so they can use a plain CMAKE_ARGS configure step
like prk_autobuild_opencoarrays() below, rather than the autotools
CONFIGURE_COMMAND dance prk_autobuild_global_arrays() needs.
#]=======================================================================]

option(PRK_AUTOBUILD_DEPS
  "Fetch and build missing optional dependencies (OpenCoarrays, Global Arrays, PETSc, ...) from source"
  ON)

include(ExternalProject)

function(prk_autobuild_opencoarrays)
  if(OpenCoarrays_FOUND OR NOT PRK_AUTOBUILD_DEPS)
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
  if(GlobalArrays_FOUND OR NOT PRK_AUTOBUILD_DEPS)
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
  find_program(_prk_autoreconf autoreconf)
  if(NOT _prk_autoreconf)
    message(STATUS "Global Arrays: not found, and cannot autobuild without autoreconf on PATH")
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
  # ci/install-ga.sh's configure --with-armci=... && make install, but NOT
  # its autogen.sh: GA's own autogen.sh runs travis/install-autotools.sh,
  # which builds a private m4/autoconf/automake/libtool from source and
  # downloads config.guess/config.sub from git.savannah.gnu.org -- slow and
  # a hard failure with no network access. autoreconf -vif with the system's
  # already-installed autotools does the same job (it recurses into GA's
  # armci/ and comex/ AC_CONFIG_SUBDIRS automatically) without any of that.
  set(_ga_prefix "${CMAKE_BINARY_DIR}/_deps/ga")
  ExternalProject_Add(global_arrays_external
    DEPENDS           armci_mpi_external
    GIT_REPOSITORY    https://github.com/GlobalArrays/ga.git
    GIT_TAG           develop
    PREFIX            "${_ga_prefix}"
    BUILD_IN_SOURCE   TRUE
    CONFIGURE_COMMAND ${_prk_autoreconf} -vif
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

function(prk_autobuild_petsc)
  if(PRK_PETSC_FOUND OR NOT PRK_AUTOBUILD_DEPS)
    return()
  endif()
  if(NOT MPI_C_FOUND)
    message(STATUS "PETSc: not found, and cannot autobuild without MPI C")
    return()
  endif()
  find_program(_prk_mpicc mpicc)
  if(NOT _prk_mpicc)
    message(STATUS "PETSc: not found, and cannot autobuild without mpicc on PATH")
    return()
  endif()
  find_program(_prk_python3 python3)
  if(NOT _prk_python3)
    message(STATUS "PETSc: not found, and cannot autobuild without python3 on PATH (needed by PETSc's configure)")
    return()
  endif()

  message(STATUS "PETSc: not found on system, fetching and building from source (GitHub: petsc/petsc) -- this is a large build and can take a while")

  # PETSc's own configure (a Python script, not autotools) only generates
  # build files; it does not build or install by itself, so those are
  # separate ExternalProject steps. --with-debugging=0 and skipping the
  # Fortran/C++ bindings and external packages keeps this to a plain C
  # library build, which is all PRK's *-petsc.c kernels need.
  set(_petsc_prefix "${CMAKE_BINARY_DIR}/_deps/petsc")
  ExternalProject_Add(petsc_external
    GIT_REPOSITORY    https://github.com/petsc/petsc.git
    GIT_TAG           release
    GIT_SHALLOW       TRUE
    PREFIX            "${_petsc_prefix}"
    BUILD_IN_SOURCE   TRUE
    CONFIGURE_COMMAND <SOURCE_DIR>/configure
                        --with-cc=${_prk_mpicc}
                        --with-cxx=0
                        --with-fc=0
                        --with-debugging=0
                        --download-fblaslapack=0
                        --prefix=<INSTALL_DIR>
    BUILD_COMMAND     make
    INSTALL_COMMAND   make install
    BUILD_BYPRODUCTS  <INSTALL_DIR>/lib/libpetsc.so
  )
  ExternalProject_Get_property(petsc_external INSTALL_DIR)

  # CMake validates IMPORTED target INTERFACE_INCLUDE_DIRECTORIES exist at
  # generate time, even though this directory is only populated once
  # petsc_external actually builds.
  file(MAKE_DIRECTORY "${INSTALL_DIR}/include")

  set(PRK_PETSC_INCLUDE_DIR "${INSTALL_DIR}/include" CACHE PATH "" FORCE)
  set(PRK_PETSC_LIBRARY "${INSTALL_DIR}/lib/libpetsc.so" CACHE FILEPATH "" FORCE)
  set(PRK_PETSC_FOUND TRUE PARENT_SCOPE)
  set(PRK_PETSC_AUTOBUILT TRUE PARENT_SCOPE)
  set(PRK_PETSC_EXTERNAL_TARGET petsc_external PARENT_SCOPE)
endfunction()
