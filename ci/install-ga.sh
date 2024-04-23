#!/bin/sh

set -e
set -x

CI_ROOT="$1"

if [ ! -d "$CI_ROOT/ga" ]; then
    if [ ! -d "$CI_ROOT/ga-src" ]; then
        git clone -b develop https://github.com/GlobalArrays/ga.git $CI_ROOT/ga-src
    else
        cd $CI_ROOT/ga-src && git pull
    fi
    cd $CI_ROOT/ga-src
    ./autogen.sh
    mkdir -p build
    cd build
    #../configure CC=mpicc --prefix=$CI_ROOT/ga
    #../configure --with-mpi3 MPICC=mpicc MPICXX=mpicxx MPIFC=mpifort MPIF77=mpifort --prefix=$CI_ROOT/ga && make -j8 install
    ../configure --with-armci=${CI_ROOT}/armci-mpi MPICC=mpicc MPICXX=mpicxx MPIFC=mpifort MPIF77=mpifort --prefix=$CI_ROOT/ga --without-blas --without-lapack --without-scalapack && make -j8 install
    make
    make install
else
    echo "Global Arrays installed..."
    find $CI_ROOT/ga -name ga.h
fi
