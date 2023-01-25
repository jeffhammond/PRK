#define JEFF 1
!
! Copyright (c) 2017, Intel Corporation
! Copyright (c) 2021, NVIDIA
!
! Redistribution and use in source and binary forms, with or without
! modification, are permitted provided that the following conditions
! are met:
!
! * Redistributions of source code must retain the above copyright
!      notice, this list of conditions and the following disclaimer.
! * Redistributions in binary form must reproduce the above
!      copyright notice, this list of conditions and the following
!      disclaimer in the documentation and/or other materials provided
!      with the distribution.
! * Neither the name of Intel Corporation nor the names of its
!      contributors may be used to endorse or promote products
!      derived from this software without specific prior written
!      permission.
!
! THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
! "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
! LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
! FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
! COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
! INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
! BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
! LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
! CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
! LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
! ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
! POSSIBILITY OF SUCH DAMAGE.

!*******************************************************************
!
! NAME:    nstream
!
! PURPOSE: To compute memory bandwidth when adding a vector of a given
!          number of double precision values to the scalar multiple of
!          another vector of the same length, and storing the result in
!          a third vector.
!
! USAGE:   The program takes as input the number
!          of iterations to loop over the triad vectors, the length of the
!          vectors, and the block_size for the GPU.
!
!          <progname> <# iterations> <vector length> <block_size>
!
!          The output consists of diagnostics to make sure the
!          algorithm worked, and of timing statistics.
!
! NOTES:   Bandwidth is determined as the number of words read, plus the
!          number of words written, times the size of the words, divided
!          by the execution time. For a vector length of N, the total
!          number of words read and written is 4*N*sizeof(double).
!
!
! HISTORY: This code is loosely based on the Stream benchmark by John
!          McCalpin, but does not follow all the Stream rules. Hence,
!          reported results should not be associated with Stream in
!          external publications
!
!          Converted to C++11 by Jeff Hammond, November May 2017.
!
! *******************************************************************

module nstream
  use iso_fortran_env

#if JEFF
  interface
    attributes(global) subroutine fake(scalar, A, B, C, DA, DB, DC) &
                                  bind(C,name="actual")
      use iso_fortran_env
      implicit none
      real(kind=REAL64), intent(in), value :: scalar
      real(kind=REAL64), intent(inout) :: A(*)
      real(kind=REAL64), intent(in) :: B(*), C(*)
      integer(8), dimension(*), intent(in) :: DA, DB, DC
    end subroutine fake
  end interface
#endif

  contains

    attributes(global) subroutine kernel(n, scalar, A, B, C)
      implicit none
      integer(kind=INT64), intent(in), value :: n
      real(kind=REAL64), intent(in), value :: scalar
      real(kind=REAL64), intent(inout) :: A(n)
      real(kind=REAL64), intent(in) :: B(n), C(n)
      integer :: i
      i = blockDim%x * (blockIdx%x - 1) + threadIdx%x
      if (i <= n) then
          A(i) = A(i) + B(i) + scalar * C(i)
      endif
    end subroutine kernel

#if JEFF
    attributes(global) subroutine actual(scalar, A, B, C) &
                                  bind(C,name="actual")
      implicit none
      real(kind=REAL64), intent(in), value :: scalar
      real(kind=REAL64), intent(inout) :: A(:)
      real(kind=REAL64), intent(in) :: B(:), C(:)
      integer :: i
      i = blockDim%x * (blockIdx%x - 1) + threadIdx%x
      if (i <= size(A)) then
          A(i) = A(i) + B(i) + scalar * C(i)
      endif
    end subroutine actual

    subroutine init_descriptor(n,D)
      implicit none
      integer(kind=INT64), intent(in) :: n
      integer(8), intent(inout) :: D(16)
      D(1)  = 35           ! tag (version)
      D(2)  =  1           ! rank
      D(3)  = 28           ! kind
      D(4)  =  4           ! len
      D(5)  =  0           ! flags
      D(6)  =  n           ! lsize
      D(7)  =  D(6)        ! gsize
      D(8)  =  0           ! lbase
      D(9)  =  0           ! gbase
      D(10) =  0           ! unused
      D(11) =  1           ! dim[0].lbound
      D(12) =  D(6)        ! dim[0].extent
      D(13) =  0           ! dim[0].sstride
      D(14) =  0           ! dim[0].soffset
      D(15) =  1           ! dim[0].lstride
      D(16) =  D(11)+D(12) ! dim[0].ubound
    end subroutine init_descriptor
#endif

end module nstream

program main
  use iso_fortran_env
  use cudafor
  use nstream
  use prk
  implicit none
  integer :: err
  ! problem definition
  integer(kind=INT32) :: iterations, block_size
  integer(kind=INT64) :: length, offset
  real(kind=REAL64), allocatable, managed :: A(:), B(:), C(:)
#if JEFF
  integer(8), dimension(16), managed :: DA, DB, DC
#endif
  real(kind=REAL64) :: scalar
  integer(kind=INT64) :: bytes
  ! runtime variables
  integer(kind=INT64) :: i
  integer(kind=INT32) :: k
  real(kind=REAL64) ::  asum, ar, br, cr
  real(kind=REAL64) ::  t0, t1, nstream_time, avgtime
  real(kind=REAL64), parameter ::  epsilon=1.D-8
  ! CUDA stuff
  type(dim3) :: grid, tblock

  ! ********************************************************************
  ! read and test input parameters
  ! ********************************************************************

  write(*,'(a25)') 'Parallel Research Kernels'
  write(*,'(a45)') 'CUDA Fortran STREAM triad: A = B + scalar * C'

  call prk_get_arguments('nstream',iterations=iterations,length=length,offset=offset,gpu_block_size=block_size)

  write(*,'(a23,i12)') 'Number of iterations = ', iterations
  write(*,'(a23,i12)') 'Vector length        = ', length
  write(*,'(a23,i12)') 'Offset               = ', offset
  write(*,'(a23,i12)') 'GPU block size       = ', block_size

  tblock = dim3(block_size,1,1)
  grid = dim3(ceiling(real(length)/tblock%x),1,1)

  ! ********************************************************************
  ! ** Allocate space and perform the computation
  ! ********************************************************************

#if JEFF
  call init_descriptor(length, DA)
  call init_descriptor(length, DB)
  call init_descriptor(length, DC)
#endif

  allocate( A(length), B(length), C(length), stat=err)
  if (err .ne. 0) then
    write(*,'(a20,i3)') 'allocation returned ',err
    stop 1
  endif

  do i=1,length
    A(i) = 0
    B(i) = 2
    C(i) = 2
  enddo

  scalar = 3

  t0 = 0

  do k=0,iterations
    if (k.eq.1) then
      t0 = prk_get_wtime()
    endif

    call kernel<<<grid, tblock>>>(length, scalar, A, B, C)

  enddo ! iterations

  err = cudaDeviceSynchronize()

  t1 = prk_get_wtime()

  nstream_time = t1 - t0

  ! ********************************************************************
  ! ** Analyze and output results.
  ! ********************************************************************

  ar  = 0
  br  = 2
  cr  = 2
  do k=0,iterations
      ar = ar + br + scalar * cr;
  enddo

  asum = 0
  do i=1,length
    asum = asum + abs(A(i)-ar)
  enddo

  deallocate( A,B,C )

  if (abs(asum) .gt. epsilon) then
    write(*,'(a35)') 'Failed Validation on output array'
    write(*,'(a30,f30.15)') '       Expected value: ', ar
    write(*,'(a30,f30.15)') '       Observed value: ', A(1)
    write(*,'(a35)')  'ERROR: solution did not validate'
    stop 1
  else
    write(*,'(a17)') 'Solution validates'
    avgtime = nstream_time/iterations;
    bytes = 4 * int(length,INT64) * storage_size(A)/8
    write(*,'(a12,f15.3,1x,a12,e15.6)')    &
            'Rate (MB/s): ', 1.d-6*bytes/avgtime, &
            'Avg time (s): ', avgtime
  endif

end program main

