!
! Copyright (c) 2015, Intel Corporation
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
! NAME:    transpose
!
! PURPOSE: This program measures the time for the transpose of a
!          column-major stored matrix into a row-major stored matrix.
!
! USAGE:   Program input is the matrix order and the number of times to
!          repeat the operation:
!
!          transpose <matrix_size> <# iterations> [tile size]
!
!          An optional parameter specifies the tile size used to divide the
!          individual matrix blocks for improved cache and TLB performance.
!
!          The output consists of diagnostics to make sure the
!          transpose worked and timing statistics.
!
! HISTORY: Written by  Rob Van der Wijngaart, February 2009.
!          Converted to Fortran by Jeff Hammond, January 2015
! *******************************************************************

module transpose
    use iso_fortran_env

    integer(kind=INT32), parameter :: tile_dim = 32
    integer(kind=INT32), parameter :: block_rows = 8

    interface
      attributes(global) subroutine fakenaive(A, B, DA, DB) &
                         bind(C,name="actualnaive")
        use iso_fortran_env
        implicit none
        real(kind=REAL64), intent(inout) :: A(*)
        real(kind=REAL64), intent(inout) :: B(*)
        integer(8), dimension(*), intent(in) :: DA, DB
      end subroutine fakenaive
    end interface

    interface
      attributes(global) subroutine fakecoalesced(A, B, DA, DB) &
                         bind(C,name="actualcoalesced")
        use iso_fortran_env
        implicit none
        real(kind=REAL64), intent(inout) :: A(*)
        real(kind=REAL64), intent(inout) :: B(*)
        integer(8), dimension(*), intent(in) :: DA, DB
      end subroutine fakecoalesced
    end interface

    interface
      attributes(global) subroutine fakenobankconflicts(A, B, DA, DB) &
                         bind(C,name="actualnobankconflicts")
        use iso_fortran_env
        implicit none
        real(kind=REAL64), intent(inout) :: A(*)
        real(kind=REAL64), intent(inout) :: B(*)
        integer(8), dimension(*), intent(in) :: DA, DB
      end subroutine fakenobankconflicts
    end interface

    contains

      attributes(global) subroutine naive(order, A, B)
        implicit none
        integer(kind=INT32), intent(in), value ::  order
        real(kind=REAL64), intent(inout) :: A(order,order)
        real(kind=REAL64), intent(inout) :: B(order,order)
        integer :: x, y, j
        x = (blockIdx%x-1) * tile_dim + (threadIdx%x);
        y = (blockIdx%y-1) * tile_dim + (threadIdx%y);
        do j = 0,tile_dim-1,block_rows
            B(y+j,x) = B(y+j,x) + A(x,y+j);
            A(x,y+j) = A(x,y+j) + 1.0d0;
        end do
      end subroutine naive

      attributes(global) subroutine coalesced(order, A, B)
        implicit none
        integer(kind=INT32), intent(in), value ::  order
        real(kind=REAL64), intent(inout) :: A(order,order)
        real(kind=REAL64), intent(inout) :: B(order,order)
        real(kind=REAL64), shared :: tile(32,32)
        integer :: x, y, j
        x = (blockIdx%x-1) * tile_dim + (threadIdx%x);
        y = (blockIdx%y-1) * tile_dim + (threadIdx%y);
        do j = 0,tile_dim-1,block_rows
            tile(threadIdx%x,threadIdx%y+j) = A(x,y+j);
            A(x,y+j) = A(x,y+j) + 1.0d0;
        end do
        call syncThreads()
        x = (blockIdx%y-1) * tile_dim + (threadIdx%x);
        y = (blockIdx%x-1) * tile_dim + (threadIdx%y);
        do j = 0,tile_dim-1,block_rows
            B(x,y+j) = B(x,y+j) + tile(threadIdx%y+j,threadIdx%x)
        end do
      end subroutine coalesced

      attributes(global) subroutine nobankconflicts(order, A, B)
        implicit none
        integer(kind=INT32), intent(in), value ::  order
        real(kind=REAL64), intent(inout) :: A(order,order)
        real(kind=REAL64), intent(inout) :: B(order,order)
        real(kind=REAL64), shared :: tile(33,32)
        integer :: x, y, j
        x = (blockIdx%x-1) * tile_dim + (threadIdx%x);
        y = (blockIdx%y-1) * tile_dim + (threadIdx%y);
        do j = 0,tile_dim-1,block_rows
            tile(threadIdx%x,threadIdx%y+j) = A(x,y+j);
            A(x,y+j) = A(x,y+j) + 1.0d0;
        end do
        call syncThreads()
        x = (blockIdx%y-1) * tile_dim + (threadIdx%x);
        y = (blockIdx%x-1) * tile_dim + (threadIdx%y);
        do j = 0,tile_dim-1,block_rows
            B(x,y+j) = B(x,y+j) + tile(threadIdx%y+j,threadIdx%x)
        end do
      end subroutine nobankconflicts

      attributes(global) subroutine actualnaive(A, B) &
                         bind(C,name="actualnaive")
        implicit none
        real(kind=REAL64), intent(inout) :: A(:,:)
        real(kind=REAL64), intent(inout) :: B(:,:)
        integer :: x, y, j
        x = (blockIdx%x-1) * tile_dim + (threadIdx%x);
        y = (blockIdx%y-1) * tile_dim + (threadIdx%y);
        !if (x==1 .and. y==1) print*,size(A,1), size(A,2), kind(A)
        do j = 0,tile_dim-1,block_rows
            B(y+j,x) = B(y+j,x) + A(x,y+j);
            A(x,y+j) = A(x,y+j) + 1.0d0;
        end do
      end subroutine actualnaive

      attributes(global) subroutine actualcoalesced(A, B) &
                         bind(C,name="actualcoalesced")
        implicit none
        real(kind=REAL64), intent(inout) :: A(:,:)
        real(kind=REAL64), intent(inout) :: B(:,:)
        real(kind=REAL64), shared :: tile(32,32)
        integer :: x, y, j
        x = (blockIdx%x-1) * tile_dim + (threadIdx%x);
        y = (blockIdx%y-1) * tile_dim + (threadIdx%y);
        do j = 0,tile_dim-1,block_rows
            tile(threadIdx%x,threadIdx%y+j) = A(x,y+j);
            A(x,y+j) = A(x,y+j) + 1.0d0;
        end do
        call syncThreads()
        x = (blockIdx%y-1) * tile_dim + (threadIdx%x);
        y = (blockIdx%x-1) * tile_dim + (threadIdx%y);
        do j = 0,tile_dim-1,block_rows
            B(x,y+j) = B(x,y+j) + tile(threadIdx%y+j,threadIdx%x)
        end do
      end subroutine actualcoalesced

      attributes(global) subroutine actualnobankconflicts(A, B) &
                         bind(C,name="actualnobankconflicts")
        implicit none
        real(kind=REAL64), intent(inout) :: A(:,:)
        real(kind=REAL64), intent(inout) :: B(:,:)
        real(kind=REAL64), shared :: tile(33,32)
        integer :: x, y, j
        x = (blockIdx%x-1) * tile_dim + (threadIdx%x);
        y = (blockIdx%y-1) * tile_dim + (threadIdx%y);
        do j = 0,tile_dim-1,block_rows
            tile(threadIdx%x,threadIdx%y+j) = A(x,y+j);
            A(x,y+j) = A(x,y+j) + 1.0d0;
        end do
        call syncThreads()
        x = (blockIdx%y-1) * tile_dim + (threadIdx%x);
        y = (blockIdx%x-1) * tile_dim + (threadIdx%y);
        do j = 0,tile_dim-1,block_rows
            B(x,y+j) = B(x,y+j) + tile(threadIdx%y+j,threadIdx%x)
        end do
      end subroutine actualnobankconflicts

      subroutine init_descriptor(n,D)
        implicit none
        integer(kind=INT32), intent(in) :: n
        integer(8), intent(inout) :: D(22)
        D(1)  = 35               ! tag (version)
        D(2)  =  1               ! rank
        D(3)  = 28               ! kind
        D(4)  =  4               ! len
        D(5)  =  0               ! flags
        D(6)  =  n*n             ! lsize
        D(7)  =  D(6)            ! gsize
        D(8)  =  -n              ! lbase
        D(9)  =  0               ! gbase
        D(10) =  0               ! unused
        D(11) =  1               ! dim[0].lbound
        D(12) =  n               ! dim[0].extent
        D(13) =  0               ! dim[0].sstride
        D(14) =  0               ! dim[0].soffset
        D(15) =  1               ! dim[0].lstride
        D(16) =  D(11)+D(12)     ! dim[0].ubound
        D(17) =  1               ! dim[1].lbound
        D(18) =  n               ! dim[1].extent
        D(19) =  0               ! dim[1].sstride
        D(20) =  0               ! dim[1].soffset
        D(21) =  n               ! dim[1].lstride
        D(22) =  D(17)+D(18)     ! dim[1].ubound
      end subroutine init_descriptor

end module transpose

program main
  use iso_fortran_env
  use cudafor
  use transpose
  use prk
  implicit none
  ! for argument parsing
  integer :: err
  integer :: arglen
  character(len=32) :: argtmp
  ! problem definition
  integer(kind=INT32) ::  iterations                ! number of times to do the transpose
  integer(kind=INT32) ::  order                     ! order of a the matrix
  real(kind=REAL64), allocatable, managed ::  A(:,:)! buffer to hold original matrix
  real(kind=REAL64), allocatable, managed ::  B(:,:)! buffer to hold transposed matrix
  integer(8), dimension(22), managed :: DA, DB, DC
  integer(kind=INT64) ::  bytes                     ! combined size of matrices
  ! runtime variables
  integer(kind=INT32) ::  i, j, k
  real(kind=REAL64) ::  abserr, addit, temp         ! squared error
  real(kind=REAL64) ::  t0, t1, trans_time, avgtime ! timing parameters
  real(kind=REAL64), parameter ::  epsilon=1.D-8    ! error tolerance
  ! CUDA stuff
  type(dim3) :: grid, tblock
  integer :: variant
  character(len=40), dimension(6) :: variant_name
  variant_name(1) = 'naive'
  variant_name(2) = 'coalesced'
  variant_name(3) = 'no bank conflicts'
  ! manual descriptor versions
  variant_name(4) = 'naive (manual descriptors)'
  variant_name(5) = 'coalesced (manual descriptors)'
  variant_name(6) = 'no bank conflicts (manual descriptors)'

  ! ********************************************************************
  ! read and test input parameters
  ! ********************************************************************

  write(*,'(a25)') 'Parallel Research Kernels'
  write(*,'(a38)') 'CUDA Fortran Matrix transpose: B = A^T'

  if (command_argument_count().lt.2) then
    write(*,'(a17,i1)') 'argument count = ', command_argument_count()
    write(*,'(a66)')    'Usage: ./transpose <# iterations> <matrix order> [variant (0/1/2)]'
    stop 1
  endif

  iterations = 1
  call get_command_argument(1,argtmp,arglen,err)
  if (err.eq.0) read(argtmp,'(i32)') iterations
  if (iterations .lt. 1) then
    write(*,'(a,i5)') 'ERROR: iterations must be >= 1 : ', iterations
    stop 1
  endif

  order = 1
  call get_command_argument(2,argtmp,arglen,err)
  if (err.eq.0) read(argtmp,'(i32)') order
  if (order .lt. 1) then
    write(*,'(a,i5)') 'ERROR: order must be >= 1 : ', order
    stop 1
  endif

  variant = 2
  if (command_argument_count().gt.2) then
      call get_command_argument(3,argtmp,arglen,err)
      if (err.eq.0) read(argtmp,'(i32)') variant
  endif
  if ((variant .lt. 0).or.(variant.gt.5)) then
    write(*,'(a,i5)') 'ERROR: variant must be 0-5 : ', variant
    stop 1
  endif

  write(*,'(a,i8)')  'Number of iterations = ', iterations
  write(*,'(a,i8)')  'Matrix order         = ', order
  write(*,'(a,a)')   'Variant              = ', variant_name(variant+1)

  grid   = dim3(order/tile_dim, order/tile_dim, 1)
  tblock = dim3(tile_dim, block_rows, 1)

  ! ********************************************************************
  ! ** Allocate space for the input and transpose matrix
  ! ********************************************************************

  call init_descriptor(order, DA)
  call init_descriptor(order, DB)

  allocate( A(order,order), stat=err)
  if (err .ne. 0) then
    write(*,'(a,i3)') 'allocation of A returned ',err
    stop 1
  endif

  allocate( B(order,order), stat=err )
  if (err .ne. 0) then
    write(*,'(a,i3)') 'allocation of B returned ',err
    stop 1
  endif

  t0 = 0

  do j=1,order
    do i=1,order
      A(i,j) = real(order,REAL64) * real(j-1,REAL64) + real(i-1,REAL64)
      B(i,j) = 0.0
    enddo
  enddo

  do k=0,iterations

    if (k.eq.1) t0 = prk_get_wtime()

    if (variant.eq.0) then
        call naive<<<grid, tblock>>>(order, A, B)
    else if (variant.eq.1) then
        call coalesced<<<grid, tblock>>>(order, A, B)
    else if (variant.eq.2) then
        call nobankconflicts<<<grid, tblock>>>(order, A, B)
    else if (variant.eq.3) then
        call fakenaive<<<grid, tblock>>>(A, B, DA, DB)
    else if (variant.eq.4) then
        call fakecoalesced<<<grid, tblock>>>(A, B, DA, DB)
    else if (variant.eq.5) then
        call fakenobankconflicts<<<grid, tblock>>>(A, B, DA, DB)
    endif
    err = cudaDeviceSynchronize()

  enddo ! iterations

  t1 = prk_get_wtime()

  trans_time = t1 - t0

  ! ********************************************************************
  ! ** Analyze and output results.
  ! ********************************************************************

  abserr = 0.0
  ! this will overflow if iterations>>1000
  addit = (0.5*iterations) * (iterations+1)
  do j=1,order
    do i=1,order
      temp = ((real(order,REAL64)*real(i-1,REAL64))+real(j-1,REAL64)) &
           * real(iterations+1,REAL64)
      abserr = abserr + abs(B(i,j) - (temp+addit))
    enddo
  enddo

  if (abserr .lt. epsilon) then
    write(*,'(a)') 'Solution validates'
    avgtime = trans_time/iterations
    bytes = 2 * int(order,INT64) * int(order,INT64) * storage_size(A)/8
    write(*,'(a,f13.6,a,f10.6)') 'Rate (MB/s): ',(1.d-6*bytes)/avgtime, &
           ' Avg time (s): ', avgtime
  else
    write(*,'(a,e30.15,a,e30.15)') 'ERROR: Aggregate squared error ',abserr, &
           'exceeds threshold ',epsilon
    call flush(0)
    do j=1,order
      do i=1,order
        temp = ((real(order,REAL64)*real(i-1,REAL64))+real(j-1,REAL64)) &
             * real(iterations+1,REAL64)
        write(*,'(i4,1x,i4,1x,e10.5,1x,e10.5,1x,e10.5)') i,j,A(i,j),B(i,j),temp+addit
      enddo
    enddo
    call flush(0)
    stop 1
  endif

  deallocate( A )
  deallocate( B )

end program main

