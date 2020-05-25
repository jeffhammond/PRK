!
! Copyright (c) 2020, Intel Corporation
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
! *******************************************************************
!
! NAME:    Stencil
!
! PURPOSE: This program tests the efficiency with which a space-invariant,
!          linear, symmetric filter (stencil) can be applied to a square
!          grid or image.
!
! USAGE:   The program takes as input the linear
!          dimension of the grid, and the number of iterations on the grid
!
!                <progname> <iterations> <grid size>
!
!          The output consists of diagnostics to make sure the
!          algorithm worked, and of timing statistics.
!
! FUNCTIONS CALLED:
!
!          Other than standard C functions, the following functions are used in
!          this program:
!          wtime()
!
! HISTORY: - Written by Rob Van der Wijngaart, February 2009.
!          - RvdW: Removed unrolling pragmas for clarity;
!            added constant to array "in" at end of each iteration to force
!            refreshing of neighbor data in parallel versions; August 2013
!          - Converted to Fortran by Jeff Hammond, January-February 2016.
!          - Global Arrays by Jeff Hammond, May 2020.
! *******************************************************************

subroutine initialize_w(is_star,r,W)
  use iso_fortran_env
  implicit none
  logical, intent(in) :: is_star
  integer(kind=INT32), intent(in) :: r
  real(kind=REAL64), intent(inout) :: W(-r:r,-r:r)
  integer(kind=INT32) :: ii, jj
  ! fill the stencil weights to reflect a discrete divergence operator
  W = 0.0d0
  if (is_star) then
    do ii=1,r
      W(0, ii) =  1.0d0/real(2*ii*r,REAL64)
      W(0,-ii) = -1.0d0/real(2*ii*r,REAL64)
      W( ii,0) =  1.0d0/real(2*ii*r,REAL64)
      W(-ii,0) = -1.0d0/real(2*ii*r,REAL64)
    enddo
  else
    ! Jeff: check that this is correct with the new W indexing
    do jj=1,r
      do ii=-jj+1,jj-1
        W( ii, jj) =  1.0d0/real(4*jj*(2*jj-1)*r,REAL64)
        W( ii,-jj) = -1.0d0/real(4*jj*(2*jj-1)*r,REAL64)
        W( jj, ii) =  1.0d0/real(4*jj*(2*jj-1)*r,REAL64)
        W(-jj, ii) = -1.0d0/real(4*jj*(2*jj-1)*r,REAL64)
      enddo
      W( jj, jj)  =  1.0d0/real(4*jj*r,REAL64)
      W(-jj,-jj)  = -1.0d0/real(4*jj*r,REAL64)
    enddo
  endif
end subroutine initialize_w

subroutine apply_stencil(is_star,tiling,tile_size,r,n,W,A,B)
  use iso_fortran_env
  implicit none
  logical, intent(in) :: is_star, tiling
  integer(kind=INT32), intent(in) :: tile_size, r, n
  real(kind=REAL64), intent(in) :: W(-r:r,-r:r)
  real(kind=REAL64), intent(in) :: A(n,n)
  real(kind=REAL64), intent(inout) :: B(n,n)
  integer(kind=INT32) :: i, j, ii, jj, it, jt
  if (is_star) then
    if (.not.tiling) then
      !$omp do
      do j=r,n-r-1
        do i=r,n-r-1
            ! do not use Intel Fortran unroll directive here (slows down)
            do jj=-r,r
              B(i+1,j+1) = B(i+1,j+1) + W(0,jj) * A(i+1,j+jj+1)
            enddo
            do ii=-r,-1
              B(i+1,j+1) = B(i+1,j+1) + W(ii,0) * A(i+ii+1,j+1)
            enddo
            do ii=1,r
              B(i+1,j+1) = B(i+1,j+1) + W(ii,0) * A(i+ii+1,j+1)
            enddo
        enddo
      enddo
      !$omp end do
    else ! tiling
      !$omp do
      do jt=r,n-r-1,tile_size
        do it=r,n-r-1,tile_size
          do j=jt,min(n-r-1,jt+tile_size-1)
            do i=it,min(n-r-1,it+tile_size-1)
              do jj=-r,r
                B(i+1,j+1) = B(i+1,j+1) + W(0,jj) * A(i+1,j+jj+1)
              enddo
              do ii=-r,-1
                B(i+1,j+1) = B(i+1,j+1) + W(ii,0) * A(i+ii+1,j+1)
              enddo
              do ii=1,r
                B(i+1,j+1) = B(i+1,j+1) + W(ii,0) * A(i+ii+1,j+1)
              enddo
            enddo
          enddo
        enddo
      enddo
      !$omp end do
    endif ! tiling
  else ! grid
    if (.not.tiling) then
      !$omp do
      do j=r,n-r-1
        do i=r,n-r-1
          do jj=-r,r
            do ii=-r,r
              B(i+1,j+1) = B(i+1,j+1) + W(ii,jj) * A(i+ii+1,j+jj+1)
            enddo
          enddo
        enddo
      enddo
      !$omp end do
    else ! tiling
      !$omp do
      do jt=r,n-r-1,tile_size
        do it=r,n-r-1,tile_size
          do j=jt,min(n-r-1,jt+tile_size-1)
            do i=it,min(n-r-1,it+tile_size-1)
              do jj=-r,r
                do ii=-r,r
                  B(i+1,j+1) = B(i+1,j+1) + W(ii,jj) * A(i+ii+1,j+jj+1)
                enddo
              enddo
            enddo
          enddo
        enddo
      enddo
      !$omp end do
    endif ! tiling
  endif ! star
end subroutine apply_stencil

program main
  use iso_fortran_env
  use mpi_f08
  implicit none
#include 'global.fh'
#include 'mafdecls.fh'
!#include 'ga-mpi.fh' ! unused
  ! for argument parsing
  integer :: err
  integer :: arglen
  character(len=32) :: argtmp
  ! MPI - should always use 32-bit INTEGER
  integer(kind=INT32), parameter :: requested = MPI_THREAD_SERIALIZED
  integer(kind=INT32) :: provided
  integer(kind=INT32) :: world_size, world_rank
  integer(kind=INT32) :: ierr
  type(MPI_Comm), parameter :: world = MPI_COMM_WORLD
  ! GA - compiled with 64-bit INTEGER
  logical :: ok
  integer :: me, np
  integer :: A, B, AT
  integer :: mylo(2),myhi(2)
  real(kind=REAL64), parameter :: one  = 1.d0
  real(kind=REAL64), allocatable ::  T(:,:)
  ! problem definition
  integer(kind=INT32) ::  iterations
  integer(kind=INT32) :: n
  integer(kind=INT32) :: stencil_size                  ! number of points in stencil
  integer(kind=INT32) :: tile_size                     ! loop nest block factor
  integer(kind=INT64) :: bytes, max_mem
  logical :: is_star                                   ! true = star, false = grid
  integer(kind=INT32), parameter :: r=RADIUS           ! radius of stencil
  real(kind=REAL64) :: W(-r:r,-r:r)                    ! weights of points in the stencil
  real(kind=REAL64), allocatable :: TA(:,:), TB(:,:)   ! grid values
  real(kind=REAL64), parameter :: cx=1.d0, cy=1.d0
  ! runtime variables
  integer(kind=INT32) :: i, j, k
  integer(kind=INT64) :: flops                          ! floating point ops per iteration
  real(kind=REAL64) :: norm, reference_norm             ! L1 norm of solution
  integer(kind=INT64) :: active_points                  ! interior of grid with respect to stencil
  real(kind=REAL64) :: t0, t1, stencil_time, avgtime    ! timing parameters
  real(kind=REAL64), parameter ::  epsilon=1.d-8        ! error tolerance

  ! ********************************************************************
  ! read and test input parameters
  ! ********************************************************************

  if (command_argument_count().lt.2) then
    write(*,'(a17,i1)') 'argument count = ', command_argument_count()
    write(*,'(a32,a29)') 'Usage: ./stencil <# iterations> ', &
                      '<array dimension> [tile_size]'
    stop 1
  endif

  iterations = 1
  call get_command_argument(1,argtmp,arglen,err)
  if (err.eq.0) read(argtmp,'(i32)') iterations
  if (iterations .lt. 1) then
    write(*,'(a,i5)') 'ERROR: iterations must be >= 1 : ', iterations
    stop 1
  endif

  n = 1
  call get_command_argument(2,argtmp,arglen,err)
  if (err.eq.0) read(argtmp,'(i32)') n
  if (n .lt. 1) then
    write(*,'(a,i5)') 'ERROR: array dimension must be >= 1 : ', n
    stop 1
  endif

  tiling    = .false.
  tile_size = n
  if (command_argument_count().gt.2) then
    call get_command_argument(3,argtmp,arglen,err)
    if (err.eq.0) read(argtmp,'(i32)') tile_size
    if ((tile_size .lt. 1).or.(tile_size.gt.n)) then
      write(*,'(a,i5,a,i5)') 'WARNING: tile_size ',tile_size,&
                             ' must be >= 1 and <= ',n
    else
      tiling = .true.
    endif
  endif

  ! TODO: parse runtime input for star/grid
#ifdef STAR
  is_star = .true.
#else
  is_star = .false.
#endif

  ! TODO: parse runtime input for radius

  if (r .lt. 1) then
    write(*,'(a,i5,a)') 'ERROR: Stencil radius ',r,' should be positive'
    stop 1
  else if ((2*r+1) .gt. n) then
    write(*,'(a,i5,a,i5)') 'ERROR: Stencil radius ',r,&
                           ' exceeds grid size ',n
    stop 1
  endif

  call mpi_init_thread(requested,provided)

  !call ga_initialize()
  ! ask GA to allocate enough memory for 4 matrices, just to be safe
  max_mem = order * order * 4 * ( storage_size(one) / 8 )
  call ga_initialize_ltd(max_mem)

  me = ga_nodeid()
  np = ga_nnodes()

  !if (me.eq.0) print*,'max_mem=',max_mem

#if PRK_CHECK_GA_MPI
  ! We do use MPI anywhere, but if we did, we would need to avoid MPI collectives
  ! on the world communicator, because it is possible for that to be larger than
  ! the GA world process group.  In this case, we need to get the MPI communicator
  ! associated with GA world, but those routines assume MPI communicators are integers.

  call MPI_Comm_rank(world, world_rank)
  call MPI_Comm_size(world, world_size)

  if ((me.ne.world_rank).or.(np.ne.world_size)) then
      write(*,'(a12,i8,i8)') 'rank=',me,world_rank
      write(*,'(a12,i8,i8)') 'size=',me,world_size
      call ga_error('MPI_COMM_WORLD is unsafe to use!!!',np)
  endif
#endif

  if (me.eq.0) then
    write(*,'(a25)') 'Parallel Research Kernels'
    write(*,'(a50)') 'Fortran Global Arrays Stencil execution on 2D grid'
    write(*,'(a22,i12)') 'Number of GA procs   = ', np
    write(*,'(a,i8)') 'Number of iterations    = ', iterations
    write(*,'(a,i8)') 'Matrix order            = ', order
  endif

  call ga_sync()

  ! ********************************************************************
  ! ** Allocate space for the input and transpose matrix
  ! ********************************************************************

  t0 = 0.0d0

  !print*,'order=',order
  ! must cast int32 order to integer...
  ok = ga_create(MT_DBL, int(order), int(order),'A',-1,-1, A)
  if (.not.ok) then
    call ga_error('allocation of A failed',100)
  endif

  ok = ga_duplicate(A,B,'B')
  if (.not.ok) then
    call ga_error('duplication of A as B failed ',101)
  endif
  call ga_zero(B)

  ok = ga_duplicate(A,AT,'A^T')
  if (.not.ok) then
    call ga_error('duplication of A as A^T failed ',102)
  endif
  call ga_zero(AT)

  call ga_sync()

  call ga_distribution( A, me, mylo(1), myhi(1), mylo(2), myhi(2) )
  !write(*,'(a7,5i6)') 'local:',me,mylo(1), myhi(1), mylo(2), myhi(2)
  allocate( T(myhi(1)-mylo(1)+1,myhi(2)-mylo(2)+1), stat=err)
  if (err .ne. 0) then
    call ga_error('allocation of T failed',err)
  endif
  do j=mylo(2),myhi(2)
    jj = j-mylo(2)+1
    do i=mylo(1),myhi(1)
      ii = i-mylo(1)+1
      T(ii,jj) = real(order,REAL64) * real(j-1,REAL64) + real(i-1,REAL64)
    enddo
  enddo
  !write(*,'(a8,5i6)') 'ga_put:',mylo(1), myhi(1), mylo(2), myhi(2), myhi(2)-mylo(2)+1
  call ga_put( A, mylo(1), myhi(1), mylo(2), myhi(2), T, myhi(1)-mylo(1)+1 )
  call ga_sync()

  ok = ma_init(MT_DBL, order*order, 0)
  if (.not.ok) then
    call ga_error('ma_init failed', order)
  endif

  if (order.lt.10) then
    call ga_print(A)
  endif

  do k=0,iterations

    ! start timer after a warmup iteration
    if (k.eq.1) then
        call ga_sync()
        t0 = ga_wtime()
    endif

    ! B += A^T
    ! A += 1
    call ga_transpose(A,AT)      ! C  = A^T
    call ga_sync()               ! ga_tranpose does not synchronize after remote updates
    call ga_add(one,B,one,AT,B)  ! B += A^T
    call ga_add_constant(A, one) ! A += 1
    !call ga_sync()               ! ga_add and ga_add_constant synchronize

  enddo ! iterations

  call ga_sync()
  t1 = ga_wtime()

  trans_time = t1 - t0

  ! ********************************************************************
  ! ** Analyze and output results.
  ! ********************************************************************

  if (order.lt.10) then
    call ga_print(A)
    call ga_print(AT)
    call ga_print(B)
  endif

  !write(*,'(a8,5i6)') 'ga_get:',mylo(1), myhi(1), mylo(2), myhi(2), myhi(2)-mylo(2)+1
  call ga_get( B, mylo(1), myhi(1), mylo(2), myhi(2), T, myhi(1)-mylo(1)+1 )

  abserr = 0.0
  ! this will overflow if iterations>>1000
  addit = (0.5*iterations) * (iterations+1)
  do j=mylo(2),myhi(2)
    jj = j-mylo(2)+1
    do i=mylo(1),myhi(1)
      ii = i-mylo(1)+1
      temp = ((real(order,REAL64)*real(i-1,REAL64))+real(j-1,REAL64)) &
           * real(iterations+1,REAL64)
      abserr = abserr + abs(T(ii,jj) - (temp+addit))
    enddo
  enddo
  call ga_dgop(MT_DBL, abserr, 1, '+')

  deallocate( T )

  ok = ga_destroy(AT)
  if (.not.ok) then
      call ga_error('ga_destroy failed',201)
  endif

  ok = ga_destroy(A)
  if (.not.ok) then
      call ga_error('ga_destroy failed',202)
  endif

  ok = ga_destroy(B)
  if (.not.ok) then
      call ga_error('ga_destroy failed',203)
  endif

  call ga_sync()

  if (me.eq.0) then
    if (abserr .lt. epsilon) then
      write(*,'(a)') 'Solution validates'
      avgtime = trans_time/iterations
      bytes = 2 * int(order,INT64) * int(order,INT64) * storage_size(one)/8
      write(*,'(a,f13.6,a,f10.6)') 'Rate (MB/s): ',(1.d-6*bytes)/avgtime, &
             ' Avg time (s): ', avgtime
    else
      write(*,'(a,f30.15,a,f30.15)') 'ERROR: Aggregate squared error ',abserr, &
             'exceeds threshold ',epsilon
      call ga_error('Answer wrong',911)
    endif
  endif

  call ga_sync()

#ifdef PRK_GA_SUMMARY
  if (me.eq.0) write(*,'(a)') ! add an empty line
  call ga_summarize(.false.)
  if (me.eq.0) then
    call ma_print_stats()
  endif
#endif

  call ga_terminate()
  call mpi_finalize()

end program main

