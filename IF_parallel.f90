module iterfilt
!to compile type f2pygnuopenmp stats_parallel.f90 stats_parallel

  use omp_lib

contains

  subroutine convolve(fin,ker,fout,nx,nker,nthreads)

    implicit none
    
    real, dimension(0:nx-1),intent(in) :: fin
    real, dimension(0:nx-1),intent(out) :: fout
    real, dimension(0:nker-1),intent(in) :: ker
     
    integer, optional :: nthreads
    
    integer :: nx,nker,hker

    integer i,j, num_thr


    real :: ostart_time,oend_time,SD

    !Check if any arguments are found
    if (present(nthreads)) call omp_set_num_threads(nthreads)
            
    hker = nker/2
    fout(:) = 0
    !$OMP PARALLEL 
    !$OMP DO
    do i = 0,hker-1
        do j = 0,i+hker
            fout(i) = fout(i) + fin(j) * ker(hker-i+j) 
            fout(nx-1-i) = fout(nx-1-i) + fin(nx-1-j) * ker(hker+i-j) 
        enddo
    enddo
    !$OMP END DO
    !$OMP DO
    do i = hker,nx - hker - 1
        do j = 0,nker - 1
            fout(i) = fout(i) + fin(i-hker+j) * ker(j)
        enddo
    enddo
    !$OMP END DO
    !$OMP END PARALLEL

    !computing norm
    SD =  sum(fout(:)**2)/sum(fin(:)**2)
    fout(:) = fin(:) - fout(:)
    
    oend_time = omp_get_wtime()

    print*, 'norm2 residual: ',SD
    print*, '("Computation elapsed time (OMP): ',oend_time-ostart_time,' sec'

  end subroutine convolve

  subroutine compute_imf
  end subroutine compute_imf

  
!  subroutine get_lkm_var(imfs,logM,lkrt,vart,nx,nimfs,nlogm,nthreads)
!
!    implicit none
!    
!    real, dimension(nx,nimfs),intent(in) :: imfs
!    real, dimension(nx,nimfs),intent(out) :: lkrt,vart
!    integer, dimension(nlogm),intent(in) :: logM
!     
!    integer :: nx,nimfs,nlogm
!
!    integer, optional :: nthreads
!    integer k,j, num_thr
!
!    real, save :: mu 
!
!    real :: ostart_time,oend_time
!
!    !$OMP THREADPRIVATE(mu)
!
!    ostart_time=0.
!    oend_time=0.
!    !Check if any arguments are found
!    if (present(nthreads)) call omp_set_num_threads(nthreads)
!
!    !$OMP PARALLEL
!    num_thr = omp_get_num_threads()
!    !$OMP END PARALLEL
!    !print some diagnostics
!    print*,'# OMP THREADS:',num_thr
!    !call cpu_time(ostart_time)
!    ostart_time = omp_get_wtime()
!    print*,nx,nimfs,nlogm 
!    !$OMP PARALLEL
!    do k = 1,nlogm
!    !$OMP DO 
!    do j=logM(k)+1,nx-logM(k)
!      
!      mu = sum(imfs(j-logM(k):j+logM(k),k))/(2*logM(k)+1)
!
!      vart(j,k) = sum( (imfs(j-logM(k):j+logM(k),k) - mu )**2 )
!      lkrt(j,k) = (2*logM(k) + 1) * sum( (imfs(j-logM(k):j+logM(k),k) - mu )**4)/vart(j,k)**2
!      vart(j,k) = vart(j,k)/(2*logM(k)+1) !BIASED SAMPLE VARIANCE
!    enddo
!    !$OMP END DO
!    enddo
!    !$OMP END PARALLEL
!    
!    
!    !call cpu_time(oend_time)
!    oend_time = omp_get_wtime()
!
!    print*, '("Computation elapsed time (OMP): ',oend_time-ostart_time,' sec'
!
!  end subroutine get_lkm_var
!
!  function kurtosis(y,n)
!    implicit none
!    real, intent(in), dimension(n) :: y
!    integer, intent(in) :: n
!    real :: kurtosis,mu
!    
!    mu = sum(y)/n
!    kurtosis = sum((y-mu)**4)/sum((y-mu)**2)**2 *n
!  
!  
!  end function kurtosis




end module iterfilt
