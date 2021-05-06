! -------------------------------- NUMERICAL FLUXES --------------------------------
subroutine rpn2(ixy,maxm,meqn,mwaves,maux,mbc,mx,ql,qr,auxl,auxr,wave,s,amdq,apdq)
! =====================================================
! INPUT
! ixy: dimension, ixy == 1 -> x-dimension
!                 ixy == 2 -> y-dimension
! maxm: TOdo: ??

! meqn: the number of equations in the system
! mwaves: the number of waves in each Riemann solution
! maux: the number of auxiliary variables
! mbc: number of boundary(ghost) cells
! mx: the number of grid cells
! ql: contains the state vector at the left edge of each cell
! qr: contains the state vector at the right edge of each cell
!   NOTE: the i'th Riemann problem has left state qr(:, i-1) and right state ql(:,i)
!   NOTE: as we do not use reconstruction the left and right state of each cell should be the same
! auxl: contains the aux variables at the left edge of each cell
! auxr: contains the aux variables at the right edge of each cell
!   NOTE: as we do not use reconstruction the left and right aux variables of each cell should be the same

! OUTPUT
! wave: contains the waves
! s: contains the speeds
! amdq: flux difference leftgoing part (A- Delta q)
! apdq: flux difference rightgoing part (A+ Delta q)
! NOTE: flux difference: f(qr(i-1)) - f(ql(i))
! ====================================================
    implicit none
    double precision :: flux_precision, condition
    common /cparam/ flux_precision

    integer, intent(in) :: ixy
    integer, intent(in) :: maxm
    integer, intent(in) :: meqn
    integer, intent(in) :: mwaves
    integer, intent(in) :: maux
    integer, intent(in) :: mbc
    integer, intent(in) :: mx
    double precision, dimension(meqn, mwaves, 1-mbc:maxm+mbc), intent(out) :: wave
    double precision, dimension(mwaves, 1-mbc:maxm+mbc), intent(out) :: s
    double precision, dimension(meqn, 1-mbc:maxm+mbc), intent(in) :: ql
    double precision, dimension(meqn, 1-mbc:maxm+mbc), intent(in) :: qr
    double precision, dimension(meqn, 1-mbc:maxm+mbc), intent(out) :: apdq
    double precision, dimension(meqn, 1-mbc:maxm+mbc), intent(out) :: amdq
    double precision, dimension(maux, 1-mbc:maxm+mbc), intent(in) :: auxl
    double precision, dimension(maux, 1-mbc:maxm+mbc), intent(in) :: auxr

    double precision, dimension(3,3) :: jacobian, eigvecs, inveigvecs
    double precision, dimension(3) :: eigvals, q_mean, dlambda, alpha

    logical :: cplx
    integer :: i, m

    do i = 1, mx+2
        q_mean = 0.5d0*(auxl(4:6, i) + auxr(4:6, i-1))
        dlambda = ql(1:3, i) - qr(1:3, i-1)
        if ((abs(q_mean(1)) > flux_precision)) THEN
            if (ixy == 1) THEN
                call calcJacobianX(q_mean, jacobian)
            ELSE
                call calcJacobianY(q_mean, jacobian)
            end if

            jacobian = -transpose(jacobian)

            call calcEigen(jacobian, eigvecs, eigvals, cplx)
            if (cplx) then
                read *, q_mean
            end if

            ! jacobian is overwritten!
            call calcInverse(eigvecs, inveigvecs)
            call decomposeJump(inveigvecs, dlambda, alpha)
            call calcWaves(alpha, eigvecs, wave(:, :, i))

            s(:, i) = eigvals

            call calcAMDQ(inveigvecs, eigvals, eigvecs, dlambda, amdq(:, i))
            call calcAPDQ(inveigvecs, eigvals, eigvecs, dlambda, apdq(:, i))
        ELSE
            s(:,i) = 1.d0
            wave(:, :, i) = 0.d0
            amdq(:,i) = 0.d0
            apdq(:,i) = 0.d0
        end if
    end do
end subroutine rpn2

! -------------------------------- XI APPROXIMATION --------------------------------
subroutine calcChi(abs_a, chi)
  implicit none
  double precision, intent(in) :: abs_a
  double precision, intent(out) :: chi
  double precision :: a_0, a_2, a_4, a_6
  double precision :: b_0, b_2
  a_0 = 0.621529d+0
  a_2 = 0.348509d+0
  a_4 = -0.139318d+0
  a_6 = 0.720371d+0
  b_0 = 1.87095d+0
  b_2 = -1.32002d+0

  chi = (abs_a**6.d0*a_6 + abs_a**4.d0*a_4 + abs_a**2.d0*a_2 + a_0)/(abs_a**4.d0 + abs_a**2.d0*b_2 + b_0)
  chi = max(min(chi, 1.d0), 1.d0/3.d0)
end subroutine calcChi

subroutine calcDChi(abs_a, dchi)
    implicit none
    double precision, intent(in) :: abs_a
    double precision, intent(out) :: dchi
    double precision :: a_0, a_2, a_4, a_6
    double precision :: b_0, b_2
    a_0 = 0.621529d+0
    a_2 = 0.348509d+0
    a_4 = -0.139318d+0
    a_6 = 0.720371d+0
    b_0 = 1.87095d+0
    b_2 = -1.32002d+0

    dchi = (-4.d0*abs_a**3.d0 - 2.d0*abs_a*b_2)*(a_0 + a_2*abs_a**2.d0 + a_4*abs_a**4.d0 + a_6*abs_a**6.d0)/(abs_a**4.d0 + abs_a**2.d0*b_2 + b_0)**2.d0 + (2.d0*a_2*abs_a + 4.d0*a_4*abs_a**3.d0 + 6.d0*a_6*abs_a**5.d0)/(abs_a**4.d0 + abs_a**2.d0*b_2 + b_0)
    dchi = max(min(dchi, 2.d0), 0.d0)
end subroutine calcDChi

subroutine calcChiTotal(q, chi)
    implicit none
    double precision, dimension(3), intent(in) :: q
    double precision, intent(out) :: chi
    double precision :: abs_a

    abs_a = sqrt(q(2)**2.d0 + q(3)**2.d0)/q(1)
    call calcChi(abs_a, chi)

end subroutine calcChiTotal

subroutine calcDChiTotal(q, dchi)
    implicit none
    double precision, dimension(3), intent(in) :: q
    double precision, dimension(3), intent(out) :: dchi
    double precision :: dchi_, abs_a
    integer :: m

    abs_a = sqrt(q(2)**2.d0 + q(3)**2.d0)/q(1)
    call calcDChi(abs_a, dchi_)

    dchi(1) = dchi_ * (-sqrt(q(2)**2.d0 + q(3)**2.d0)/q(1)**2.d0)
    dchi(2) = dchi_ * (q(2)/(q(1)*sqrt(q(2)**2.d0 + q(3)**2.d0)))
    dchi(3) = dchi_ * (q(3)/(q(1)*sqrt(q(2)**2.d0 + q(3)**2.d0)))
    do m = 1,3
        if (isnan(dchi(m))) then
   !         print *, "dchi(", m, ") = nan"
            dchi(m) = 0.d0
        end if
    end do
end subroutine calcDChiTotal

! -------------------------------- JACOBIAN --------------------------------

subroutine calcJacobianX(q, jacF)
    implicit none
    double precision, dimension(3), intent(in) :: q
    double precision, dimension(3,3), intent(out) :: jacF

    double precision, dimension(3) :: dchi
    double precision :: chi
    integer :: i,m

    call calcChiTotal(q, chi)
    call calcDChiTotal(q, dchi)

    jacF(1,1) = 0.d0
    jacF(1,2) = 1.d0
    jacF(1,3) = 0.d0
    jacF(2,1) = q(1)*(3.d0*q(2)**2.d0*dchi(1)/(2.d0*q(2)**2.d0 + 2.d0*q(3)**2.d0) - dchi(1)/2.d0) + q(2)**2.d0*(3.d0*chi - 1.d0)/(2.d0*q(2)**2.d0 + 2.d0*q(3)**2.d0) - chi/2.d0 + 1.d0/2
    jacF(2,2) = q(1)*(-4.d0*q(2)**3.d0*(3.d0*chi - 1.d0)/(2.d0*q(2)**2.d0 + 2.d0*q(3)**2.d0)**2.d0 + 3.d0*q(2)**2.d0*dchi(2)/(2.d0*q(2)**2.d0 + 2.d0*q(3)**2.d0) + 2.d0*q(2)*(3.d0*chi - 1.d0)/(2.d0*q(2)**2.d0 + 2.d0*q(3)**2.d0) - dchi(2)/2.d0)
    jacF(2,3) = q(1)*(-4.d0*q(2)**2.d0*q(3)*(3.d0*chi - 1.d0)/(2.d0*q(2)**2.d0 + 2.d0*q(3)**2.d0)**2.d0 + 3.d0*q(2)**2.d0*dchi(3)/(2.d0*q(2)**2.d0 + 2.d0*q(3)**2.d0) - dchi(3)/2.d0)
    jacF(3,1) = 3.d0*q(1)*q(2)*q(3)*dchi(1)/(2.d0*q(2)**2.d0 + 2.d0*q(3)**2.d0) + q(2)*q(3)*(3.d0*chi - 1.d0)/(2.d0*q(2)**2.d0 + 2.d0*q(3)**2.d0)
    jacF(3,2) = -4.d0*q(1)*q(2)**2.d0*q(3)*(3.d0*chi - 1.d0)/(2.d0*q(2)**2.d0 + 2.d0*q(3)**2.d0)**2.d0 + 3.d0*q(1)*q(2)*q(3)*dchi(2)/(2.d0*q(2)**2.d0 + 2.d0*q(3)**2.d0) + q(1)*q(3)*(3.d0*chi - 1.d0)/(2.d0*q(2)**2.d0 + 2.d0*q(3)**2.d0)
    jacF(3,3) = -4.d0*q(1)*q(2)*q(3)**2.d0*(3.d0*chi - 1.d0)/(2.d0*q(2)**2.d0 + 2.d0*q(3)**2.d0)**2.d0 + 3.d0*q(1)*q(2)*q(3)*dchi(3)/(2.d0*q(2)**2.d0 + 2.d0*q(3)**2.d0) + q(1)*q(2)*(3.d0*chi - 1.d0)/(2.d0*q(2)**2.d0 + 2.d0*q(3)**2.d0)
    do i = 1,3
        do m = 1,3
            if (isnan(jacF(i,m))) then
    !            print *, "nan", i, m
     !           print *, q
                jacF(i,m) = 0.d0
            end if
        end do
    end do
end subroutine calcJacobianX

subroutine calcJacobianY(q, jacF)
    implicit none
    double precision, dimension(3), intent(in) :: q
    double precision, dimension(3,3), intent(out) :: jacF

    double precision, dimension(3) :: dchi
    double precision :: chi
    integer :: i,m

    call calcChiTotal(q, chi)
    call calcDChiTotal(q, dchi)

    jacF(1,1) = 0.d0
    jacF(1,2) = 0.d0
    jacF(1,3) = 1.d0
    jacF(2,1) = 3.d0*q(1)*q(2)*q(3)*dchi(1)/(2.d0*q(2)**2.d0 + 2.d0*q(3)**2.d0) + q(2)*q(3)*(3.d0*chi - 1.d0)/(2.d0*q(2)**2.d0 + 2.d0*q(3)**2.d0)
    jacF(2,2) = -4.d0*q(1)*q(2)**2.d0*q(3)*(3.d0*chi - 1.d0)/(2.d0*q(2)**2.d0 + 2.d0*q(3)**2.d0)**2.d0 + 3.d0*q(1)*q(2)*q(3)*dchi(2)/(2.d0*q(2)**2.d0 + 2.d0*q(3)**2.d0) + q(1)*q(3)*(3.d0*chi - 1.d0)/(2.d0*q(2)**2.d0 + 2.d0*q(3)**2.d0)
    jacF(2,3) = -4.d0*q(1)*q(2)*q(3)**2.d0*(3.d0*chi - 1.d0)/(2.d0*q(2)**2.d0 + 2.d0*q(3)**2.d0)**2.d0 + 3.d0*q(1)*q(2)*q(3)*dchi(3)/(2.d0*q(2)**2.d0 + 2.d0*q(3)**2.d0) + q(1)*q(2)*(3.d0*chi - 1.d0)/(2.d0*q(2)**2.d0 + 2.d0*q(3)**2.d0)
    jacF(3,1) = q(1)*(3.d0*q(3)**2.d0*dchi(1)/(2.d0*q(2)**2.d0 + 2.d0*q(3)**2.d0) - dchi(1)/2.d0) + q(3)**2.d0*(3.d0*chi - 1.d0)/(2.d0*q(2)**2.d0 + 2.d0*q(3)**2.d0) - chi/2.d0 + 1.d0/2
    jacF(3,2) = q(1)*(-4.d0*q(2)*q(3)**2.d0*(3.d0*chi - 1.d0)/(2.d0*q(2)**2.d0 + 2.d0*q(3)**2.d0)**2.d0 + 3.d0*q(3)**2.d0*dchi(2)/(2.d0*q(2)**2.d0 + 2.d0*q(3)**2.d0) - dchi(2)/2.d0)
    jacF(3,3) = q(1)*(-4.d0*q(3)**3.d0*(3.d0*chi - 1.d0)/(2.d0*q(2)**2.d0 + 2.d0*q(3)**2.d0)**2.d0 + 3.d0*q(3)**2.d0*dchi(3)/(2.d0*q(2)**2.d0 + 2.d0*q(3)**2.d0) + 2.d0*q(3)*(3.d0*chi - 1.d0)/(2.d0*q(2)**2.d0 + 2.d0*q(3)**2.d0) - dchi(3)/2.d0)
    do i = 1,3
        do m = 1,3
            if (isnan(jacF(i,m))) then
      !          print *, "nan"
      !          print *, q
                jacF(i,m) = 0.d0
            end if
        end do
    end do
end subroutine calcJacobianY

subroutine calcEigen(mat, evcR, evlR, cplx)
    double precision, dimension(3,3), intent(in) :: mat
    double precision, dimension(3,3), intent(out) :: evcR
    logical, intent(out) :: cplx
    double precision, dimension(3,3) :: dummy
    double precision, dimension(3), intent(out) :: evlR
    double precision, dimension(3) :: evlI
    integer :: LWORK, INFO, N
    double precision, dimension(102) :: WORK
    N = 3
    LWORK = 102
    cplx = .false.
    call DGEEV( 'N', 'V', N, mat, N, evlR, evlI, dummy, N, evcR, N, WORK, LWORK, INFO )
    if (ANY(abs(evlI) > 1.d-30)) THEN
!        print *, "complex eigenvalue"
        cplx = .true.
        !read *, s
    end if
!    if (INFO /= 0) THEN
!        print *, "DGEEV", INFO
!    end if
end subroutine calcEigen

subroutine calcInverse(mat, inv)
    implicit none
    double precision, dimension(3,3), intent(in) :: mat
    double precision, dimension(3,3), intent(out) :: inv
    integer, dimension(3) :: IPIV
    integer :: LWORK, INFO, N
    double precision, dimension(102) :: WORK
    N = 3
    LWORK = 102
    inv = mat
    call DGETRF(N, N, inv, N, IPIV, INFO)
  !  if (INFO /= 0) THEN
 !       print *, "DGETRF", INFO
  !  end if
    call DGETRI(N, inv, N, IPIV, WORK, LWORK, INFO)
  !  if (INFO /= 0) THEN
  !      print *, "DGETRI", INFO
  !  end if
end subroutine calcInverse

subroutine calcAP(eigvals, Aplus)
    implicit none
    double precision, dimension(3, 3), intent(out) :: Aplus
    double precision, dimension(3), intent(in) :: eigvals
    integer :: i
    Aplus = 0.d0
    do i = 1,3
        Aplus(i,i) = max(0.d0, eigvals(i))
    end do
end subroutine calcAP

subroutine calcAM(eigvals, Aminus)
    implicit none
    double precision, dimension(3, 3), intent(out) :: Aminus
    double precision, dimension(3), intent(in) :: eigvals
    integer :: i
    Aminus = 0.d0
    do i = 1,3
        Aminus(i,i) = min(0.d0, eigvals(i))
    end do
end subroutine calcAM

subroutine decomposeJump(inveigvecs, dq, alpha)
    implicit none
    double precision, dimension(3,3), intent(in) :: inveigvecs
    double precision, dimension(3), intent(in) :: dq
    double precision, dimension(3), intent(out) :: alpha

    alpha = matmul(inveigvecs, dq)
end subroutine decomposeJump

subroutine calcWaves(alpha, eigvecs, wave)
    implicit none
    double precision, dimension(3), intent(in) :: alpha
    double precision, dimension(3, 3), intent(in) :: eigvecs
    double precision, dimension(3, 3), intent(out) :: wave
    integer :: i

    do i = 1, 3
        wave(:, i) = eigvecs(:, i)*alpha(i)
        !print *, i, "th wave", eigvecs(:, i)*alpha(i)
    end do
end subroutine calcWaves

subroutine calcAPDQ(inveigvecs, eigvals, eigvecs, dq, apdq)
    implicit none
    double precision, dimension(3, 3), intent(in) :: inveigvecs, eigvecs
    double precision, dimension(3), intent(in) :: dq, eigvals
    double precision, dimension(3), intent(out) :: apdq
    !amdq= matmul(matmul(inveigvecs, matmul(APM, eigvecs)), dq)
    apdq = matmul(eigvecs, max(eigvals, 0.d0)*matmul(inveigvecs, dq))
end subroutine calcAPDQ

subroutine calcAMDQ(inveigvecs, eigvals, eigvecs, dq, amdq)
    implicit none
    double precision, dimension(3, 3), intent(in) :: inveigvecs, eigvecs
    double precision, dimension(3), intent(in) :: dq, eigvals
    double precision, dimension(3), intent(out) :: amdq
    !amdq= matmul(matmul(inveigvecs, matmul(APM, eigvecs)), dq)
    amdq = matmul(eigvecs, min(eigvals, 0.d0)*matmul(inveigvecs, dq))
end subroutine calcAMDQ

