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

    !double precision, dimension(3) :: rflux, lflux
    double precision, dimension(3,3) :: jacobian, eigvecs, inveigvecs
    double precision, dimension(3) :: eigvals, q_mean, dq, alpha
    double precision, dimension(2) :: alpha_mean
    double precision :: abs_alpha

    logical :: cplx

    integer :: i, m

    do i = 1, mx+2
        q_mean(:) = 0.5d0*(ql(:, i) + qr(:, i-1))
        dq(:) = ql(:, i) - qr(:, i-1)
        !alpha_mean(:) = q_mean(2:3)/q_mean(1)
        !abs_alpha = sqrt(alpha_mean(1)**2.d0 + alpha_mean(2)**2.d0)

        if ((abs(q_mean(1)) > flux_precision)) then
            if (ixy == 1) then
                !call calcJacobianX_MTM(alpha_mean, jacobian)
                call calcJacobianX(q_mean, jacobian)
                !call calcJacobianX_advection(q_mean, jacobian)
                !call directCalcEigenX3(eigvecs, eigvals, q_mean)
            else
                !call calcJacobianY_MTM(alpha_mean, jacobian)
                call calcJacobianY(q_mean, jacobian)
                !call calcJacobianY_advection(q_mean, jacobian)
                !call directCalcEigenY3(eigvecs, eigvals, q_mean)
            end if

            call calcEigen(jacobian, eigvecs, eigvals, cplx)
            ! jacobian is overwritten!
            if (cplx) then
                ! calculate flux according to lax-friedrichs
                !call numericalFluxLaxFriedrichs(ixy, qr(:, i-1), ql(:, i), s(:, i), wave(:, :, i), amdq(:, i), apdq(:, i))
                read *, q_mean
            end if

            call calcInverse(eigvecs, inveigvecs)
            call decomposeJump(inveigvecs, dq, alpha)

            !print *, jacobian
            !print *, " "
            !print *, eigvecs
            !print *, " "
            !print *, eigvals
            !print *, " "
            !print *, inveigvecs
            !print *, " "
!            if (ANY(ABS(eigvals) > 1.0)) THEN
!                print *, "eigenvalues"
!                print *, q_mean
!                print *, eigvals
!                print *, " "
!            end if

            ! limit wave speeds to -1 < s < 1
            ! eigvals = max(min(eigvals, 1.d0), -1.d0)

            !if(ANY(abs(matmul(inveigvecs, eigvecs) - reshape((/ 1., 0., 0., 0., 1., 0., 0., 0., 1. /), shape(eigvecs))) > 1.d-2)) THEN
            !    print *, "eigenvectors"
            !    print *, q_mean
            !    print *, eigvecs
            !    print *, matmul(inveigvecs, eigvecs)
            !    read *, test
            !end if

            !condition = maxval(eigvals)/minval(eigvals)
            !if (condition > 10.d0) THEN
            !    print *, condition
            !    !GO TO 111
            !end if

            call calcWaves(alpha, eigvecs, wave(:, :, i))
            s(:, i) = eigvals
            call calcAMDQ(inveigvecs, eigvals, eigvecs, dq, amdq(:, i))
            call calcAPDQ(inveigvecs, eigvals, eigvecs, dq, apdq(:, i))
!            if(any(abs(amdq(:,i)) > 1.d-3)) then
!                print *, "q_mean", q_mean
!                print *, "amdq", amdq(:,i)
!            end if
!            if(any(abs(apdq(:,i)) > 1.d-3)) then
!                print *, "q_mean", q_mean
!                print *, "apdq", apdq(:,i)
!            end if
        else
            s(:,i) = 1.d0
            wave(:, :, i) = 0.d0
            amdq(:,i) = 0.d0
            apdq(:,i) = 0.d0
        end if
    end do
end subroutine rpn2

! -------------------------------- MODEL FLUX --------------------------------

subroutine numericalFluxLaxFriedrichs(ixy, ql, qr, s, wave, amdq, apdq)
    implicit none
    integer, intent(in) :: ixy
    double precision, dimension(3), intent(in) :: ql, qr
    double precision, dimension(3), intent(out) :: s, amdq, apdq
    double precision, dimension(3,3), intent(out) :: wave
    double precision, dimension(3) :: lflux, rflux
    double precision :: flux_precision, dxdt, dydt

    common /cparam/ flux_precision
    common /gridparameters/ dxdt, dydt

    call calcFlux(ixy, ql, lflux, flux_precision)
    call calcFlux(ixy, qr, rflux, flux_precision)

    if (ixy == 1) then
        amdq = 0.5d0*((rflux - lflux) - dxdt*(qr-ql))
        apdq = 0.5d0*((rflux - lflux) + dxdt*(qr-ql))
    else
        amdq = 0.5d0*((rflux - lflux) - dydt*(qr-ql))
        apdq = 0.5d0*((rflux - lflux) + dydt*(qr-ql))
    end if

    ! use max wave speed
    s = 1.d0
    ! use no wave height, so second order reconstruction == 0
    wave = 0.d0
    wave(:, 1) = ql-qr
end subroutine numericalFluxLaxFriedrichs

subroutine calcFlux(ixy, q, fl, flux_precision)
  implicit none
  integer, intent(in) :: ixy
  double precision, dimension(3), intent(in) :: q
  double precision, dimension(3), intent(out) :: fl
  double precision, intent(in) :: flux_precision

  double precision :: a_0, a_1, abs_a, xi
  double precision :: infinity

  infinity = HUGE(infinity)


  if (abs(q(1)) < flux_precision) THEN
    fl(1) = 0.d0
    fl(2) = 0.d0
    fl(3) = 0.d0
    RETURN
  end if

  a_0 = q(2)/q(1)
  a_1 = q(3)/q(1)
  abs_a = sqrt(a_0**2.d0 + a_1**2.d0)
  if (abs_a > 1.d0) THEN
    abs_a = 1
  end if

  call calcChi(abs_a, xi)

  if (isnan(xi) .or. xi > infinity) THEN
    print *, "xi"
    print *, xi
    read *, xi
  end if

  !print *, q(1)
  !print *, q(2)
  !print *, q(3)

  !print *, ixy

  if (ixy == 1) THEN
    ! x-richtung
    fl(1) = q(2)
    fl(2) = q(1) * ( ( 1.d0 - xi ) / 2.d0 + ( q(2)**2.d0 * ( 3.d0 * xi - 1.d0 ))/(2.d0*(q(2)**2.d0+q(3)**2.d0)))
    fl(3) = q(1) * ( (q(2)*q(3)*(3.d0*xi-1.d0))/(2.d0*(q(2)**2.d0+q(3)**2.d0)))
  ELSE
    ! y-richtung
    fl(1) = q(3)
    fl(2) = q(1) * ( (q(2)*q(3)*(3.d0*xi-1.d0))/(2.d0*(q(2)**2.d0+q(3)**2.d0)))
    fl(3) = q(1) * ( (1.d0-xi)/2.d0 + (q(3)**2.d0*(3.d0*xi-1.d0))/(2.d0*(q(2)**2.d0+q(3)**2.d0)))
  end if

  !print *, fl_0
  !print *, fl_1
  !print *, fl_2

  !read *, test

  !if (isnan(xi) .or. (xi > infinity)) THEN
  !  fl_1 = 0.d0
  !  fl_2 = 0.d0
  !end if
  !if (((q(2)**2.d0+q(3)**2.d0) < flux_precision) .or. (q(1) < flux_precision)) THEN
  !  fl_1 = 0.d0
  !  fl_2 = 0.d0
  !end if

  if (isnan(fl(1)) .OR. (fl(1) > infinity)) THEN
    print *, "fl(1)"
    print *, fl(1)
    print *, xi
    print *, q(1)
    print *, q(2)
    print *, q(3)
    print *,ixy
    read *, xi
  end if
  if (isnan(fl(2)) .OR. (fl(2) > infinity)) THEN
    print *, "fl(2)"
    print *, fl(2)
    print *, xi
    print *, q(1)
    print *, q(2)
    print *, q(3)
    print *,ixy
    read *, xi
  end if
  if (isnan(fl(3)) .OR. (fl(3) > infinity)) THEN
    print *, "fl(3)"
    print *, fl(3)
    print *, xi
    print *, q(1)
    print *, q(2)
    print *, q(3)
    print *,ixy
    read *, xi
  end if
end subroutine calcFlux

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
            print *, "dchi(", m, ") = nan"
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
                print *, "nan", i, m
                print *, q
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
                print *, "nan"
                print *, q
                jacF(i,m) = 0.d0
            end if
        end do
    end do
end subroutine calcJacobianY

subroutine calcJacobianX_MTM(alpha, jacF)
    ! calculating the jacobian in x direction according to MTM p.153
    implicit none
    double precision, dimension(2), intent(in) :: alpha
    double precision, dimension(3,3), intent(out) :: jacF

    double precision :: t1, t2, t3, t4
    double precision :: chi, dchi, abs_alpha

    abs_alpha = sqrt(alpha(1)**2.d0 + alpha(2)**2.d0)
    call calcChi(abs_alpha, chi)
    call calcDChi(abs_alpha, dchi)

    t1 = (3.d0*chi - 1)/(2.d0)
    t2 = chi - abs_alpha*dchi
    t3 = (6.d0*chi - 3.d0*abs_alpha*dchi - 2.d0)/(2.d0)
    t4 = (alpha(1)**2.d0 *3.d0 *dchi)/(abs_alpha**2.d0 *2.d0) - (dchi)/(2.d0)

    jacF(1,1) = 0.d0
    jacF(1,2) = 1.d0
    jacF(1,3) = 0.d0
    jacF(2,1) = (1.d0 - t2)/(2.d0) + (alpha(1)**2.d0)/(abs_alpha**2.d0)*(3.d0*t2 - 1.d0)/(2.d0)
    jacF(2,2) = (alpha(1))/(abs_alpha)*(t4 + (2.d0/abs_alpha - (2.d0*alpha(1)**2.d0)/(abs_alpha**3.d0))*t1)
    jacF(2,3) = (alpha(2))/(abs_alpha)*(-dchi/2.d0 - (alpha(1)**2.d0)/(abs_alpha**3.d0)*t3)
    jacF(3,1) = (alpha(1)*alpha(2))/(abs_alpha**2.d0)*(3.d0*t2-1)/(2.d0)
    jacF(3,2) = (alpha(2))/(abs_alpha**2.d0)*(t1 - (alpha(1)**2.d0)/(abs_alpha**2.d0)*t3)
    jacF(3,3) = (alpha(1))/(abs_alpha**2.d0)*(t1 - (alpha(2)**2.d0)/(abs_alpha**2.d0)*t3)
end subroutine calcJacobianX_MTM

subroutine calcJacobianY_MTM(alpha, jacF)
    ! calculating the jacobian in x direction according to MTM p.153
    implicit none
    double precision, dimension(2), intent(in) :: alpha
    double precision, dimension(3,3), intent(out) :: jacF

    double precision :: t1, t2, t3, t4
    double precision :: chi, dchi, abs_alpha

    abs_alpha = sqrt(alpha(1)**2.d0 + alpha(2)**2.d0)
    call calcChi(abs_alpha, chi)
    call calcDChi(abs_alpha, dchi)

    t1 = (3.d0*chi - 1)/(2.d0)
    t2 = chi - abs_alpha*dchi
    t3 = (6.d0*chi - 3.d0*abs_alpha*dchi - 2.d0)/(2.d0)
    t4 = (alpha(2)**2.d0 *3.d0 *dchi)/(abs_alpha**2.d0 *2.d0) - (dchi)/(2.d0)

    jacF(1,1) = 0.d0
    jacF(1,2) = 0.d0
    jacF(1,3) = 1.d0
    jacF(2,1) = (alpha(1)*alpha(2))/(abs_alpha**2.d0)*(3.d0*t2-1)/(2.d0)
    jacF(2,2) = (alpha(2))/(abs_alpha**2.d0)*(t1 - (alpha(1)**2.d0)/(abs_alpha**2.d0)*t3)
    jacF(2,3) = (alpha(1))/(abs_alpha**2.d0)*(t1 - (alpha(2)**2.d0)/(abs_alpha**2.d0)*t3)
    jacF(3,1) = (1.d0 - t2)/(2.d0) + (alpha(2)**2.d0)/(abs_alpha**2.d0)*(3.d0*t2 - 1.d0)/(2.d0)
    jacF(3,2) = (alpha(1))/(abs_alpha)*(-dchi/2.d0 - (alpha(2)**2.d0)/(abs_alpha**3.d0)*t3)
    jacF(3,3) = (alpha(2))/(abs_alpha)*(t4 + (2.d0/abs_alpha - (2.d0*alpha(2)**2.d0)/(abs_alpha**3.d0))*t1)
end subroutine calcJacobianY_MTM

subroutine calcJacobianX_advection(q, jacF)
    implicit none
    double precision, dimension(3), intent(in) :: q
    double precision, dimension(3,3), intent(out) :: jacF
    jacF = 0.d0
    jacF(1,1) = 1
    jacF(2,2) = 1
    jacF(3,3) = 1
end subroutine calcJacobianX_advection

subroutine calcJacobianY_advection(q, jacF)
    implicit none
    double precision, dimension(3), intent(in) :: q
    double precision, dimension(3,3), intent(out) :: jacF
    jacF = 0.d0
    jacF(1,1) = 1
    jacF(2,2) = 1
    jacF(3,3) = 1
end subroutine calcJacobianY_advection

subroutine calcEigen(mat, evcR, evlR, cplx)
    implicit none
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
        print *, "complex eigenvalue"
        cplx = .true.
        !read *, s
    end if
    if (INFO /= 0) THEN
        print *, "DGEEV", INFO
    end if
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
    if (INFO /= 0) THEN
        print *, "DGETRF", INFO
    end if
    call DGETRI(N, inv, N, IPIV, WORK, LWORK, INFO)
    if (INFO /= 0) THEN
        print *, "DGETRI", INFO
    end if
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

! -------------------------------- WAVE LIMITATION --------------------------------

!subroutine minmod(a,b,value)
!    double precision, intent(in) :: a, b
!    double precision, intent(out) :: value
!    double precision :: r
!    r = b/a
!    value = dmax1(0.d0, dmin1(1.d0, r))
!end subroutine minmod
!
!subroutine limiter(maxm,num_eqn,num_waves,num_ghost,mx,wave,s)
!    ! copied from: https://github.com/clawpack/pyclaw/blob/master/src/pyclaw/classic/limiter.f90
!    ! and modified to fit purpose
!    implicit double precision (a-h,o-z)
!    dimension wave(num_eqn, num_waves,1-num_ghost:maxm+num_ghost)
!    dimension s(num_waves,1-num_ghost:maxm+num_ghost)
!
!
!    do 50 mw=1,num_waves
!        dotr = 0.d0
!        do 40 i = 0, mx+1
!            wnorm2 = 0.d0
!            dotl = dotr
!            dotr = 0.d0
!            do 20 m=1,3 !3
!                wnorm2 = wnorm2 + wave(m,mw,i)**2 !scalar product of wave
!                dotr = dotr + wave(m,mw,i)*wave(m,mw,i+1)
!            20 end do
!            if (i == 0) go to 40
!            if (wnorm2 == 0.d0) go to 40
!
!            if (s(mw,i) > 0.d0) then
!                call minmod(wnorm2, dotl, wlimitr)
!            else
!                call minmod(wnorm2, dotr, wlimitr)
!            endif
!            wave(:,mw,i) = wlimitr * wave(:,mw,i)
!        40 end do
!    50 end do
!
!    return
!end subroutine limiter

! -------------------------------- DIRECT EIGVAL/VEC IMPLEMTATION --------------------------------

subroutine directCalcEigenX3(eVec, eVal, q)
  double precision, dimension(3, 3), intent(out) :: eVec
  double precision, dimension(3), intent(out) :: eVal
  double precision, dimension(3), intent(in) :: q

  double precision, dimension(4, 4) :: eVec4
  double precision, dimension(4):: eVal4, q4

  q4(1:3) = q
  q4(4) = 0.
  call directCalcEigenX(eVec4, eVal4, q4)
  eVec = eVec4(1:3, 2:4)
  eVal = eVal4(2:4)
end subroutine directCalcEigenX3

subroutine directCalcEigenY3(eVec, eVal, q)
  double precision, dimension(3, 3), intent(out) :: eVec
  double precision, dimension(3), intent(out) :: eVal
  double precision, dimension(3), intent(in) :: q

  double precision, dimension(4, 4) :: eVec4
  double precision, dimension(4):: eVal4, q4

  q4(1:3) = q
  q4(4) = 0.
  call directCalcEigenY(eVec4, eVal4, q4)
  eVec = eVec4(1:3, 2:4)
  eVal = eVal4(2:4)
end subroutine directCalcEigenY3

subroutine directCalcEigenX(eVec, eVal, q)
    double precision, dimension(4, 4), intent(out) :: eVec
    double precision, dimension(4), intent(out) :: eVal
    double precision, dimension(4), intent(in) :: q

    double precision :: aAbs, a1, a2, a3

    doUBLE COMPLEX :: chi, chiprime, c1, c2, c3, c4, c5, k11, k12, k13, k14, k15, k16, kn1, n11, n12, n13

    a1 = q(2)/q(1)
    a2 = q(3)/q(1)
    a3 = q(4)/q(1)
    aAbs = (a1**2. + a2**2. + a3**2.)**.5
    chi = (0.621529 + aAbs**2*(0.348509 + aAbs**2*(-0.139318 + 0.720371*aAbs**2)))/(1.87095 + aAbs**2*(-1.32002 + aAbs**2))
    chiprime = (aAbs*(0.697018 + aAbs**2*(-0.557272 + 4.322226*aAbs**2)))/(1.87095 + aAbs**2*(-1.32002 + aAbs**2)) - (aAbs*(-2.64004 + 4*aAbs**2)*(0.621529 + aAbs**2*(0.348509 + aAbs**2*(-0.139318 + 0.720371*aAbs**2))))/(1.87095 + aAbs**2*(-1.32002 + aAbs**2))**2
    c1 = (aAbs*((aAbs*(0.697018 + aAbs**2*(-0.557272 + 4.322226*aAbs**2)))/(1.87095 + aAbs**2*(-1.32002 + aAbs**2)) - (aAbs*(-2.64004 + 4*aAbs**2)*(0.621529 + aAbs**2*(0.348509 + aAbs**2*(-0.139318 + 0.720371*aAbs**2))))/(1.87095 + aAbs**2*(-1.32002 + aAbs**2))**2))/2. + (1 - (0.621529 + aAbs**2*(0.348509 + aAbs**2*(-0.139318 + 0.720371*aAbs**2)))/(1.87095 + aAbs**2*(-1.32002 + aAbs**2)))/2.
    c2 = (-((aAbs*(0.697018 + aAbs**2*(-0.557272 + 4.322226*aAbs**2)))/(1.87095 + aAbs**2*(-1.32002 + aAbs**2))) + (aAbs*(-2.64004 + 4*aAbs**2)*(0.621529 + aAbs**2*(0.348509 + aAbs**2*(-0.139318 + 0.720371*aAbs**2))))/(1.87095 + aAbs**2*(-1.32002 + aAbs**2))**2)/2.
    c3 = (a1*((-3*aAbs*((aAbs*(0.697018 + aAbs**2*(-0.557272 + 4.322226*aAbs**2)))/(1.87095 + aAbs**2*(-1.32002 + aAbs**2)) - (aAbs*(-2.64004 + 4*aAbs**2)*(0.621529 + aAbs**2*(0.348509 + aAbs**2*(-0.139318 + 0.720371*aAbs**2))))/(1.87095 + aAbs**2*(-1.32002 + aAbs**2))**2))/2. + (-1 + (3*(0.621529 + aAbs**2*(0.348509 + aAbs**2*(-0.139318 + 0.720371*aAbs**2))))/(1.87095 + aAbs**2*(-1.32002 + aAbs**2)))/2.))/aAbs
    c4 = (a1*(1 - (3*(0.621529 + aAbs**2*(0.348509 + aAbs**2*(-0.139318 + 0.720371*aAbs**2))))/(1.87095 + aAbs**2*(-1.32002 + aAbs**2)) + (3*aAbs*((aAbs*(0.697018 + aAbs**2*(-0.557272 + 4.322226*aAbs**2)))/(1.87095 + aAbs**2*(-1.32002 + aAbs**2)) - (aAbs*(-2.64004 + 4*aAbs**2)*(0.621529 + aAbs**2*(0.348509 + aAbs**2*(-0.139318 + 0.720371*aAbs**2))))/(1.87095 + aAbs**2*(-1.32002 + aAbs**2))**2))/2.))/aAbs**2
    c5 = (-1 + (3*(0.621529 + aAbs**2*(0.348509 + aAbs**2*(-0.139318 + 0.720371*aAbs**2))))/(1.87095 + aAbs**2*(-1.32002 + aAbs**2)))/(2.*aAbs)

    k11 = a2**2*C4 + a3**2*C4 + a1*C5
    k12 = -(a2**2*C2*C3) - a3**2*C2*C3 + a2**2*C1*C4 + a3**2*C1*C4
    k13 = a1*C1*C5 + a1**2*C3*C5
    k14 = -C1 - a1*C3 + a1**2*C2*C5 - a2**2*C2*C5 - a3**2*C2*C5 + a1**3*C4*C5 + a1*a2**2*C4*C5 + a1*a3**2*C4*C5 + 2*a1**2*C5**2
    k15 = -(a1*C2) - a1**2*C4 - a2**2*C4 - a3**2*C4 - 3*a1*C5
    k16 = a1*C4 + C5
    kn1 = -27*K12 - 27*K13 + 9*K14*K15 - 2*K15**3 + Sqrt(4*(3*K14 - K15**2)**3 + (-27*K12 - 27*K13 + 9*K14*K15 - 2*K15**3)**2)
    n11 = -K15/3. - (2**0.3333333333333333*(3*K14 - K15**2))/(3.*KN1**0.3333333333333333) + KN1**0.3333333333333333/(3.*2**0.3333333333333333)
    n12 = -K15/3. + ((1 + (0,1)*Sqrt(3.))*(3*K14 - K15**2))/(3.*2**0.6666666666666666*KN1**0.3333333333333333) - ((1 - (0,1)*Sqrt(3.))*KN1**0.3333333333333333)/(6.*2**0.3333333333333333)
    n13 = -K15/3. + ((1 - (0,1)*Sqrt(3.))*(3*K14 - K15**2))/(3.*2**0.6666666666666666*KN1**0.3333333333333333) - ((1 + (0,1)*Sqrt(3.))*KN1**0.3333333333333333)/(6.*2**0.3333333333333333)

    eVal(1) = REAL(a1*c5)
    eVal(2) = REAL(n11)
    eVal(3) = REAL(n12)
    eVal(4) = REAL(n13)

    eVec(1,1) = REAL(0)
    eVec(1,2) = REAL(-(((K11 - N11)*(C2*C3 - C1*C4 + (C2 - a1*C4)*C5*N11 + C4*N11**2))/(C3 + K16*N11)))
    eVec(1,3) = REAL(-(((K11 - N12)*(C2*C3 - C1*C4 + (C2 - a1*C4)*C5*N12 + C4*N12**2))/(C3 + K16*N12)))
    eVec(1,4) = REAL(-(((K11 - N13)*(C2*C3 - C1*C4 + (C2 - a1*C4)*C5*N13 + C4*N13**2))/(C3 + K16*N13)))
    eVec(2,1) = REAL(0)
    eVec(2,2) = REAL(-(((K11 - N11)*N11*(C2*C3 - C1*C4 + (C2 - a1*C4)*C5*N11 + C4*N11**2))/(C3 + K16*N11)))
    eVec(2,3) = REAL(-(((K11 - N12)*N12*(C2*C3 - C1*C4 + (C2 - a1*C4)*C5*N12 + C4*N12**2))/(C3 + K16*N12)))
    eVec(2,4) = REAL(-(((K11 - N13)*N13*(C2*C3 - C1*C4 + (C2 - a1*C4)*C5*N13 + C4*N13**2))/(C3 + K16*N13)))
    eVec(3,1) = REAL(-(a3/a2))
    eVec(3,2) = REAL(a2*(C2*C3 - C1*C4 + (C2 - a1*C4)*C5*N11 + C4*N11**2))
    eVec(3,3) = REAL(a2*(C2*C3 - C1*C4 + (C2 - a1*C4)*C5*N12 + C4*N12**2))
    eVec(3,4) = REAL(a2*(C2*C3 - C1*C4 + (C2 - a1*C4)*C5*N13 + C4*N13**2))
    eVec(4,1) = REAL(1)
    eVec(4,2) = REAL(a3*(C2*C3 - C1*C4 + (C2 - a1*C4)*C5*N11 + C4*N11**2))
    eVec(4,3) = REAL(a3*(C2*C3 - C1*C4 + (C2 - a1*C4)*C5*N12 + C4*N12**2))
    eVec(4,4) = REAL(a3*(C2*C3 - C1*C4 + (C2 - a1*C4)*C5*N13 + C4*N13**2))
end subroutine directCalcEigenX

subroutine directCalcEigenY(eVec, eVal, q)
    double precision, dimension(4, 4), intent(out) :: eVec
    double precision, dimension(4), intent(out) :: eVal
    double precision, dimension(4), intent(in) :: q

    double precision :: aAbs, a1, a2, a3

    doUBLE COMPLEX :: chi, chiprime, c1, c2, c3, c4, c5, k21, k22, k23, k24, k25, k26, kn2, n21, n22, n23, k28

    a1 = q(2)/q(1)
    a2 = q(3)/q(1)
    a3 = q(4)/q(1)
    aAbs = (a1**2. + a2**2. + a3**2.)**.5
    chi = (0.621529 + aAbs**2*(0.348509 + aAbs**2*(-0.139318 + 0.720371*aAbs**2)))/(1.87095 + aAbs**2*(-1.32002 + aAbs**2))
    chiprime = (aAbs*(0.697018 + aAbs**2*(-0.557272 + 4.322226*aAbs**2)))/(1.87095 + aAbs**2*(-1.32002 + aAbs**2)) - (aAbs*(-2.64004 + 4*aAbs**2)*(0.621529 + aAbs**2*(0.348509 + aAbs**2*(-0.139318 + 0.720371*aAbs**2))))/(1.87095 + aAbs**2*(-1.32002 + aAbs**2))**2

    c1 = (aAbs*((aAbs*(0.697018 + aAbs**2*(-0.557272 + 4.322226*aAbs**2)))/(1.87095 + aAbs**2*(-1.32002 + aAbs**2)) - (aAbs*(-2.64004 + 4*aAbs**2)*(0.621529 + aAbs**2*(0.348509 + aAbs**2*(-0.139318 + 0.720371*aAbs**2))))/(1.87095 + aAbs**2*(-1.32002 + aAbs**2))**2))/2.+ (1 - (0.621529 + aAbs**2*(0.348509 + aAbs**2*(-0.139318 + 0.720371*aAbs**2)))/(1.87095 + aAbs**2*(-1.32002 + aAbs**2)))/2.
    c2 = (-((aAbs*(0.697018 + aAbs**2*(-0.557272 + 4.322226*aAbs**2)))/(1.87095 + aAbs**2*(-1.32002 + aAbs**2))) + (aAbs*(-2.64004 + 4*aAbs**2)*(0.621529 + aAbs**2*(0.348509 + aAbs**2*(-0.139318 + 0.720371*aAbs**2))))/(1.87095 + aAbs**2*(-1.32002 + aAbs**2))**2)/2.
    c3 =  (a2*((-3*aAbs*((aAbs*(0.697018 + aAbs**2*(-0.557272 + 4.322226*aAbs**2)))/(1.87095 + aAbs**2*(-1.32002 + aAbs**2)) - (aAbs*(-2.64004 + 4*aAbs**2)*(0.621529 + aAbs**2*(0.348509 + aAbs**2*(-0.139318 + 0.720371*aAbs**2))))/(1.87095 + aAbs**2*(-1.32002 + aAbs**2))**2))/2. + (-1 + (3*(0.621529 + aAbs**2*(0.348509 + aAbs**2*(-0.139318 + 0.720371*aAbs**2))))/(1.87095 + aAbs**2*(-1.32002 + aAbs**2)))/2.))/aAbs
    c4 = (a2*(1 - (3*(0.621529 + aAbs**2*(0.348509 + aAbs**2*(-0.139318 + 0.720371*aAbs**2))))/(1.87095 + aAbs**2*(-1.32002 + aAbs**2)) + (3*aAbs*((aAbs*(0.697018 + aAbs**2*(-0.557272 + 4.322226*aAbs**2)))/(1.87095 + aAbs**2*(-1.32002 + aAbs**2)) - (aAbs*(-2.64004 + 4*aAbs**2)*(0.621529 + aAbs**2*(0.348509 + aAbs**2*(-0.139318 + 0.720371*aAbs**2))))/(1.87095 + aAbs**2*(-1.32002 + aAbs**2))**2))/2.))/aAbs**2
    c5 = (-1 + (3*(0.621529 + aAbs**2*(0.348509 + aAbs**2*(-0.139318 + 0.720371*aAbs**2))))/(1.87095 + aAbs**2*(-1.32002 + aAbs**2)))/(2.*aAbs)

    k21 = C2 + a2*C4
    k22 = -(a1**2*C2*C3) - a3**2*C2*C3 + a1**2*C1*C4 + a3**2*C1*C4
    k23 = a2*C1*C5 + a2**2*C3*C5
    k24 = -C1 - a2*C3 - a1**2*C2*C5 + a2**2*C2*C5 - a3**2*C2*C5 + a1**2*a2*C4*C5 + a2**3*C4*C5 + a2*a3**2*C4*C5 + 2*a2**2*C5**2
    k25 = -(a2*C2) - a1**2*C4 - a2**2*C4 - a3**2*C4 - 3*a2*C5
    k26 = C2*C3 - C1*C4
    kn2 = -27*K22 - 27*K23 + 9*K24*K25 - 2*K25**3 + Sqrt(4*(3*K24 - K25**2)**3 + (-27*K22 - 27*K23 + 9*K24*K25 - 2*K25**3)**2)
    n21 = -K25/3. - (2**0.3333333333333333*(3*K24 - K25**2))/(3.*KN2**0.3333333333333333) + KN2**0.3333333333333333/(3.*2**0.3333333333333333)
    n22 = -K25/3. + ((1 + (0,1)*Sqrt(3.))*(3*K24 - K25**2))/(3.*2**0.6666666666666666*KN2**0.3333333333333333) - ((1 - (0,1)*Sqrt(3.))*KN2**0.3333333333333333)/(6.*2**0.3333333333333333)
    n23 = -K25/3. + ((1 - (0,1)*Sqrt(3.))*(3*K24 - K25**2))/(3.*2**0.6666666666666666*KN2**0.3333333333333333) - ((1 + (0,1)*Sqrt(3.))*KN2**0.3333333333333333)/(6.*2**0.3333333333333333)
    k28 = (C2 - a2*C4)*C5

    eVal(1) = REAL(a2*C5)
    eVal(2) = REAL(n21)
    eVal(3) = REAL(n22)
    eVal(4) = REAL(n23)

    eVec(1,1) = REAL(0)
    eVec(1,2) = REAL(-(K21*(a2*C5 - N21)))
    eVec(1,3) = REAL(-(K21*(a2*C5 - N22)))
    eVec(1,4) = REAL(-(K21*(a2*C5 - N23)))
    eVec(2,1) = REAL(-(a3/a1))
    eVec(2,2) = REAL(a1*(K26 + K28*N21 + C4*N21**2))
    eVec(2,3) = REAL(a1*(K26 + K28*N22 + C4*N22**2))
    eVec(2,4) = REAL(a1*(K26 + K28*N23 + C4*N23**2))
    eVec(3,1) = REAL(0)
    eVec(3,2) = REAL(-(K21*(a2*C5 - N21)*N21))
    eVec(3,3) = REAL(-(K21*(a2*C5 - N22)*N22))
    eVec(3,4) = REAL(-(K21*(a2*C5 - N23)*N23))
    eVec(4,1) = REAL(1)
    eVec(4,2) = REAL(a3*(K26 + K28*N21 + C4*N21**2))
    eVec(4,3) = REAL(a3*(K26 + K28*N22 + C4*N22**2))
    eVec(4,4) = REAL(a3*(K26 + K28*N23 + C4*N23**2))
end subroutine directCalcEigenY

subroutine directCalcEigenZ(eVec, eVal, q)
    double precision, dimension(4, 4), intent(out) :: eVec
    double precision, dimension(4), intent(out) :: eVal
    double precision, dimension(4), intent(in) :: q

    double precision :: aAbs, a1, a2, a3

    doUBLE COMPLEX :: chi, chiprime, c1, c2, c3, c4, c5, k31, k32, k33, k34, k35, k36, kn3, n31, n32, n33, k38

    a1 = q(2)/q(1)
    a2 = q(3)/q(1)
    a3 = q(4)/q(1)
    aAbs = (a1**2. + a2**2. + a3**2.)**.5
    chi = (0.621529 + aAbs**2*(0.348509 + aAbs**2*(-0.139318 + 0.720371*aAbs**2)))/(1.87095 + aAbs**2*(-1.32002 + aAbs**2))
    chiprime = (aAbs*(0.697018 + aAbs**2*(-0.557272 + 4.322226*aAbs**2)))/(1.87095 + aAbs**2*(-1.32002 + aAbs**2)) - (aAbs*(-2.64004 + 4*aAbs**2)*(0.621529 + aAbs**2*(0.348509 + aAbs**2*(-0.139318 + 0.720371*aAbs**2))))/(1.87095 + aAbs**2*(-1.32002 + aAbs**2))**2

    c1 = (aAbs*((aAbs*(0.697018 + aAbs**2*(-0.557272 + 4.322226*aAbs**2)))/(1.87095 + aAbs**2*(-1.32002 + aAbs**2)) - (aAbs*(-2.64004 + 4*aAbs**2)*(0.621529 + aAbs**2*(0.348509 + aAbs**2*(-0.139318 + 0.720371*aAbs**2))))/(1.87095 + aAbs**2*(-1.32002 + aAbs**2))**2))/2. + (1 - (0.621529 + aAbs**2*(0.348509 + aAbs**2*(-0.139318 + 0.720371*aAbs**2)))/(1.87095 + aAbs**2*(-1.32002 + aAbs**2)))/2.
    c2 = (-((aAbs*(0.697018 + aAbs**2*(-0.557272 + 4.322226*aAbs**2)))/(1.87095 + aAbs**2*(-1.32002 + aAbs**2))) + (aAbs*(-2.64004 + 4*aAbs**2)*(0.621529 + aAbs**2*(0.348509 + aAbs**2*(-0.139318 + 0.720371*aAbs**2))))/(1.87095 + aAbs**2*(-1.32002 + aAbs**2))**2)/2.
    c3 = (a3*((-3*aAbs*((aAbs*(0.697018 + aAbs**2*(-0.557272 + 4.322226*aAbs**2)))/(1.87095 + aAbs**2*(-1.32002 + aAbs**2)) - (aAbs*(-2.64004 + 4*aAbs**2)*(0.621529 + aAbs**2*(0.348509 + aAbs**2*(-0.139318 + 0.720371*aAbs**2))))/(1.87095 + aAbs**2*(-1.32002 + aAbs**2))**2))/2. + (-1 + (3*(0.621529 + aAbs**2*(0.348509 + aAbs**2*(-0.139318 + 0.720371*aAbs**2))))/(1.87095 + aAbs**2*(-1.32002 + aAbs**2)))/2.))/aAbs
    c4 = (a3*(1 - (3*(0.621529 + aAbs**2*(0.348509 + aAbs**2*(-0.139318 + 0.720371*aAbs**2))))/(1.87095 + aAbs**2*(-1.32002 + aAbs**2)) + (3*aAbs*((aAbs*(0.697018 + aAbs**2*(-0.557272 + 4.322226*aAbs**2)))/(1.87095 + aAbs**2*(-1.32002 + aAbs**2)) - (aAbs*(-2.64004 + 4*aAbs**2)*(0.621529 + aAbs**2*(0.348509 + aAbs**2*(-0.139318 + 0.720371*aAbs**2))))/(1.87095 + aAbs**2*(-1.32002 + aAbs**2))**2))/2.))/aAbs**2
    c5 = (-1 + (3*(0.621529 + aAbs**2*(0.348509 + aAbs**2*(-0.139318 + 0.720371*aAbs**2))))/(1.87095 + aAbs**2*(-1.32002 + aAbs**2)))/(2.*aAbs)

    k31 = C2 + a3*C4
    k32 = -(a1**2*C2*C3) - a2**2*C2*C3 + a1**2*C1*C4 + a2**2*C1*C4
    k33 = a3*C1*C5 + a3**2*C3*C5
    k34 = -C1 - a3*C3 - a1**2*C2*C5 - a2**2*C2*C5 + a3**2*C2*C5 + a1**2*a3*C4*C5 + a2**2*a3*C4*C5 + a3**3*C4*C5 + 2*a3**2*C5**2
    k35 = -(a3*C2) - a1**2*C4 - a2**2*C4 - a3**2*C4 - 3*a3*C5
    k36 = C2*C3 - C1*C4
    kn3 = -27*K32 - 27*K33 + 9*K34*K35 - 2*K35**3 + Sqrt(4*(3*K34 - K35**2)**3 + (-27*K32 - 27*K33 + 9*K34*K35 - 2*K35**3)**2)
    n31 = -K35/3. - (2**0.3333333333333333*(3*K34 - K35**2))/(3.*KN3**0.3333333333333333) + KN3**0.3333333333333333/(3.*2**0.3333333333333333)
    n32 = -K35/3. + ((1 + (0,1)*Sqrt(3.))*(3*K34 - K35**2))/(3.*2**0.6666666666666666*KN3**0.3333333333333333) - ((1 - (0,1)*Sqrt(3.))*KN3**0.3333333333333333)/(6.*2**0.3333333333333333)
    n33 = -K35/3. + ((1 - (0,1)*Sqrt(3.))*(3*K34 - K35**2))/(3.*2**0.6666666666666666*KN3**0.3333333333333333) - ((1 + (0,1)*Sqrt(3.))*KN3**0.3333333333333333)/(6.*2**0.3333333333333333)
    k38 = (C2 - a3*C4)*C5

    eVal(1) = REAL(a3*C5)
    eVal(2) = REAL(N31)
    eVal(3) = REAL(N32)
    eVal(4) = REAL(N33)

    eVec(1,1) = REAL(0)
    eVec(1,2) = REAL(K31*(a3*C5 - N31))
    eVec(1,3) = REAL(K31*(a3*C5 - N32))
    eVec(1,4) = REAL(K31*(a3*C5 - N33))
    eVec(2,1) = REAL(-(a2/a1))
    eVec(2,2) = REAL(-(a1*(K36 + K38*N31 + C4*N31**2)))
    eVec(2,3) = REAL(-(a1*(K36 + K38*N32 + C4*N32**2)))
    eVec(2,4) = REAL(-(a1*(K36 + K38*N33 + C4*N33**2)))
    eVec(3,1) = REAL(1)
    eVec(3,2) = REAL(-(a2*(K36 + K38*N31 + C4*N31**2)))
    eVec(3,3) = REAL(-(a2*(K36 + K38*N32 + C4*N32**2)))
    eVec(3,4) = REAL(-(a2*(K36 + K38*N33 + C4*N33**2)))
    eVec(4,1) = REAL(0)
    eVec(4,2) = REAL(K31*(a3*C5 - N31)*N31)
    eVec(4,3) = REAL(K31*(a3*C5 - N32)*N32)
    eVec(4,4) = REAL(K31*(a3*C5 - N33)*N33)
end subroutine directCalcEigenZ
