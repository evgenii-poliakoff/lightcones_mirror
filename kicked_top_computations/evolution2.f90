subroutine evolution2(a, b, dt, begin_step, apply_H, o, psi_in, n_psi, psi, psi_mid, psi_mid_next)

    implicit none
    
    integer, intent(in) :: a, b
    real*8, intent(in) :: dt
    
    complex*16, intent(inout), dimension(n_psi) :: psi_in
    integer :: n_psi
    !f2py intent(in,out,overwrite) psi_in
    !f2py integer intent(hide), depend(psi_in) :: n_psi = len(psi_in)
    
    complex*16, intent(inout), dimension(n_psi) :: psi
    !f2py intent(in,out,overwrite) psi
    
    complex*16, intent(inout), dimension(n_psi) :: psi_mid
    !f2py intent(in,out,overwrite) psi_mid
    
    complex*16, intent(inout), dimension(n_psi) :: psi_mid_next
    !f2py intent(in,out,overwrite) psi_mid_next
    
    real*8 :: tol, err
    
    integer :: cont 

    external begin_step
    external apply_H
    external o

    !integer :: o
    
    integer :: i
    
    tol = dt**3
    
    psi = psi_in
    
    !cont = o(a, psi, n_psi)
    
    !if (cont .ne. 0) then
    !    return
    !end if
    
    call o(a, psi, n_psi)
    
    do i = a, b - 2
    
        call begin_step(i)
    
        psi_mid = psi
        
        do while(.true.)
        
            psi_mid_next = 0d0
        
            call apply_H(i, psi_mid, psi_mid_next, n_psi)
            psi_mid_next = psi - (0d0, 1d0) * dt / 2 * psi_mid_next
        
            err = sum(abs(psi_mid_next - psi_mid))
    
            psi_mid = psi_mid_next
        
            if (err < tol) then
                exit
            end if
        
        end do
    
        psi = 2 * psi_mid - psi
        
        !cont = o(i + 1, psi, n_psi)
        
        !if (cont .ne. 0) then
        !    exit
        !end if
    
        call o(i + 1, psi, n_psi)
    
    end do
    
end subroutine evolution2
