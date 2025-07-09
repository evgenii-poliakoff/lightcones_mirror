subroutine fastmuli(a_data, n_data, a_ind, n_ind, a_ptr, n_ptr, cin, vin,  n_vin, cout, vout, length)
    implicit none
    
    complex*16, intent(inout), dimension(n_data) :: a_data
    integer :: n_data
    !f2py intent(in,out,overwrite) a_data
    !f2py integer intent(hide), depend(a_data) :: n_data = len(a_data)
    
    integer, intent(inout), dimension(n_ind) :: a_ind
    integer :: n_ind
    !f2py intent(in,out,overwrite) a_ind
    !f2py integer intent(hide), depend(a_ind) :: n_ind = len(a_ind)
    
    integer, intent(out), dimension(n_ptr) :: a_ptr
    integer :: n_ptr
    !f2py intent(in,out,overwrite) a_ptr
    !f2py integer intent(hide), depend(a_ptr) :: n_ptr = len(a_ptr)
    
    complex*16, intent(in) :: cin, cout
    
    complex*16, intent(inout), dimension(n_vin) :: vin
    integer :: n_vin
    !f2py intent(in,out,overwrite) vin
    !f2py integer intent(hide), depend(vin) :: n_vin = len(vin)
    
    integer :: length
    
    complex*16, intent(inout), dimension(n_vin) :: vout
    !f2py intent(in,out,overwrite) vout
    
    integer :: i, j, k
    complex*16 :: vd, md
    
    if (cout .eq. (0d0, 0d0)) then
    
        vout = 0d0
    
    else 
    
        if (cout .ne. (1d0, 0d0)) then

            vout(1:length) = cout * vout(1:length)
            
        end if
    
    end if
    

    
    if (cin .ne. (1d0, 0d0)) then
    
        do i = 1, length
            vd = vin(i)
            do j = a_ptr(i) + 1, a_ptr(i + 1)
                k = a_ind(j) + 1
                md = a_data(j)
                vout(k) = vout(k) + cin * md * vd
            end do
        end do
    
    else
    
        do i = 1, length
            vd = vin(i)
            do j = a_ptr(i) + 1, a_ptr(i + 1)
                k = a_ind(j) + 1
                md = a_data(j)
                vout(k) = vout(k) + md * vd
            end do
        end do
    
    end if
    
end subroutine fastmuli
