!subroutine local_op(a_data, n_data, a_ind, n_ind, a_ptr, n_ptr, b_data, m_data, b_ind, m_ind, b_ptr, m_ptr, o_data, l_data, o_ind, l_ind, o_ptr, l_ptr, o, K, ket_index, bra_index)
subroutine local_op(a_data, n_data, a_ind, n_ind, a_ptr, n_ptr, b_data, m_data, b_ind, m_ind, b_ptr, m_ptr, o_data, l_data, o_ind, l_ind, o, K, ket_index, bra_index)
    implicit none
    
    complex*16, intent(inout), dimension(n_data) :: a_data
    integer :: n_data
    !f2py intent(in,out,overwrite) a_data
    !f2py integer intent(hide), depend(a_data) :: n_data = len(a_data)
    
    integer, intent(inout), dimension(n_ind) :: a_ind
    integer :: n_ind
    !f2py intent(in,out,overwrite) a_ind
    !f2py integer intent(hide), depend(a_ind) :: n_ind = len(a_ind)
    
    integer, intent(inout), dimension(n_ptr) :: a_ptr
    integer :: n_ptr
    !f2py intent(in,out,overwrite) a_ptr
    !f2py integer intent(hide), depend(a_ptr) :: n_ptr = len(a_ptr)
    
    !!!
    
    complex*16, intent(inout), dimension(m_data) :: b_data
    integer :: m_data
    !f2py intent(in,out,overwrite) b_data
    !f2py integer intent(hide), depend(b_data) :: m_data = len(b_data)
    
    integer, intent(inout), dimension(m_ind) :: b_ind
    integer :: m_ind
    !f2py intent(in,out,overwrite) b_ind
    !f2py integer intent(hide), depend(b_ind) :: m_ind = len(b_ind)
    
    integer, intent(inout), dimension(m_ptr) :: b_ptr
    integer :: m_ptr
    !f2py intent(in,out,overwrite) b_ptr
    !f2py integer intent(hide), depend(b_ptr) :: m_ptr = len(b_ptr)
    
    !!!
    
    complex*16, intent(inout), dimension(l_data) :: o_data
    integer :: l_data
    !f2py intent(in,out,overwrite) o_data
    !f2py integer intent(hide), depend(o_data) :: l_data = len(o_data)
    
    integer, intent(inout), dimension(l_ind) :: o_ind
    integer :: l_ind
    !f2py intent(in,out,overwrite) o_ind
    !f2py integer intent(hide), depend(o_ind) :: l_ind = len(o_ind)
    
    !integer, intent(inout), dimension(l_ptr) :: o_ptr
    !integer :: l_ptr
    !! f2py intent(in,out,overwrite) o_ptr
    !! f2py integer intent(hide), depend(o_ptr) :: l_ptr = len(o_ptr)
    
    !!!!!!
    
    integer, intent(inout), dimension(K) :: o
    integer :: K
    !f2py intent(in,out,overwrite) o
    !f2py integer intent(hide), depend(o) :: K = len(o)
    
    !!!!!!
    
    integer, intent(in) :: ket_index, bra_index
    
    !!!!!!
    
    integer :: m, n, i, j, ptr, p
    complex*16 :: melem
    
    !print *, '*************'
    !print *, ket_index
    !print *, bra_index
    !print *, '-------------'

    !j = o_ptr(1)

    do j = 1, K
    
        ! assuming o_ptr is always as [0 ... K-1]
        
        n = o(j)
            
        melem = 0d0

        p = j

        if (n .eq. bra_index) then

            melem = 1d0
                   
            m = ket_index - bra_index
                
            if (m > 0) then
                    
                do i = 1, m
                        
                    !s = create(s, target_mode + 1) * (1d0 / sqrt(1d0 * (bra_index + i)))
                        
                    ptr = b_ptr(p) + 1
                        
                    if (ptr > b_ptr(p + 1)) then
                        
                        melem = 0
                        exit
                        
                    end if
                        
                    melem = melem * sign(1d0, real(b_data(ptr)))
                    
                    if (b_data(ptr) .eq. 0) then
                        melem = 0
                    end if
                        
                    if (melem .eq. 0) then
                        exit
                    end if
                        
                    p = b_ind(ptr) + 1
                        
                end do !i
                    
            end if ! m>0
                
            if (m < 0) then
                    
                do i = 0, m + 1, -1
                        
                    ptr = a_ptr(p) + 1
                        
                    if (ptr > a_ptr(p + 1)) then
                        
                        melem = 0
                        exit
                        
                    end if
                        
                    melem = melem * sign(1d0, real(a_data(ptr)))
                    
                    if (a_data(ptr) .eq. 0) then
                        melem = 0
                    end if
                    
                    if (melem .eq. 0) then
                        exit
                    end if
                        
                    p = a_ind(ptr) + 1
                        !s = annihilate(s, target_mode + 1) * (1d0 / sqrt(1d0 * (bra_index + i)))
                        
                end do !i
                    
            end if ! m<0
                    
            !call store(s, output)
                
        end if !(n .eq. bra_index) then

        !print *, '------------'
        !print *, j
        !print *, p
        !print *, melem

        o_ind(j) = p - 1
                
        if (melem .eq. 0) then
                
            o_data(j) = 0
                
        else
            
            o_data(j) = melem

     !       print *, j
            
        end if !melem

       
            
    end do
    

    !print *, o_ind
    
    !!!!!!
    
    
end subroutine local_op
