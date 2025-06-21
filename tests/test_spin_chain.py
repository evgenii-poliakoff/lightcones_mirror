import numpy as np
import pytest
import lightcones.linalg as ll
import lightcones.space as sp
from lightcones import models

tol = 1e-10

def test_spin_chain():

    # length of chain
    length = 5

    # where to truncate the spin chain Hilbert space
    max_num_flips = 2**5

    # the truncated Hilbert space
    m = models.spin_chain(length, max_num_flips)

    # operators
    s_x = m.s_x
    s_y = m.s_y
    s_z = m.s_z

    # transverse field
    h = 0.5
    j_x = 0.7
    j_y = 0.9

    H_ising = sum([h * s_z[i] for i in range(length)]) \
        + j_x * sum([s_x[i] @ s_x[i + 1] for i in range(length - 1)]) \
        + j_y * sum([s_x[i] @ s_x[i + 1] for i in range(length - 1)])
    
    # find ground state
    [eigs, eigvecs] = ll.find_eigs_ascending(H_ising.todense())

    # check eigenenergies
    eigs_expected = np.array([-6.67993205281208, -6.67132629481462, -4.17797599098542,
       -4.16937023298794, -3.57316033464694, -3.56455457664944,
       -3.00195606182666, -2.99335030382916, -2.61537747616266,
       -2.60677171816516, -1.07120427282027, -1.06259851482279,
       -0.5             , -0.4913942420025 , -0.113421414336  ,
       -0.1048156563385 ,  0.10481565633851,  0.113421414336  ,
        0.4913942420025 ,  0.5             ,  1.06259851482279,
        1.07120427282028,  2.60677171816516,  2.61537747616266,
        2.99335030382915,  3.00195606182665,  3.56455457664944,
        3.57316033464694,  4.16937023298794,  4.17797599098544,
        6.6713262948146 ,  6.6799320528121 ])
    
    assert np.allclose(eigs, eigs_expected, atol=tol), \
        f"eigs does not match"