import numpy as np
from scipy.linalg import eig
from scipy.linalg import expm
from typing import Optional, Tuple, List
import lightcones.linalg as ll

def minimal_forward(spread: np.ndarray[np.complex128], rho_plus: np.ndarray[np.complex128], dt: float, rtol: float) -> Tuple[List[int], np.ndarray[np.complex128], np.ndarray[np.complex128], np.ndarray[np.complex128]]:
    """
    Construct a minimal forward frame and identify causal mode arrival times.

    Implements the "minimal light cone" algorithm: finds a unitary U_min that localizes quantum 
    information propagation by iteratively pruning modes that fall outside a dynamically determined
    light cone boundary, creating a minimal basis for representing the quantum dynamics.
    
    The algorithm proceeds in two steps:
    1. Forward pass: identifies all significant modes (λ_i ≥ rtol·λ_max) over
       the entire time evolution from the ρ₊ eigendecomposition.
    2. Backward pass: propagates ρ₊ backward, removing modes that fall below 
       rtol·λ_max(t) at each step and recording their "arrival times".

    Parameters:
    ----------
    spread : numpy.ndarray of numpy.complex128
        2D array of shape (n_sites, nt) containing spread coefficients α[k,l].
    rho_plus : numpy.ndarray of numpy.complex128  
        2D array of shape (n_sites, n_sites) containing the retarded density matrix ρ₊.
        Typically computed by rho_plus(spread, dt). Must be Hermitian positive semi-definite.   
    dt : float
        Time step size used in computing spread.
    rtol : float
        Relative tolerance threshold for the light cone boundary. Modes with eigenvalues
        λ_i < rtol·λ_max are considered outside the light cone and decoupled.
        Typical values: 1e-3 to 1e-6 depending on desired accuracy.

    Returns:
    -------
    times_in : list of int
        Arrival times for modes entering the minimal light cone.
        The list has length (n_modes + 1) where n_modes is the number of modes in the
        minimal frame. 
        times_in[i] is the time step when mode i enters, and 
        times_in[-1] = nt (total number of time steps).  
    spread_min : numpy.ndarray of numpy.complex128
        2D array of shape (n_sites, nt) containing the spread coefficients in the
        minimal frame basis.
    U_min : numpy.ndarray of numpy.complex128
        Unitary matrix of shape (n_sites, n_sites) representing the transformation to the
        minimal frame. The first n_modes columns correspond to modes in the minimal light cone,
        ordered by their arrival times. 
    rho_plus_min : numpy.ndarray of numpy.complex128
        Retarded density matrix ρ₊ transformed to the minimal frame.
        The matrix has size n_sites x n_sites, with the first n_modes x n_modes block 
        corresponding to modes inside the minimal light cone.
    
    Notes:
    -----
    - The "minimal light cone" represents the smallest set of modes needed to describe
      quantum information propagation within tolerance rtol.
    - Modes are ordered by arrival time.
    - The backward propagation ensures that once a mode exits the light cone, it never
      re-enters, maintaining causality.

    See Also:
    --------
    spread : Computes the time-evolved spread coefficients α[k, l].
    rho_plus : Compute the retarded density matrix ρ₊
    """
    
    # find the modes which manage to couple before the end of the
    # time interval
    pi, U_rel = ll.find_largest_eigs(rho_plus)
    g_metric = pi - rtol * pi[0]
    inside_lightcone = g_metric > 0
    n_rel =  sum(inside_lightcone)
    
    # begin to construct the rotation U_min
    U_min = U_rel.T.conj()
    # also recompute the spread
    spread_min = U_min @ spread
    # and rho_plus 
    rho_plus_min = np.diag(pi[: n_rel].astype('cdouble'))

    # number of time steps
    ntg = np.size(spread, 1)

    # Here we store the arrival times
    times_in = []
    
    # Number of non-optimal modes
    # (which we continue to transform)
    n = n_rel
        
    # Propagate backwards in time
    for i in reversed(range(0, ntg)):
        pi_min, _ = ll.find_smallest_eigs(rho_plus_min, 1)
        pi_max, _ = ll.find_largest_eigs(rho_plus_min, 1)
        g_metric = pi_min - rtol * pi_max
        outside_lightcone = g_metric < 0

        if outside_lightcone:
            pi, U = ll.find_eigs_descending(rho_plus_min)
            spread_min[: n, :] = U.T.conj() @ spread_min[: n, :]
            U_min[: n, :] = U.T.conj() @ U_min[: n, :]
            rho_plus_min = np.diag(pi[: -1].astype('cdouble'))
            times_in.insert(0, i + 1)
            n = n - 1

        psi = ll.as_column_vector(spread_min[: n, i])
        rho_plus_min -= ll.dyad(psi, psi) * dt
        ll.make_hermitean(rho_plus_min)
        
    if n > 0:
        for i in range(n):
            times_in.insert(0, 0)
    times_in.append(ntg)
    
    rho_plus_min = U_min @ rho_plus @ U_min.T.conj()
    
    return times_in, spread_min, U_min, rho_plus_min 

def m_in(times_in: List[int], ti: int) -> int:
    """
    Determine the number of modes that have entered the minimal light cone by a given time `ti`.

    Given the arrival times `times_in` produced by minimal_forward_frame(),
    this function returns the count of modes whose arrival time is ≤ ti.
    
    Parameters:
    ----------
    times_in : List[int]
        Arrival times of modes in the minimal light cone.
        Must satisfy: times_in[0] = 0 ≤ ... ≤ times_in[-1] = nt.
    ti : int
        Time step index (0 ≤ ti < times_in[-1]). 

    Returns:
    -------
    int
        Number of modes that have entered by time ti.

    See Also:
    --------
    minimal_forward_frame : Produces the `times_in` array.
    get_inout_range : Uses m_in to compute input/output mode ranges.
    """
    for i in range(len(times_in) - 1):
        if ti < times_in[0]:
            return 0
        if times_in[i] <= ti < times_in[i+1]:
            return i + 1
    raise ValueError("Index is out of maximal time")

import lightcones.frames.causal_diamonds as causal_diamonds

__all__ = ['minimal_forward',
           'm_in',
           'causal_diamonds']
