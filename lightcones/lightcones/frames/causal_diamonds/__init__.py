import numpy as np
from scipy.linalg import eig
from scipy.linalg import expm
from typing import Optional, Tuple, List
import lightcones.linalg as ll
import lightcones.frames

def get_inout_range_fixed(times_in: List[int], ti: int, m: int) -> Tuple[int, int]:
    """  
    Given a time step `ti` and causal diamond dimension `m`, this function computes
    which modes are currently inside the causal diamond as input modes and which
    have already exited as output modes. 
    
    Parameters:
    ----------
    times_in : List[int]
        List of arrival time indices from minimal_forward_frame().
    ti : int
        Time step index (0 ≤ ti < times_in[-1]). The current time for which
        to compute the input/output mode ranges.
    m : int
        Causal diamond dimension. The maximum number of modes that can be
        simultaneously inside a causal diamond. Must be positive.
    
    Returns:
    -------
    m_out : int
        Index of the first mode currently inside the causal diamond.
        Modes with indices < m_out have already exited the diamond.
    m_in : int
        Index one past the last mode currently inside the causal diamond.
        Modes with indices in [m_out, m_in) are inside the diamond.
        Number of modes inside = m_in - m_out ≤ m.

    See Also:
    --------
    m_in : Computes the number of modes that have entered by time ti
    minimal_forward_frame : Produces the times_in array
    causal_diamond_frame : Uses input/output ranges for diamond construction
    """
    _m_in = lightcones.frames.m_in(times_in, ti)
    _m_out = max(_m_in - m, 0)
    return _m_out, _m_in


def fixed(spread_min: np.ndarray[np.complex128], times_in: List[int], U_min: np.ndarray[np.complex128], rho_plus_min: np.ndarray[np.complex128], dt: float, rtol: float, m: int) -> Tuple[np.ndarray[np.complex128], List[Optional[np.ndarray[np.complex128]]]]:
    """
    Transform spread coefficients to the causal diamond frame for open quantum systems.

    This function constructs overlapping causal diamonds of fixed dimension `m` that represent
    subsystems evolving causally in time. Each diamond has an input-output structure
    and a time-dependent optimal basis.
    Algorithm:
    1. Initialize the remaining density matrix ρ₋(t) = ρ₊ - Σ |α(τ)⟩⟨α(τ)| dt for τ ≤ t.
    2. For each diamond [t_i, t_{i+1}):
        • extract the m x m submatrix of ρ₋,
        • diagonalize it to obtain the optimal basis,
        • rotate future spread coefficients,
        • update ρ₋ as modes exit the diamond.
    The resulting frame provides a time-dependent basis where quantum information
    flows causally from input to output modes through each diamond.

    Parameters:
    ----------
    spread_min : np.ndarray[np.complex128]
        2D array of shape (n_sites, nt) containing spread coefficients in the minimal frame.  
    times_in : List[int]
        List of arrival time indices from minimal_forward_frame().
    U_min : np.ndarray[np.complex128]
        Unitary matrix of shape (n_sites, n_sites) for the minimal frame transformation.
    rho_plus_min : np.ndarray[np.complex128]
        Retarded density matrix ρ₊ in the minimal frame.
    dt : float
        Time step size used in computing spread.   
    rtol : float
        Relative tolerance threshold (unused in current implementation but kept for
        interface consistency with related functions).    
    m : int
        Causal diamond dimension. The number of modes in each overlapping diamond.
        Must satisfy: 1 ≤ m ≤ number of modes in minimal light cone.
        Typical values: 2-6 for numerical stability and physical interpretation.
    
    Returns:
    -------
    spread_cd : np.ndarray[np.complex128]
        2D array of shape (n_sites, nt) containing spread coefficients in the
        causal diamond frame.    
    U_cd : List[Optional[np.ndarray[np.complex128]]]
        List of unitary transformations for each causal diamond.
        - U_cd[i] = None for i < m (first m diamonds are skipped)
        - U_cd[i] = unitary matrix of shape (m, m) for diamond starting at times_in[i]
        - U_cd[-1] = None (no transformation at final time)
        Each U_cd[i] transforms from the basis of diamond i-1 to diamond i.

    Notes
    -----
    - Each diamond acts as a sliding m-mode window ensuring causal propagation.
    - Complexity is O(nt x m³) due to repeated eigen-decompositions.

    See Also:
    --------
    minimal_forward_frame : Computes the minimal frame input for this function
    m_in : Determines number of modes at given time
    get_inout_range : Computes input/output mode indices for diamonds
    moving_frame : Further transforms causal diamond frame to moving frame

    """
    
    # spread in the causal diamond frame
    spread_cd = np.copy(spread_min)
    
    # skip first m arrival times
    U_cd = [None]*m
    
    # rho_minus initial value:
    # skip first m arrival times
    rho_minus = np.copy(rho_plus_min)
    for ti in range(0, times_in[m]):
        psi = ll.as_column_vector(spread_cd[:, ti])
        rho_minus -= ll.dyad(psi, psi) * dt
    
    # produce U_cdia
    for i in range(m, len(times_in) - 1):
        
        n_out = i - m
        n_in = i
        
        ll.make_hermitean(rho_minus)
        rho_cd = rho_minus[n_out : n_in, n_out : n_in]
        pi, U = ll.find_eigs_ascending(rho_cd)

        # switch spread
        spread_cd[n_out : n_in, times_in[i] :] = U.T.conj() @ spread_cd[n_out : n_in, times_in[i] :]

        # switch rho_minus
        rho_minus[n_out : n_in, n_out : ] = U.T.conj() @ rho_minus[n_out : n_in, n_out : ] 
        rho_minus[n_out : , n_out : n_in] = rho_minus[n_out : , n_out : n_in] @ U 

        # store rotation
        U_cd.append(U.T.conj())
        
        # propagate rho_minus 
        for ti in range(times_in[i], times_in[i + 1]):
            psi = ll.as_column_vector(spread_cd[n_out + 1 : , ti])
            rho_minus[n_out + 1 : , n_out + 1 : ] -= ll.dyad(psi, psi) * dt
            
    # at the final time we do not produce output modes
    U_cd.append(None)
    
    return spread_cd, U_cd


def moving(spread_cd: np.ndarray[np.complex128], ti_arrival: List[int], U_cd: List[Optional[np.ndarray[np.complex128]]], dt: float, m: int) -> Tuple[np.ndarray[np.complex128], List[Optional[np.ndarray[np.complex128]]]]:
    """
    Transform from causal diamond frame to a continuously evolving moving frame.
    
    Each discrete unitary transformation U_cd[i] is represented by a generator H_mv[i] such that
        U_cd[i] = exp(H_mv[i] · Δt_i),
    where Δt_i is the duration of the i-th causal diamond. The evolution within each
    diamond is then applied incrementally as exp(H_mv[i] · dt) at each time step.

    Parameters:
    ----------
    spread_cd : np.ndarray[np.complex128]
        2D array of shape (n_sites, nt) containing spread coefficients in the
        causal diamond frame.
    ti_arrival : List[int]
        Mode arrival times (times_in from minimal_forward_frame).
    U_cd : List[Optional[np.ndarray[np.complex128]]]
        Unitary transformations for each diamond; (U_cd from causal_diamond_frame).
    dt : float
        Time step size.
    m : int
        Causal diamond dimension. Must match the m used in causal_diamond_frame().

    Returns:
    -------
    spread_mv : np.ndarray[np.complex128]
        2D array of shape (n_sites, nt) containing spread coefficients in the
        moving frame.         
    H_mv : List[Optional[np.ndarray[np.complex128]]]
        List of generator Hamiltonians for the moving frame transformations.
        - H_mv[i] = None for i < m (first m diamonds) and i = len(U_cd)-1 (final)
        - H_mv[i] = m x m Hermitian matrix for i ∈ [m, len(U_cd)-1)
        Each H_mv[i] satisfies: U_cd[i] = exp(-i H_mv[i] Δt_i) where Δt_i is the
        duration of diamond i.

    Notes:
    -----
    - The generators H_mv[i] are generally not Hermitian; they are matrix logarithms
    of unitary transformations U_cd[i] and do not correspond directly to physical Hamiltonians.
    - The matrix logarithm is multi-valued; the principal branch is used via np.log.
    - The moving frame provides a smooth, continuously evolving basis suitable for 
    integration with time-dependent Schrödinger equations. 

    See Also:
    --------
    causal_diamond_frame : Computes the U_cd input for this function
    minimal_forward_frame : Computes the ti_arrival/times_in input
    get_H : Retrieves the generator Hamiltonian at a specific time
    """
    # find generator 
    H_mv = []
    for i in range(len(U_cd)):
        U = U_cd[i]
        if U is None:
            H_mv.append(None)
            continue
        a = ti_arrival[i - 1]
        b = ti_arrival[i]
        duration = (b - a) * dt
        e_, v_ = eig(U)
        e_ = np.log(e_ + 0j) / duration
        H = v_ @ np.diag(e_) @ v_.conj().T
        H_mv.append(H)
        
    # recompute couplings
    spread_mv = np.copy(spread_cd)
    for i in range(len(H_mv)):
        H = H_mv[i]
        if H is None:
            continue
        a = ti_arrival[i - 1]
        b = ti_arrival[i]
        n_out = i - m
        n_in = i
        U = np.eye(m, dtype = complex)
        dU = expm(dt * H)
        for j in range(a, b):
            spread_mv[n_out : n_in, j] = U @ spread_mv[n_out : n_in, j]
            U = dU @ U
            
    return spread_mv, H_mv

def get_H(times_in, H_mv, ti):
    """
    Retrieve the generator Hamiltonian at a given time step in the moving frame.

    Parameters:
    ----------
    times_in : List[int]
        Mode arrival times from minimal_forward_frame().
    H_mv : List[Optional[np.ndarray[np.complex128]]]
        Generator Hamiltonians from moving_frame().
        Entries may be None for initial diamonds or the final interval.
    ti : int
        Time step at which the generator Hamiltonian is requested.

    Returns
    -------
    np.ndarray[np.complex128] or None
        Generator Hamiltonian H_mv active at time step `ti`,
        or None if no generator is defined at time step `ti`.

    Notes
    -----
    - The Hamiltonian is piecewise constant over time intervals
      [times_in[i], times_in[i+1]).
    - This function maps a time step `ti` to the corresponding generator
      used for evolution in the moving frame.
    """
    for i in range(len(times_in) - 1):
        if ti < times_in[0]:
            return None
        if times_in[i] <= ti < times_in[i+1]:
            return H_mv[i + 1]
    raise ValueError("Index is out of maximal time")

__all__ = ['get_inout_range_fixed',
           'fixed',
           'moving',
           'get_H']


