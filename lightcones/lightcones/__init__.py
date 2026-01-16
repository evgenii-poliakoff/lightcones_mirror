import numpy as np
import lightcones.linalg as la
from scipy.linalg import eig
from scipy.linalg import expm
from lightcones.solvers.schrodinger import solve
from typing import Optional, Tuple, List

def spread(e: np.ndarray[np.float64], h: np.ndarray[np.float64], nt: int, dt: float) -> np.ndarray[np.complex128]:
    """
    Computes the spread coefficients α_k(l·dt) = ⟨k|exp(-iH·l·dt)|0⟩ representing the 
    amplitude for a quantum state initially localized at site 0 to be found at site k 
    after time l·dt, where H is the Hamiltonian of a tight-binding chain.
    
    Parameters:
    ----------
    e : numpy.ndarray of numpy.float64
        On-site energies of the chain e[0] ... e[n_sites - 1], length n_sites.
    h : numpy.ndarray of numpy.float64  
        Hopping amplitudes between neighboring sites h[0] ... h[n_sites - 2], 
        length n_sites-1.     
    nt : int
        Number of time steps. The total evolution time is T = nt * dt.  
    dt : float
        Size of one time step (time discretization interval).
    
    Returns:
    -------
    numpy.ndarray of numpy.complex128
        2D array of shape (n_sites, nt) containing the spread coefficients:
        phi_lc[k, l] = α_k(l·dt).
    
    Notes:
    -----
    - Hamiltonian: H = Σ_i e[i] |i⟩⟨i| + Σ_i h[i] (|i⟩⟨i+1| + |i+1⟩⟨i|)
    - Initial state |ψ(0)⟩ = |0⟩.
    - Uses `lightcones.solvers.schrodinger` for time evolution.
    - Can be memory-intensive for large nt or n_sites.
    """
    n_sites = len(e)
    # first-quantized Hamiltonian
    H = la.tridiag(e, h)

    # initial condition: quantum on site 0 of the chain
    phi_0 = np.zeros(n_sites, dtype = complex)
    phi_0[0] = 1 # initially quantum is on the site 0

    # Here we store the propagated orbitals (the spread)
    phi_lc = np.zeros((n_sites, nt), dtype = np.cdouble) 

    def apply_h(ti, phi_in, phi_out):
        la.mv(H, phi_in, phi_out, cout=1)

    def eval_o(ti, phi):
        phi_lc[:, ti] = phi
    
    solve(0, nt-1, dt, apply_h, phi_0, eval_o = eval_o)
    
    return phi_lc

def rho_plus(spread: np.ndarray[np.complex128], dt: float) -> np.ndarray[np.complex128]:
    """
    Compute the retarded density matrix ρ₊ given the spread coefficients.
    
    The retarded density matrix ρ₊ is defined as the time-integrated density matrix:
        ρ₊ = ∫₀ᵀ |α(t)⟩⟨α(t)| dt ≈ Σₗ |α(l·dt)⟩⟨α(l·dt)| · dt
    where α_k(l·dt) are the spread coefficients of the time-evolved state |α(t)⟩. 
    
    Parameters:
    ----------
    spread : numpy.ndarray of numpy.complex128
        2D array of shape (n_sites, nt) containing spread coefficients α[k,l].
    dt : float
        Time step size used in the spread computation.
    
    Returns:
    -------
    numpy.ndarray of numpy.complex128
        2D array of shape (n_sites, n_sites) containing the retarded density matrix ρ₊.
        - The matrix is Hermitian positive semi-definite
        - Eigenvalues indicate the significance of each mode over the full time evolution.
        
    Notes:
    ------
    - This matrix represents the time-integrated contributions of each mode to the 
    quantum state and is used in light cone analysis.
    - Computation uses a simple Riemann sum to approximate the time integral.
    - The matrix is explicitly Hermitian to correct numerical errors.
    - Memory cost is O(n_sites x n_sites).

    See Also:
    --------
    spread : Computes the time-evolved spread coefficients α[k, l].
    """
    
    # get dimensions
    n_sites, nt = spread.shape
    
    # here rho_plus will be stored
    rho_lc = np.zeros((n_sites, n_sites), dtype = np.cdouble)
    
    for i in range(0, nt):
        phi = la.as_column_vector(spread[:, i])
        rho_lc += la.dyad(phi, phi) * dt

    la.make_hermitean(rho_lc)

    return rho_lc

def minimal_forward_frame(spread: np.ndarray[np.complex128], rho_plus: np.ndarray[np.complex128], dt: float, rtol: float) -> Tuple[List[int], np.ndarray[np.complex128], np.ndarray[np.complex128], np.ndarray[np.complex128]]:
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
    pi, U_rel = la.find_largest_eigs(rho_plus)
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
        pi_min, _ = la.find_smallest_eigs(rho_plus_min, 1)
        pi_max, _ = la.find_largest_eigs(rho_plus_min, 1)
        g_metric = pi_min - rtol * pi_max
        outside_lightcone = g_metric < 0

        if outside_lightcone:
            pi, U = la.find_eigs_descending(rho_plus_min)
            spread_min[: n, :] = U.T.conj() @ spread_min[: n, :]
            U_min[: n, :] = U.T.conj() @ U_min[: n, :]
            rho_plus_min = np.diag(pi[: -1].astype('cdouble'))
            times_in.insert(0, i + 1)
            n = n - 1

        psi = la.as_column_vector(spread_min[: n, i])
        rho_plus_min -= la.dyad(psi, psi) * dt
        la.make_hermitean(rho_plus_min)
        
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

def get_inout_range(times_in: List[int], ti: int, m: int) -> Tuple[int, int]:
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
    _m_in = m_in(times_in, ti)
    _m_out = max(_m_in - m, 0)
    return _m_out, _m_in

def causal_diamond_frame(spread_min: np.ndarray[np.complex128], times_in: List[int], U_min: np.ndarray[np.complex128], rho_plus_min: np.ndarray[np.complex128], dt: float, rtol: float, m: int) -> Tuple[np.ndarray[np.complex128], List[Optional[np.ndarray[np.complex128]]]]:
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
        psi = la.as_column_vector(spread_cd[:, ti])
        rho_minus -= la.dyad(psi, psi) * dt
    
    # produce U_cdia
    for i in range(m, len(times_in) - 1):
        
        n_out = i - m
        n_in = i
        
        la.make_hermitean(rho_minus)
        rho_cd = rho_minus[n_out : n_in, n_out : n_in]
        pi, U = la.find_eigs_ascending(rho_cd)

        # switch spread
        spread_cd[n_out : n_in, times_in[i] :] = U.T.conj() @ spread_cd[n_out : n_in, times_in[i] :]

        # switch rho_minus
        rho_minus[n_out : n_in, n_out : ] = U.T.conj() @ rho_minus[n_out : n_in, n_out : ] 
        rho_minus[n_out : , n_out : n_in] = rho_minus[n_out : , n_out : n_in] @ U 

        # store rotation
        U_cd.append(U.T.conj())
        
        # propagate rho_minus 
        for ti in range(times_in[i], times_in[i + 1]):
            psi = la.as_column_vector(spread_cd[n_out + 1 : , ti])
            rho_minus[n_out + 1 : , n_out + 1 : ] -= la.dyad(psi, psi) * dt
            
    # at the final time we do not produce output modes
    U_cd.append(None)
    
    return spread_cd, U_cd
    
def moving_frame(spread_cd: np.ndarray[np.complex128], ti_arrival: List[int], U_cd: List[Optional[np.ndarray[np.complex128]]], dt: float, m: int) -> Tuple[np.ndarray[np.complex128], List[Optional[np.ndarray[np.complex128]]]]:
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


import lightcones.linalg as linalg
import lightcones.solvers as solvers
import lightcones.models as models
import lightcones.jumps as jumps

import lightcones.space as space

__all__ = ['linalg',
           'space',
           'bounding_condition',
           'skip_condition', 
           'solvers', 
           'models', 
           'jumps',
           'spread',
           'rho_plus',
           'minimal_forward_frame',
           'm_in',
           'get_inout_range',
           'causal_diamond_frame',
           'moving_frame',
           'get_H']