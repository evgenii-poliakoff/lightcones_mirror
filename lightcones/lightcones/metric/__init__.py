import numpy as np
import lightcones.linalg as ll
from lightcones.solvers.schrodinger import solve

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
    H = ll.tridiag(e, h)

    # initial condition: quantum on site 0 of the chain
    phi_0 = np.zeros(n_sites, dtype = complex)
    phi_0[0] = 1 # initially quantum is on the site 0

    # Here we store the propagated orbitals (the spread)
    phi_lc = np.zeros((n_sites, nt), dtype = np.cdouble) 

    def apply_h(ti, phi_in, phi_out):
        ll.mv(H, phi_in, phi_out, cout=1)

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
        phi = ll.as_column_vector(spread[:, i])
        rho_lc += ll.dyad(phi, phi) * dt

    ll.make_hermitean(rho_lc)

    return rho_lc