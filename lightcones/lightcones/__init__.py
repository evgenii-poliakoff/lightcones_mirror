import numpy as np
import lightcones.linalg as la
from scipy.linalg import eig
from scipy.linalg import expm
from lightcones.solvers.schrodinger import solve

def spread(e, h, nt, dt):
    """
    Calculate the spread of operator

    Parameters:
    - e (numpy.ndarray of numpy.float64): array of on-site energies e[0] ... e[n_sites - 1]
    - h (numpy.ndarray of numpy.float64): array of hoppings h[0] ... h[n_sites - 2]
    - nt (int): number of time steps during which to compute spread
    - dt (float): size of one time step

    Returns:
    - numpy.ndarray of numpy.complex128: 2D array containing the spread of operator alpha[k, l] = \alpha(l*dt)_{k}
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

def rho_plus(spread, dt):
    """
    Compute rho_plus given the spread of operator
    
    Parameters:
    - spread (numpy.ndarray of numpy.complex128): 2D array of coefficients spread[k, l] = \alpha(l*dt)_{k}
    - dt (float): size of one time step
    
    Returns:
    - numpy.ndarray of numpy.complex128: 2D array containing matrix elements for the rho_plus for the duration of the spread
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

def minimal_forward_frame(spread, rho_plus, dt, rtol):
    """
    Compute the frame of a minimal light cone

    Parameters:
    - spread (numpy.ndarray of numpy.complex128): 2D array of coefficients spread[k, l] = \alpha(l*dt)_{k}
    - rho_plus (numpy.ndarray of numpy.complex128): 2D array containing matrix elements for the rho_plus for the duration of the spread
    - dt (float): size of one time step
    - rtol (float): relative cutoff treshold for the light cone boundary
    
    Returns:
    - numpy.ndarray of numpy.complex128: Unitary matrix U_min
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

def m_in(times_in, ti):
    for i in range(len(times_in) - 1):
        if ti < times_in[0]:
            return 0
        if times_in[i] <= ti < times_in[i+1]:
            return i + 1
    raise ValueError("Index is out of maximal time")

def get_inout_range(times_in, ti, m):
    _m_in = m_in(times_in, ti)
    _m_out = max(_m_in - m, 0)
    return _m_out, _m_in

def causal_diamond_frame(spread_min, times_in, U_min, rho_plus_min, dt, rtol, m):
    
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
    
def moving_frame(spread_cd, ti_arrival, U_cd, dt, m):
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

from lightcones.constructor import constructor

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
           'get_H',
           'constructor']