import numpy as np
from numpy import linalg as LA
import random
import fock

# given wavefunction 'psi' from some multimode Hilbert space 'space'
# make random quantum jump with the probabilities given by
# Schmidt decomposition for bipartition into
# mode 'mode_index' and 'the rest of the system'

def density_matrix(psi, space, mode_index):
    local_dim = space.max_total_occupation
    if not space.max_local_occupation is None:
        local_dim = min(local_dim, space.max_local_occupation[mode_index])
    rho = np.zeros((local_dim, local_dim), dtype = complex)
    for i in range(local_dim):
        for j in range(local_dim):
            rho[i, j] = np.vdot(psi, space.outer(j, i, mode_index) @ psi)
    return rho
    
def make_jump(psi, space, mode_index):
    # find the "preferred" jump basis
    rho = density_matrix(psi, space, mode_index)
    jump_probs, jump_states = LA.eigh(rho)
    
    # select the jump 
    xi = random.random()
    jump_index = 0
    local_dim, _ = rho.shape
    for k in range(0, local_dim):
        xi = xi - jump_probs[k]
        if xi < 0:
            jump_index = k
            break
        
    # apply the jump
    psi_collapsed = np.zeros(space.dimension, dtype = complex)
    for k in range(local_dim):
        psi_collapsed = psi_collapsed + np.conj(jump_states[k, jump_index]) * space.outer(0, k, mode_index) @ psi
        
    # normalize    
    psi_collapsed = psi_collapsed / np.sqrt(np.vdot(psi_collapsed, psi_collapsed))
    
    return psi_collapsed