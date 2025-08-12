import numpy as np
from numpy import linalg as LA
import random
from lightcones.jumps import density_matrix

# given wavefunction 'psi' from some multimode Hilbert space 'space'
# make random quantum jump with the probabilities given by
# Schmidt decomposition for bipartition into
# mode 'mode_index' and 'the rest of the system'
def make_jump(psi, space, mode_index, reset_to_vac = True):
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
    
    psi_collapsed = np.zeros(space.dimension, dtype = complex)
    # apply the jump
    if reset_to_vac:
        for k in range(local_dim):
            psi_collapsed = psi_collapsed + np.conj(jump_states[k, jump_index]) * space.outer(0, k, mode_index) @ psi
    else:
        for k in range(local_dim):
            for l in range(local_dim):
                psi_collapsed = psi_collapsed + jump_states[l, jump_index] * np.conj(jump_states[k, jump_index]) * space.outer(l, k, mode_index) @ psi
        
    # normalize    
    psi_collapsed = psi_collapsed / np.sqrt(np.vdot(psi_collapsed, psi_collapsed))
    
    return psi_collapsed