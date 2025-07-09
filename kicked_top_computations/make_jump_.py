import numpy as np
import linalg as ll
import random

# given density matrix rho
# find its eigendecomposition as
# (pi_k, psi_k), k = 0 ... dimension - 1
# pi_k is probability
# psi_k is eigenstate
# then make random jump into the state psi_k with the probability pi_k
# returns (sampled k_jump, [ordered list of psi_k], [ordered list of pi_k])
# so that the jumped state is psi_k[k_jump]
# and the probability of the occured jump is pi_k[k_jump]
def make_jump_(rho):
    # find the "preferred" jump basis
    jump_probs, jump_states = ll.find_eigs_descending(rho)
    
    # select the jump 
    xi = random.random()
    jump_index = 0
    local_dim, _ = rho.shape
    for k in range(0, local_dim):
        xi = xi - jump_probs[k]
        if xi < 0:
            jump_index = k
            break
        
    return jump_index, jump_states, jump_probs