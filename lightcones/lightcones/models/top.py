import math
import numpy as np
import scipy
from scipy.sparse import coo_matrix
import lightcones.space as sp

class top:
    def __init__(self, j):
        
        self.j = j
        
        #
        
        s = sp.states(1, bounding_condition=sp.bounding_condition.more_than_n_occupied(2 * j))
        q = sp.spins(j, s)
        
        #
        
        self.states = s
        self.dimension = s.dimension
        
        #
        
        self.j_x = q.j_x[0]
        self.j_y = q.j_y[0]
        self.j_z = q.j_z[0]
        
        self.j_m = q.j_m[0]
        self.j_p = q.j_p[0]        

    # zero operator acting in the state space of the model 
    def zero(self):
        return coo_matrix((self.states.dimension , self.states.dimension), dtype = complex).tocsc()
    
    # identity operator acting in the state space of the model
    def eye(self):
        return scipy.sparse.eye(self.states.dimension).tocsc()
    
    # vacuum state
    def vac(self):
        state = np.zeros(self.states.dimension, dtype = complex)
        state[0] = 1.0
        return state
    
    # state with given j_z
    def state_with(self, j_z):
        p = j_z + self.j
        p = round(p)
        state = self.vac()
        for _ in range(p):
            state = self.j_p @ state 
        state = state / math.sqrt(np.vdot(state, state).real)
        return state