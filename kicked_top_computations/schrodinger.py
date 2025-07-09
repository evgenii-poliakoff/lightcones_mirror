import numpy as np
import solve as _solve

# solve the Schrodinger equation from time a to time b and a time step dt
def solve(a, b, dt, apply_h, psi_0, begin_step = None, eval_o = None, psi = None, psi_mid = None, psi_mid_next = None, eval_a = 1):
    if begin_step is None:
        def begin_step(ti, psi):
            pass
        
    if eval_o is None:
        def eval_o(ti, psi):
            pass
        
    if psi is None:
        psi = np.zeros(psi_0.size, dtype = complex)
        
    if psi_mid is None:
        psi_mid = np.zeros(psi_0.size, dtype = complex)
        
    if psi_mid_next is None:
        psi_mid_next = np.zeros(psi_0.size, dtype = complex)
        
    _solve.solve(a, b, dt, begin_step, apply_h, eval_o, psi_0, psi, psi_mid, psi_mid_next, eval_a)
    
__all__ = ['solve']