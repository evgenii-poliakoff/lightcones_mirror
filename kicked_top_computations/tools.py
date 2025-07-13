import numpy as np
import scipy
from scipy.sparse import spdiags
from scipy.linalg import eigh
from scipy import sparse

import math
from numpy import linalg as LA
import scipy.sparse.linalg as sl

from scipy.sparse import coo_matrix

import random

import fastmul

from evolution import evolution

def mv(m, vin, vout, cin=1, cout=0):
    fastmul.fastmul(m.data, m.indices, m.indptr, cin, vin, cout, vout)

def tridiag(e, h):
    data = [np.concatenate((h, np.array([0]))), np.array(e), np.concatenate((np.array([0]), h))]
    diags = np.array([-1, 0, 1])

    n = len(e)
    return spdiags(data, diags, n, n).tocsc()

def dyad(ket, bra):
    return np.kron(ket, bra.T.conj())

def as_column_vector(ket):
    return ket[:, None]

def make_hermitean(m):
    m += m.T.conj()
    m /= 2

def find_largest_eigs(m, k = None):
    n = len(m)
    if k is None:
        k = n
    k = min(k, n)
    e, v = eigh(m, subset_by_index = [n - k, n - 1])
    e = np.flip(e)
    v = np.flip(v, axis = 1)
    return (e, v)

def find_smallest_eigs(m, k = None):
    n = len(m)
    if k is None:
        k = n
    k = min(k, n)
    e, v = eigh(m, subset_by_index = [0, k - 1])
    return (e, v)

def find_eigs_ascending(m):
    e, v = eigh(m)
    return (e, v)

def find_eigs_descending(m):
    e, v = eigh(m)
    e = np.flip(e)
    v = np.flip(v, axis = 1)
    return (e, v)
   
def evolutionpy_chained(dt, H, initial_state, start_time = None, start_index = None, end_time = None, end_index = None, tol = 10**(-6), first_in_chain = True):

    K = initial_state.size

    use_time = False
    use_index = False

    if not start_time is None and not end_time is None:

        use_time = True
        nt = math.floor((end_time - start_time) / dt)

    if not start_index is None and not end_index is None:

        use_index = True
        nt = end_index - start_index

    if not use_index != use_time:
        raise ValueError('evolution should be called either in time or in step-index mode')

    if (first_in_chain):

        if (use_index):
            yield (start_index, initial_state)
        else:
            yield (start_time, initial_state)

    psi = np.copy(initial_state)
    psi_mid = np.copy(psi)

    b = nt

    for i in range(0, b):

        psi_mid[:] = psi

        if (use_index):

            H_mid = H(start_index + i)

        else:

            H_mid = H(start_time + (i + 0.5) * dt)

        while(True):

            psi_mid_next = H_mid @ psi_mid

            psi_mid_next = psi - 1j * dt / 2 * psi_mid_next

            err = max(abs(psi_mid_next - psi_mid))

            swp = psi_mid_next
            psi_mid_next = psi_mid
            psi_mid = swp

            if err < tol:
                break

        psi = 2 * psi_mid - psi

        if (use_index):
            yield (start_index + i + 1, psi)
        else:
            yield (start_time + i * dt, psi)
    
def evolutionpy(dt, H, initial_state, start_time = None, start_index = None, end_time = None, end_index = None, tol = 10**(-6)):

    K = initial_state.size

    use_time = False
    use_index = False

    if not start_time is None and not end_time is None:

        use_time = True
        nt = math.floor((end_time - start_time) / dt)

    if not start_index is None and not end_index is None:

        use_index = True
        nt = end_index - start_index

    if not use_index != use_time:
        raise ValueError('evolution should be called either in time or in step-index mode')


    if (use_index):
        yield (start_index, initial_state)
    else:
        yield (start_time, initial_state)

    psi = np.copy(initial_state)
    psi_mid = np.copy(psi)

    b = nt - 1

    for i in range(0, b):

        psi_mid[:] = psi

        if (use_index):

            H_mid = H(start_index + i)

        else:

            H_mid = H(start_time + (i + 0.5) * dt)

        while(True):

            psi_mid_next = H_mid @ psi_mid

            psi_mid_next = psi - 1j * dt / 2 * psi_mid_next

            err = max(abs(psi_mid_next - psi_mid))

            swp = psi_mid_next
            psi_mid_next = psi_mid
            psi_mid = swp

            if err < tol:
                break

        psi = 2 * psi_mid - psi

        if (use_index):
            yield (start_index + i + 1, psi)
        else:
            yield (start_time + i * dt, psi)
            
def evolutionpy2(dt, apply_H, eval_O, initial_state, start_time = None, start_index = None, end_time = None, end_index = None, tol = 10**(-6)):

    K = initial_state.size

    use_time = False
    use_index = False

    if not start_time is None and not end_time is None:

        use_time = True
        nt = math.floor((end_time - start_time) / dt)

    if not start_index is None and not end_index is None:

        use_index = True
        nt = end_index - start_index

    if not use_index != use_time:
        raise ValueError('evolution should be called either in time or in step-index mode')


    if (use_index):
        eval_O(start_index, initial_state)
    else:
        eval_O(start_time, initial_state)

    psi = np.copy(initial_state)
    psi_mid = np.copy(psi)

    psi_mid_next = np.zeros(K, dtype = complex)

    for i in range(0, nt - 1):

        #psi_tmp = psi
        #psi = psi_mid
        #psi_mid = psi_tmp
        
        psi_mid[:] = psi

        while(True):

            psi_mid_next.fill(0)
            
            if (use_index):
                apply_H(start_index + i, psi_mid, psi_mid_next)
            else:
                apply_H(start_time + (i + 0.5) * dt, psi_mid, psi_mid_next)
            
            #psi_mid_next = H_mid @ psi_mid

            psi_mid_next = psi - 1j * dt / 2 * psi_mid_next

            err = max(abs(psi_mid_next - psi_mid))

            swp = psi_mid_next
            psi_mid_next = psi_mid
            psi_mid = swp

            if err < tol:
                break

        psi = 2 * psi_mid - psi

        if (use_index):
            eval_O(start_index + i + 1, psi)
        else:
            eval_O(start_time + i * dt, psi)

class LocalObservables:
    def __init__(self, f, n_spin, m_max, n_max, id_s = None):
    
        # self.local_observables = self.local_projections(f, m_max, n_max)
        self.local_observables = self.local_projections_f(f, n_spin, m_max, n_max, id_s)

    def local_projections_f(self, f, n_spin, m_max, n_max, id_s):
        
        import secondquant
        import local_op
        from scipy.sparse import csc_matrix
        
        self.K = f.dimension # total space dim
        self.local_dim = n_max + 1 # boson mode space dim
        
        a = f.annihilate
        a_dag = f.create
        
        local_ops = []
        
        o_data = np.zeros(self.K, dtype = complex)
        o_ind = np.zeros(self.K, dtype = np.int32)
        
        o_ptr = np.zeros(self.K + 1, dtype = np.int32)
        for i in range(self.K + 1):
            o_ptr[i] = i
        
        for i in range(n_spin, n_spin + m_max):
        # for i in range(0, m_max):

            o = np.array([f.occupations(j)[i] for j in range(self.K)])
            
            a_ = a[i]
            b_ = a_dag[i]
            
            mode_op = [[] for l in range(self.local_dim)]
            
            for p in range(self.local_dim):
                for q in range(self.local_dim):
                    
                    local_op.local_op(a_.data, a_.indices, a_.indptr, \
                        b_.data, b_.indices, b_.indptr, \
                            o_data, o_ind, o, p, q)
            
                    #print(o_data)
                    #print(o_ind)
                    #print(o_ptr)

                    op = csc_matrix((np.copy(o_data), np.copy(o_ind), np.copy(o_ptr)), shape = (self.K, self.K))

                    if not id_s is None:
                        op = kron(id_s, op)
                    mode_op[p].append(op)
                    
            local_ops.append(mode_op)

            #print *, 'K = ', self.K
            
        if not id_s is None:
            self.K = self.K * id_s.shape[0]

            #print *, 'K = ', self.K
        
        return local_ops
        
    # def local_projections(self, f, m_max, n_max):
    #     # construct projections to local hilbertspaces

    #     self.K = f.dimension

    #     a = f.annihilate
    #     a_dag = f.create

    #     local_op = []

    #     self.local_dim = n_max + 1

    #     # for each mode

    #     for i in range(0, m_max + 1):
   
    #         mode_op = [ [ ([], [], []) for l in range(self.local_dim) ] for k in range(self.local_dim) ]  # ([data], [row], [col])

    #         # diagonal 

    #         for j in range(0, self.K):

    #             o = f.occupations(j)
    #             p = o[i]
    #             q = p

    #             mode_op[q][p][0].append(1.0)
    #             mode_op[q][p][1].append(j)
    #             mode_op[q][p][2].append(j)

    #         # upper diagonal

    #         c_dag = a_dag[i].tocoo()
    #         c_dag.eliminate_zeros()
    #         row = c_dag.row
    #         col = c_dag.col

    #         for j in col:

    #             o = f.occupations(j)
    #             p = o[i]
        
    #             j_ = j
    #             d_ = 1.0

    #             for q in range(p + 1, self.local_dim):

    #                 if (not j_ in col):
    #                     break

    #                 j_ = row[col.tolist().index(j_)]

    #                 o_ = f.occupations(j_)

    #                 assert q == o_[i]


    #                 mode_op[q][p][0].append(d_)
    #                 mode_op[q][p][1].append(j_)
    #                 mode_op[q][p][2].append(j)

    #         # lower diagonal

    #         c = a[i].tocoo()
    #         c.eliminate_zeros()
    #         row = c.row
    #         col = c.col

    #         for j in col:

    #             o = f.occupations(j)
    #             p = o[i]
        
    #             j_ = j
    #             d_ = 1.0

    #             for q in reversed(range(0, p)):

    #                 if (not j_ in col):
    #                     break

    #                 j_ = row[col.tolist().index(j_)]

    #                 o_ = f.occupations(j_)
    #                 q = o_[i]

    #                 mode_op[q][p][0].append(d_)
    #                 mode_op[q][p][1].append(j_)
    #                 mode_op[q][p][2].append(j)

    #         for k in range(self.local_dim):
    #             for l in range(self.local_dim):
    #                 s = mode_op[k][l]
    #                 mode_op[k][l] = coo_matrix((s[0], (s[1], s[2])), shape = (self.K, self.K), dtype = complex).tocsc()

    #         local_op.append(mode_op)

    #     return local_op

    def partial_trace(self, psi, measured_mode):

        rho = np.zeros((self.local_dim, self.local_dim), dtype = complex)

        for i in range(self.local_dim):
            for j in range(self.local_dim):
                rho[i, j] = np.vdot(psi, self.local_observables[measured_mode][j][i] @ psi)

        return rho

    def project_to_vacuum(self, psi, measured_mode):

        psi_vac = self.local_observables[measured_mode][0][0] @ psi  
        return psi_vac

    def quantum_jump(self, psi, measured_mode):

        # find the "preferred" jump basis
        rho = self.partial_trace(psi, measured_mode)
        jump_probs, jump_states = LA.eigh(rho)

        # select the jump 

        xi = random.random()

        jump_index = 0

        for k in range(0, self.local_dim):
            xi = xi - jump_probs[k]

            if xi < 0:
                jump_index = k
                break

        # execute the jump

        psi_collapsed = np.zeros(self.K, dtype = complex)

        for k in range(self.local_dim):
            psi_collapsed = psi_collapsed + np.conj(jump_states[k, jump_index]) * self.local_observables[measured_mode][0][k] @ psi  

        # normalize
            
        psi_collapsed = psi_collapsed / np.sqrt(np.vdot(psi_collapsed, psi_collapsed))

        return psi_collapsed
    
    def quantum_jumpEx(self, psi, measured_mode):

        # find the "preferred" jump basis
        rho = self.partial_trace(psi, measured_mode)
        jump_probs, jump_states = LA.eigh(rho)

        # select the jump 

        xi = random.random()

        jump_index = 0

        for k in range(0, self.local_dim):
            xi = xi - jump_probs[k]

            if xi < 0:
                jump_index = k
                break

        # execute the jump

        psi_collapsed = np.zeros(self.K, dtype = complex)

        for k in range(self.local_dim):
            psi_collapsed = psi_collapsed + np.conj(jump_states[k, jump_index]) * self.local_observables[measured_mode][0][k] @ psi  

        # normalize
            
        psi_collapsed = psi_collapsed / np.sqrt(np.vdot(psi_collapsed, psi_collapsed))

        return (psi_collapsed, jump_probs, jump_states, jump_index, rho)
    
    def a(self):
        
        a_ = np.zeros((self.local_dim, self.local_dim), dtype = complex)
        
        for i in range(self.local_dim-1):
            a_[i, i + 1] = np.sqrt(i+1)
          
        return a_
        
    def a_dag(self):
        
        a_dag_ = np.zeros((self.local_dim, self.local_dim), dtype = complex)
        
        for i in range(self.local_dim-1):
            a_dag_[i + 1, i] = np.sqrt(i+1)
          
        return a_dag_

    def pair_quantum_jump(self, psi_r, psi_l, measured_mode):

        rho_l = self.partial_trace(psi_l, measured_mode)
        c_l, phi_l = LA.eigh(rho_l)

        c_r = np.zeros(self.local_dim, dtype = np.cdouble)

        psi_r_collapsed = np.zeros((self.K, self.local_dim), dtype = complex)

        for k in range(self.local_dim):

            for l in range(self.local_dim):
                psi_r_collapsed[:, k] = psi_r_collapsed[:, k] + np.conj(phi_l[l, k]) * self.local_observables[measured_mode][0][l] @ psi_r  

            norm = np.vdot(psi_r_collapsed[:, k], psi_r_collapsed[:, k])

            if norm > 10**(-9):
                c_r[k] = math.sqrt(norm)
                psi_r_collapsed[:, k] = psi_r_collapsed[:, k] / c_r[k]
            else:
                c_r[k] = 0

        jump_probs = c_l * c_r
        z = np.sum(jump_probs)
        jump_probs = jump_probs / z

        # select the jump 
        xi = random.random()
        jump_index = 0

        for k in range(0, self.local_dim):
            xi = xi - jump_probs[k]

            if xi < 0:
                jump_index = k
                break

        psi_l_collapsed = np.zeros(self.K, dtype = complex)

        for k in range(self.local_dim):
            psi_l_collapsed = psi_l_collapsed + np.conj(phi_l[k, jump_index]) * self.local_observables[measured_mode][0][k] @ psi_l  

        # normalize
            
        psi_l_collapsed = psi_l_collapsed / np.sqrt(np.vdot(psi_l_collapsed, psi_l_collapsed))

        return (psi_r_collapsed[:, jump_index], psi_l_collapsed, z)
    
class ray:
    def __init__(self, chain_generator, n_sites, dt, tmax, U = None):
        
        es = []
        hs = []
    
        ns = n_sites
        cg = chain_generator
    
        for i in range(ns):
            try:
                e_, h_ = next(chain_generator)
            except StopIteration:
                break
            
            es.append(e_)
            hs.append(h_)
        
        ns = i + 1
        
        self.tmax = tmax
        self.n_sites = ns
        self.coupling = hs[0]
        self.H = tridiag(es, hs[1:])
        
        tg = np.arange(0, tmax + dt, dt)
        self.t = tg
        self.dt = dt
        ntg = tg.size
        self.nt = ntg
        
        self.U = U
        if U is None:
            self.U = np.eye(n_sites, dtype = complex)    
    
        #-------------------------------------------------
    
        psi0 = np.zeros(ns, dtype = np.cdouble)
        psi0[0] = 1

        psi_lc = np.zeros((ns, ntg), dtype = np.cdouble)

        def Ht(t):
            return self.H

        for i, psi in evolutionpy(start_index = 0, end_index = ntg, H = Ht, dt = dt, initial_state = psi0):
            psi_lc[:, i] = np.copy(psi)
        
        self.psi = psi_lc
        
class rho:
    def __init__(self, matrix, ray, ti = 0 , a = None, b = None):
        
        self.ti = ti
        self.matrix = matrix
        self.ray = ray
        
        self.a = a
        self.b = b
        
        a, b = self.check_bounds()
        self.a = a
        self.b = b
        
        self.pi_max = None

    def clone(self):
        
        return rho(np.copy(self.matrix), ray, self.ti, self.a, self.b)
        
    def check_bounds(self):
        
        a = 0
        b = self.matrix.shape[0]
            
        if not self.a is None:
            a = self.a
                
        if not self.b is None:
            b = self.b
            
        return a, b
    
    def get_pi_max(self):
        
        if self.pi_max is None:
            
            a, b = self.check_bounds()
            m = self.matrix[a:b, a:b]
        
            self.pi_max, _ = find_largest_eigs(m, 1)
        
        return self.pi_max
        
        
    def is_inside(self, chi, r_cut):
        
        a, b = self.check_bounds()
            
        m = self.matrix[a:b, a:b]
        
        pi_max = self.get_pi_max()

        lr_metric = chi[a:b].T.conj() @ m @ chi[a:b] - r_cut * pi_max
        return lr_metric >= 0
        
    def subspace(self, a = None, b = None):
        
        if not a is None:
            self.a = a
        
        if not b is None:
            self.b = b
            
        a, b = self.check_bounds()
        
        self.a = a
        self.b = b
            
        self.pi_max = None
        
    def restrict_left_causal(self, ti_from, r_cut = None):
        
        from scipy.linalg import eig
        from scipy.linalg import expm
        
        a, b = self.check_bounds()
        m = self.matrix[a:b, a:b]
    
        outside = True
            
        H = None
            
        if not r_cut is None:
                
            pi_max = self.get_pi_max()
            pi_min, _ = find_smallest_eigs(m, 1)
            lr_metric = pi_min - r_cut * pi_max
            outside = lr_metric < 0
        
        if outside:
            
            pi, U = find_eigs_ascending(m)
            
            U = U.T.conj()
            
            duration = (self.ti - ti_from) * self.ray.dt

            e_, v_ = eig(U)
            e_ = np.log(e_ + 0j) / duration
            H = v_ @ np.diag(e_) @ v_.conj().T
            
            for i in range(ti_from, self.ti):
                
                du = expm(i * self.ray.dt * H)
                self.ray.psi[a : b, i] = du @ self.ray.psi[a : b, i]
            
            for i in range(self.ti, self.ray.nt):
                self.ray.psi[a : b, i] = U @ self.ray.psi[a : b, i]
                
            H = 1j * H
            
        return H
        
    def restrict_left(self, r_cut = None):
        
        a, b = self.check_bounds()
        m = self.matrix[a:b, a:b]
    
        outside = True
            
        if not r_cut is None:
            
            pi_max = self.get_pi_max()
            pi_min, _ = find_smallest_eigs(m, 1)
            lr_metric = pi_min - r_cut * pi_max
            outside = lr_metric < 0
        
        if outside:
            
            pi, U = find_eigs_ascending(m)
            
            self.ray.U[a:b, :] = U.T.conj() @ self.ray.U[a:b, :]
            self.ray.psi[a:b, :] = U.T.conj() @ self.ray.psi[a:b,:]
            self.matrix[a:b, a:b] = np.diag(pi.astype('cdouble'))
            make_hermitean(self.matrix)
            self.a = a + 1
            self.b = b
            #self.pi_max = None
            
        return outside

    def restrict_right(self, r_cut = None):
        
        a, b = self.check_bounds()
        m = self.matrix[a:b, a:b]
        make_hermitean(m)
        
        outside = True
            
        if not r_cut is None:
            
            pi_max = self.get_pi_max()
            pi_min, _ = find_smallest_eigs(m, 1)
            lr_metric = pi_min - r_cut * pi_max
            outside = lr_metric < 0
        
        if outside:
            
            pi, U = find_eigs_descending(m)
            
            self.ray.U[a:b, :] = U.T.conj() @ self.ray.U[a:b, :]
            self.ray.psi[a:b, :] = U.T.conj() @ self.ray.psi[a:b,:]
            self.matrix[a:b, a:b] = np.diag(pi.astype('cdouble'))
            make_hermitean(self.matrix)
            self.b = b - 1
            self.a = a
            #self.pi_max = None
            
        return outside
    
def rho_plus_forward(ray):
    
    matrix = np.zeros((ray.n_sites, ray.n_sites), dtype=complex)
    r = rho(matrix, ray)
    
    for i in range(ray.nt):
        
        phi = as_column_vector(ray.psi[:, i])
        r.matrix = r.matrix + dyad(phi, phi) * ray.dt
        make_hermitean(r.matrix)
        r.pi_max = None
        r.ti = i
        
        yield r
        
def rho_plus_backward(r):
    
    for i in reversed(range(r.ray.nt)):
        
        r.ti = i
        yield r
        
        phi = as_column_vector(r.ray.psi[:, i])
        r.matrix = r.matrix - dyad(phi, phi) * r.ray.dt
        make_hermitean(r.matrix)
        r.pi_max = None
        
def rho_minus_forward(r):
    
    for i in range(r.ray.nt):
            
        r.ti = i
        yield r
        
        phi = as_column_vector(r.ray.psi[:, i])
        r.matrix = r.matrix - dyad(phi, phi) * r.ray.dt
        make_hermitean(r.matrix)
        r.pi_max = None

class spin_boson_model:
    def __init__(self, num_modes, max_num_quanta):
        import secondquant as sq
    
        hs_atom = sq.fock_space(num_modes = 1, max_total_occupation = 1, statistics = 'Bose')
        m = num_modes
        n = max_num_quanta
        fs_chain = sq.fock_space(num_modes = m, max_total_occupation = n, statistics = 'Bose') 
        hs_joint = sq.fock_space_kron(hs_atom, fs_chain)
        b_hat = hs_joint.annihilate
        b_hat_dag = hs_joint.create
        sigma_m = b_hat[0]
        sigma_p = b_hat_dag[0]
        sigma = [hs_joint.sigmax(0), hs_joint.sigmay(0), hs_joint.sigmaz(0)]
        a_hat = b_hat[1:]
        a_hat_dag = b_hat_dag[1:]
    
        self.hs_joint = hs_joint
        self.space = hs_joint
        
        self.s_m = sigma_m
        self.s_p = sigma_p
        self.s = sigma
        
        self.s_x = sigma[0]
        self.s_y = sigma[1]
        self.s_z = sigma[2]
                         
        self.a = a_hat
        self.a_dag = a_hat_dag
        
        self.eye = hs_joint.eye
        
        self.num_modes = num_modes
        self.max_num_quanta = max_num_quanta
        
        self.dimension = hs_joint.dimension
        
    def get_local_observables(self):
        return LocalObservables(self.hs_joint, self.num_modes, self.max_num_quanta)
    
class fermion_2lead_fermion_model:
    def __init__(self, num_impurity_modes, num_reservoir_modes, max_num_quanta):
        
        import secondquant as sq
        
        self.n_qua = max_num_quanta
        self.m_imp = num_impurity_modes
        self.m_env = num_reservoir_modes
         
        
        total_num_modes = num_impurity_modes + 2 * num_reservoir_modes
        
        self.m_tot = total_num_modes
        
        joint = sq.fock_space(num_modes = total_num_modes, max_total_occupation = max_num_quanta, statistics = 'Fermi')
        
        self.d = joint.annihilate[: num_impurity_modes] # impurity modes
        self.d_dag = joint.create[: num_impurity_modes]
        
        self.l = joint.annihilate[num_impurity_modes : num_impurity_modes + num_reservoir_modes] # reservoir modes
        self.l_dag = joint.create[num_impurity_modes : num_impurity_modes + num_reservoir_modes]
        
        self.r = joint.annihilate[num_impurity_modes + num_reservoir_modes :] # reservoir modes
        self.r_dag = joint.create[num_impurity_modes + num_reservoir_modes :]
        
        self.space = joint
        self.dimension = joint.dimension
        
    def get_local_observables(self):
        return LocalObservables(self.space, self.m_tot - 1, 1)
    
class fermion_fermion_model:
    def __init__(self, num_impurity_modes, num_reservoir_modes, max_num_quanta):
        
        import secondquant as sq
        
        self.n_qua = max_num_quanta
        self.m_imp = num_impurity_modes
        self.m_env = num_reservoir_modes
         
        
        total_num_modes = num_impurity_modes + num_reservoir_modes
        
        self.m_tot = total_num_modes
        
        joint = sq.fock_space(num_modes = total_num_modes, max_total_occupation = max_num_quanta, statistics = 'Fermi')
        
        self.d = joint.annihilate[: num_impurity_modes] # impurity modes
        self.d_dag = joint.create[: num_impurity_modes]
        
        self.c = joint.annihilate[num_impurity_modes :] # reservoir modes
        self.c_dag = joint.create[num_impurity_modes :]
        
        self.space = joint
        self.dimension = joint.dimension
        
    def get_local_observables(self):
        return LocalObservables(self.space, self.m_tot - 1, 1)
  
class qubit:
    def __init__(self):
        import secondquant as sq
        
        space = sq.fock_space(num_modes = 1, max_total_occupation = 1, statistics = 'Bose')
        
        sigma_m = space.annihilate[0]
        sigma_p = space.create[0]
        sigma = [space.sigmax(0), space.sigmay(0), space.sigmaz(0)]
        
        self.space = space
        
        self.s_m = sigma_m
        self.s_p = sigma_p
        self.s = sigma
        
        self.s_x = sigma[0]
        self.s_y = sigma[1]
        self.s_z = sigma[2]
        
        self.eye = space.eye
        
        self.dimension = space.dimension

class quantum_top:
    def __init__(self, j):
    
        import secondquant as sq
        
        space = sq.fock_space(num_modes = 1, max_total_occupation = round(2*j), statistics = 'Bose')
        self.space = space
        
        self.j_m = space.j_m[0]
        self.j_p = space.j_p[0]
        self.j_x = space.j_x[0]
        self.j_y = space.j_y[0]
        self.j_z = space.j_z[0]
        
        self.eye = space.eye
        
        self.j = j
        self.dimension = space.dimension
        
    def kick_z(self, a, b):
        
        j_z = self.j_z
        eye = self.eye
       
        return sl.expm(-1j * a * (j_z - b * eye) @ (j_z - b * eye)).tocsc()
    
class spin_chain:
    def __init__(self, num_modes):
        import secondquant as sq
        m = num_modes
        space = sq.fock_space(num_modes = m, max_total_occupation = 2**m, statistics = 'Bose', max_local_occupations = [1]*m)
        
        self.space = space
        
        self.sm = space.annihilate
        self.sp = space.create
        
        self.sx = [space.sigmax(i) for i in range(m)]
        self.sy = [space.sigmay(i) for i in range(m)]
        self.sz = [space.sigmaz(i) for i in range(m)]
        self.states = space.states_list
        self.num_modes = num_modes
        #self.max_num_quanta = max_num_quanta
        
        self.dimension = space.dimension

        #self.id_s = id_s

class spin_chain_boson_model:
    def __init__(self, length, num_modes, max_num_quanta):
        import secondquant as sq
        # local_bounds = [1] * length
        local_bounds = [1] * int(length)
        m_spin = length
        
        space_spin = sq.fock_space(num_modes = m_spin, max_total_occupation = 2**m_spin, statistics = 'Bose', max_local_occupations=local_bounds)
        
        m_boson = num_modes
        n = max_num_quanta
        
        fs_chain = sq.fock_space(num_modes = m_boson, max_total_occupation = n, statistics = 'Bose') 
        
        hs_joint = sq.fock_space_kron(space_spin, fs_chain)
        
        b_hat = hs_joint.annihilate
        b_hat_dag = hs_joint.create
        
        a_hat = b_hat[m_spin:]
        a_hat_dag = b_hat_dag[m_spin:]
        
        self.space = space_spin
        self.hs_joint = hs_joint
        self.fs_chain = fs_chain


        self.sx = [hs_joint.sigmax(i) for i in range(m_spin)]
        self.sy = [hs_joint.sigmay(i) for i in range(m_spin)]
        self.sz = [hs_joint.sigmaz(i) for i in range(m_spin)]

        self.sm  = [0.5 * (sx - 1j * sy) for sx, sy in zip(self.sx, self.sy)]
        self.sp  = [0.5 * (sx + 1j * sy) for sx, sy in zip(self.sx, self.sy)]
        
        self.a = a_hat
        self.a_dag = a_hat_dag
        
        self.num_modes = num_modes
        self.num_spin = m_spin

        self.dimension = hs_joint.dimension
        
        # self.num_modes = num_modes
        self.max_num_quanta = max_num_quanta
        
    def get_local_observables(self):
        return LocalObservables(self.hs_joint, self.num_spin, self.num_modes, self.max_num_quanta)
# class spin_chain_boson_model:
#     def __init__(self, length, max_num_flips, num_modes, max_num_quanta):
#         import secondquant as sq
#         local_bounds = [1]*length
#         m_spin = length
#         space_spin = sq.fock_space(num_modes = m_spin, max_total_occupation = max_num_flips, statistics = 'Bose', max_local_occupations = local_bounds)
        
#         m_boson = num_modes
#         n = max_num_quanta
        
#         fs_chain = sq.fock_space(num_modes = m_boson, max_total_occupation = n, statistics = 'Bose') 
#         hs_joint = sq.fock_space_kron(space_spin, fs_chain)
#         b_hat = hs_joint.annihilate
#         b_hat_dag = hs_joint.create

#         a_hat = b_hat[m_spin:]
#         a_hat_dag = b_hat_dag[m_spin:]
        
#         self.space = space_spin
        
#         self.sm = space_spin.annihilate
#         self.sp = space_spin.create
        
#         self.sx = [space_spin.sigmax(i) for i in range(m_spin)]
#         self.sy = [space_spin.sigmay(i) for i in range(m_spin)]
#         self.sz = [space_spin.sigmaz(i) for i in range(m_spin)]
        
#         self.num_modes = length
#         #self.max_num_quanta = max_num_quanta
        
#         self.dimension = hs_joint.dimension

#         self.hs_joint = hs_joint
#         self.a = a_hat
#         self.a_dag = a_hat_dag

#         #self.id_s = id_s

class top_boson_model:
    def __init__(self, j, num_modes, max_num_quanta):
        
        import secondquant as sq
    
        top = sq.fock_space(num_modes = 1, max_total_occupation = round(2*j), statistics = 'Bose')
    
        # hs_atom = sq.fock_space(num_modes = 1, max_total_occupation = 1, statistics = 'Bose')
        m = num_modes
        n = max_num_quanta
        fs_chain = sq.fock_space(num_modes = m, max_total_occupation = n, statistics = 'Bose') 
        hs_joint = sq.fock_space_kron(top, fs_chain)
        b_hat = hs_joint.annihilate
        b_hat_dag = hs_joint.create
        #sigma_m = b_hat[0]
        #sigma_p = b_hat_dag[0]
        
        j_m = hs_joint.j_m[0]
        j_p = hs_joint.j_p[0]
        j_x = hs_joint.j_x[0]
        j_y = hs_joint.j_y[0]
        j_z = hs_joint.j_z[0]
        
        #sigma = [hs_joint.sigmax(0), hs_joint.sigmay(0), hs_joint.sigmaz(0)]
        a_hat = b_hat[1:]
        a_hat_dag = b_hat_dag[1:]
    
        self.hs_joint = hs_joint
        
        self.j_m = j_m
        self.j_p = j_p
        self.j_x = j_x
        self.j_y = j_y
        self.j_z = j_z
        self.j = j
                         
        self.a = a_hat
        self.a_dag = a_hat_dag
        
        self.eye = hs_joint.eye
        
        self.num_modes = num_modes
        self.max_num_quanta = max_num_quanta
        
        self.dimension = hs_joint.dimension
        
    def kick_z(self, a, b):
        
        j_z = self.hs_joint.f1.j_z[0]
        
        eye_1 = self.hs_joint.f1.eye
        eye_2 = self.hs_joint.f2.eye
        
        #sparse.kron(f1.create[k], f2.eye).tocsc()
        
        return sparse.kron(sl.expm(-1j * a * (j_z - b * eye_1) @ (j_z - b * eye_1)), eye_2).tocsc()
        
    def get_local_observables(self):
        return LocalObservables(self.hs_joint, self.num_modes, self.max_num_quanta)
    
def kron(a, b):
    return scipy.sparse.kron(a, b, format = 'csc')
        

        
class boson_chain:
    def __init__(self, num_modes, max_num_quanta, id_s = None):
        import secondquant as sq
        m = num_modes
        n = max_num_quanta
        
        space = sq.fock_space(num_modes = m, max_total_occupation = n, statistics = 'Bose')
        
        self.space = space
        
        self.a = space.annihilate
        self.a_dag = space.create    
        
        self.num_modes = num_modes
        self.max_num_quanta = max_num_quanta
        
        self.dimension = space.dimension

        self.id_s = id_s
        
    def get_local_observables(self):
        return LocalObservables(self.space, self.num_modes, self.max_num_quanta, self.id_s)
    
def forward_spatial_lightcone_generator(chain_generator, dt, chunk_size=100, guard_size=10, time_chunk = None, rcut = 10**(-6)):
        
        
    e = []
    h = []
    
    if chunk_size < 2:
        raise Exception('chunk_size must be >= 2')

    if time_chunk is None:
        time_chunk = 100 * dt
        
    for i in range(chunk_size):
        try:
            e_, h_ = next(chain_generator)
        except StopIteration:
            break
            
        e.append(e_)
        h.append(h_)
  
    ns = len(e)

    H1 = tridiag(e, h[:-1])
        
    phi_ini = np.zeros(ns, dtype = np.cdouble)
    phi_ini[0] = 1
        
    m = 1
    a = 0
        
    rho_plus = np.zeros((ns, ns), dtype = complex)
        
    n_time_chunk = round(time_chunk / dt)
        
    a_chunk = 0
    b_chunk = n_time_chunk
    phi_begin_chunk = phi_ini
    phi_begin = np.copy(phi_ini)
        
    first_chunk = True
    first_interval = True

    while(True):
                
        end_of_chain = False
        i_end = b_chunk
        phi_end = None
        
    
        def H1t(ti):
            return H1
        
        for i, phi in evolutionpy_chained(start_index = a_chunk, end_index = b_chunk, H = H1t, dt = dt, initial_state = phi_begin_chunk, first_in_chain = first_chunk):
                
            if not end_of_chain:
                
                phi_ = as_column_vector(phi)
                rho_plus += dyad(phi_, phi_) * dt
                    
                # find max eigenvalue
                pi_max, _ = find_largest_eigs(rho_plus, 1)
                    
                # check whether the next site is inside the lightcone
                site_sig = rho_plus[m + 1 - 1, m + 1 - 1]
                lr_metric = site_sig - rcut * pi_max
                if lr_metric > 0:
                        
                    b = i
                    #yield (a, b, m, evolutionpy_chained(start_index = a, end_index = b, H = H1t, dt = dt, initial_state = phi_begin, first_in_chain = first_interval))
                                        
                    yield (a, b, m, evolutionpy_chained(start_index = a, end_index = b - 1, H = H1t, dt = dt, initial_state = phi_begin, first_in_chain = True))
                         
                    first_interval = False

                    m += 1
                    a = i
                    phi_begin = np.copy(phi)
                        
                    # need to enlong the chain
                    if m > ns - guard_size:
                            
                        end_of_chain = True
                        i_end = i
                        phi_end = np.copy(phi)

        first_chunk = False
                            
        if end_of_chain:
                                
            for i in range(chunk_size):
                try:
                    e_, h_ = next(chain_generator)
                except StopIteration:
                    break
                    
                e.append(e_)
                h.append(h_)
                    
            H1 = tridiag(e, h[:-1])
            
            ns_ = len(e)
            
            phi_begin = np.concatenate((phi_begin, np.zeros(ns_ - ns, dtype = complex)))
            
            rho_plus_ = np.zeros((ns_, ns_), dtype = complex)
            rho_plus_[0:ns, 0:ns] = rho_plus
                
            
                
            b_chunk = i_end
            phi = phi_end
                
            phi = np.concatenate((phi, np.zeros(ns_ - ns, dtype = complex)))

            ns = ns_
            rho_plus = rho_plus_

        a_chunk = b_chunk
        b_chunk += n_time_chunk
            
        phi_begin_chunk = phi
        
def minimal_lightcone_generator(chain_generator, max_time, dt, chunk_size=100, guard_size=10, time_chunk = None, rcut = 10**(-6)):
    
    import itertools
    cg1, cg2 = itertools.tee(chain_generator)
    
    fg = forward_spatial_lightcone_generator(cg1, dt, chunk_size, guard_size, time_chunk, rcut = 10**(-9))
    
    for a, b, j, _ in fg:
        if b*dt >= max_time:
            break
    
    ns = j
    
    print('Resulting length of spatial chain: ', ns)
    
    #----------------------------------------------------
    
    t = max_time
    rel_tol = rcut
    
    tg = np.arange(0, t + dt, dt)
    ntg = tg.size
    
    #----------------------------------------------------
    
    es = []
    hs = []
    
    for i in range(ns):
        try:
            e_, h_ = next(cg2)
        except StopIteration:
            break
            
        es.append(e_)
        hs.append(h_)
        
    scoupling = hs[0]
    print('Coupling to impurity: ', scoupling)
    H = tridiag(es, hs[1:])
    
    #-------------------------------------------------
    
    psi0 = np.zeros(ns, dtype = np.cdouble)
    psi0[0] = 1

    psi_lc = np.zeros((ns, ntg), dtype = np.cdouble)

    def Ht(t):
        return H

    for i, psi in evolutionpy(start_index = 0, end_index = ntg, H = Ht, dt = dt, initial_state = psi0):
        psi_lc[:, i] = np.copy(psi)
    
    #------------------------------------------------
    
    rho_lc = np.zeros((ns, ns), dtype = np.cdouble)

    for i in range(0, ntg):
        psi = as_column_vector(psi_lc[:, i])
        rho_lc += dyad(psi, psi) * dt

    make_hermitean(rho_lc)
    
    #------------------------------------------------
    
    pi, U_rel = find_largest_eigs(rho_lc)

    lr_metric = pi - rel_tol * pi[0]
    inside_lightcone = lr_metric > 0

    pi_rel = pi[inside_lightcone]
    n_rel =  np.size(pi_rel)

    U_rel = U_rel[:, inside_lightcone]

    rho_lc_rel = np.diag(pi_rel.astype('cdouble'))

    #--------------------------------------------------------------------------------------------------

    psi_lc_rel = U_rel.T.conj() @ psi_lc

    #--------------------------------------------------------------------------------------------------

    rho_ret = np.copy(rho_lc_rel)

    #--------------------------------------------------------------------------------------------------

    times_in = []
    n_in = [n_rel]

    #U_min = np.eye(n_rel, dtype = np.cdouble)
    U_min = U_rel.T.conj() 

    n = n_rel

    for i in reversed(range(0, ntg)):
    
        pi_min, _ = find_smallest_eigs(rho_ret, 1)
        pi_max, _ = find_largest_eigs(rho_ret, 1)

        lr_metric = pi_min - rel_tol * pi_max
        outside_lightcone = lr_metric < 0

        if outside_lightcone:
            pi, U = find_eigs_descending(rho_ret)
            psi_lc_rel[: n, :] = U.T.conj() @ psi_lc_rel[: n, :]
            U_min[: n, :] = U.T.conj() @ U_min[: n, :]
            rho_ret = np.diag(pi[: -1].astype('cdouble'))
            times_in.insert(0, i + 1)
            n = n_rel - len(times_in)
            n_in.insert(0, n) 

        psi = as_column_vector(psi_lc_rel[: n, i])
        rho_ret -= dyad(psi, psi) * dt

        make_hermitean(rho_ret)

    #-------------------------------------------------------------------------------------------------

    #intervals_in = []

    i_left = 0

    for i_right, n in zip(times_in + [ntg], n_in):

        yield (i_left, i_right, n, U_min, ns)
        
        #intervals_in.append((i_left, i_right, n))
        i_left = i_right    
