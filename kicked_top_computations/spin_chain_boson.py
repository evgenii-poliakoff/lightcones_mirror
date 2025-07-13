import numpy as np
import scipy
from scipy.sparse import spdiags
from scipy.linalg import eigh
from scipy import sparse

import math
from numpy import linalg as LA
import scipy.sparse.linalg as sl

from scipy.sparse import coo_matrix
from scipy.sparse import csc_matrix

import random
import secondquant as sq
import local_op


class LocalObservables:
    def __init__(self, f, n_spin, m_max, n_max, id_s = None):
        self.local_observables = self.local_projections_f(f, n_spin, m_max, n_max, id_s)

    def local_projections_f(self, f, n_spin, m_max, n_max, id_s):
    
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
            
        if not id_s is None:
            self.K = self.K * id_s.shape[0]
        
        return local_ops

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
    
# class which computes sparce matrices for spin-boson model 
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
        self.max_num_quanta = max_num_quanta

        self.dimension = hs_joint.dimension

        
    def get_local_observables(self):
        return LocalObservables(self.hs_joint, self.num_spin, self.num_modes, self.max_num_quanta)