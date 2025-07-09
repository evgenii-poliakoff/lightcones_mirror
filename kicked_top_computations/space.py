import numpy as np
import scipy
from scipy.sparse import coo_matrix


class bounding_condition:
    @staticmethod
    def none():
        return lambda state: False
    
    @staticmethod
    def more_than_singly_occupied():
        return lambda state: max(state) > 1
    
    @staticmethod
    def more_than_n_occupied(n):
        return lambda state: max(state) > n
    
    @staticmethod
    def more_than_in_total(n):
        return lambda state: sum(state) > n

class skip_condition:
    @staticmethod
    def none():
        return lambda state: False
    
    @staticmethod
    def odd():
        return lambda state: sum(state) % 2 != 0

class states:
    def __init__(self, num_modes, bounding_condition, skip_condition = skip_condition.none()):
        self.num_modes = num_modes
        self.skip_condition = skip_condition
        self.bounding_condition = bounding_condition
        self.all_states = list(self.state_generator())
        self.enumerated_states = {state: idx for idx, state in enumerate(self.all_states)}
        self.dimension = len(self.all_states)
        
    def state_generator(self):
        current_state = (0,) * self.num_modes
        while True:
            if (not self.skip_condition(current_state)):
                yield current_state
            j = self.num_modes - 1
            current_state = current_state[:j] + (current_state[j]+1,)
            while self.bounding_condition(current_state):
                j -= 1
                if j < 0:
                    return
                current_state = current_state[:j] + (current_state[j]+1, 0) + current_state[j+2:]
            
    def state_at(self, index):
        return self.all_states[index]
    
    def index_of(self, state):
        return self.enumerated_states[state]
    
    def vector_with(self, state):
        v = np.zeros(self.dimension, dtype = complex)
        v[self.index_of(state)] = 1.0
        return v
            
class bosons:
    def __init__(self, states):
        self.states = states
        
        #
        self.zero = coo_matrix((self.states.dimension , self.states.dimension), dtype = complex).tocsc()      
        
        #
        self.eye = scipy.sparse.eye(self.states.dimension).tocsc()
        
        #
        self.a = []
        for k in range(self.states.num_modes):
            row = np.zeros(self.states.dimension)
            col = np.arange(self.states.dimension, dtype=int)
            data = np.zeros(self.states.dimension)
            for i in range(self.states.dimension):
                in_state = self.states.all_states[i]
                if not in_state[k] == 0:
                    out_state = in_state[:k] + (in_state[k]-1,) + in_state[k+1:]
                    ind = self.states.index_of(out_state)
                    row[i] = ind
                    data[i]=(in_state[k])**0.5
            a_k = coo_matrix((data, (row, col)), shape=(self.states.dimension , self.states.dimension), dtype = complex)
            a_k.eliminate_zeros()
            self.a.append(a_k.tocsc())
                
        #
        self.a_dag=[]
        for k in range(self.states.num_modes):
            row = np.zeros(self.states.dimension)
            col = np.arange(self.states.dimension, dtype=int)
            data = np.zeros(self.states.dimension)
            for i in range(self.states.dimension):
                in_state = self.states.all_states[i]
                out_state = in_state[:k] + (in_state[k]+1,) + in_state[k+1:]
                if not self.states.bounding_condition(out_state):
                    ind = self.states.index_of(out_state)
                    row[i] = ind
                    data[i]=(out_state[k])**0.5
            a_k_dag = coo_matrix((data, (row, col)), shape=(self.states.dimension , self.states.dimension), dtype = complex)
            a_k_dag.eliminate_zeros()
            self.a_dag.append(a_k_dag.tocsc())
            
    def vac(self):
        state = np.zeros(self.states.dimension, dtype = complex)
        state[0] = 1.0
        return state
            
class fermions:
    def __init__(self, states):
        self.states = states
        
        more_than_singly = bounding_condition.more_than_singly_occupied()
        
        #
        self.zero = coo_matrix((self.states.dimension , self.states.dimension), dtype = complex).tocsc()      
        
        #
        self.eye = scipy.sparse.eye(self.states.dimension).tocsc()
        
        # parity operator
        data = np.zeros(self.states.dimension)
        for i in range(self.states.dimension):
            p = sum(self.states.state_at(i))
            data[i] = (-1)**p
        self.parity = scipy.sparse.diags(data).tocsc()
        
        #
        self.a = []
        for k in range(self.states.num_modes):
            row = np.zeros(self.states.dimension)
            col = np.arange(self.states.dimension, dtype=int)
            data = np.zeros(self.states.dimension)
            for i in range(self.states.dimension):
                in_state = self.states.all_states[i]
                if not in_state[k] == 0:
                    out_state = in_state[:k] + (in_state[k]-1,) + in_state[k+1:]
                    ind = self.states.index_of(out_state)
                    row[i] = ind
                    p = sum(in_state[:k])
                    data[i]=(-1)**p * (in_state[k])**0.5
            a_k = coo_matrix((data, (row, col)), shape=(self.states.dimension , self.states.dimension), dtype = complex)
            a_k.eliminate_zeros()
            self.a.append(a_k.tocsc())
                
        #
        self.a_dag=[]
        for k in range(self.states.num_modes):
            row = np.zeros(self.states.dimension)
            col = np.arange(self.states.dimension, dtype=int)
            data = np.zeros(self.states.dimension)
            for i in range(self.states.dimension):
                in_state = self.states.all_states[i]
                if more_than_singly(in_state):
                    raise ValueError("Using more than singly occupied states to construct fermionic operators")
                out_state = in_state[:k] + (in_state[k]+1,) + in_state[k+1:]
                if not self.states.bounding_condition(out_state):
                    ind = self.states.index_of(out_state)
                    row[i] = ind
                    p = sum(in_state[:k])
                    data[i]=(-1)**p * (out_state[k])**0.5
            a_k_dag = coo_matrix((data, (row, col)), shape=(self.states.dimension , self.states.dimension), dtype = complex)
            a_k_dag.eliminate_zeros()
            self.a_dag.append(a_k_dag.tocsc())
            
        #
        self.n = []
        for k in range(self.states.num_modes):
            self.n.append(self.a_dag[k] @ self.a[k])
            
    def vac(self):
        state = np.zeros(self.states.dimension, dtype = complex)
        state[0] = 1.0
        return state
        
class spins:
    def __init__(self, j, states):
        self.states = states
        self.j = j

        more_than_2j = bounding_condition.more_than_n_occupied(2 * j)
        
        #
        self.zero = coo_matrix((self.states.dimension , self.states.dimension), dtype = complex).tocsc()      
        
        #
        self.eye = scipy.sparse.eye(self.states.dimension).tocsc()
        
        #
        self.j_m = []
        for k in range(self.states.num_modes):
            row = np.zeros(self.states.dimension)
            col = np.arange(self.states.dimension, dtype=int)
            data = np.zeros(self.states.dimension)
            for i in range(self.states.dimension):
                in_state = self.states.all_states[i]
                if not in_state[k] == 0:
                    out_state = in_state[:k] + (in_state[k]-1,) + in_state[k+1:]
                    ind = self.states.index_of(out_state)
                    row[i] = ind
                    m = in_state[k] - self.j
                    data[i]= (self.j * (self.j + 1) - m * (m - 1))**0.5
            j_m_k = coo_matrix((data, (row, col)), shape=(self.states.dimension , self.states.dimension), dtype = complex)
            j_m_k.eliminate_zeros()
            self.j_m.append(j_m_k.tocsc())
        
        #
        self.j_p = []
        for k in range(self.states.num_modes):
            row = np.zeros(self.states.dimension)
            col = np.arange(self.states.dimension, dtype=int)
            data = np.zeros(self.states.dimension)
            for i in range(self.states.dimension):
                in_state = self.states.all_states[i]
                if more_than_2j(in_state):
                    raise ValueError("Using more than 2j occupied states to construct spin j operators")
                out_state = in_state[:k] + (in_state[k]+1,) + in_state[k+1:]
                if not self.states.bounding_condition(out_state):
                    ind = self.states.index_of(out_state)
                    row[i] = ind
                    m = in_state[k] - self.j
                    data[i]=(self.j * (self.j + 1) - m * (m + 1))**0.5
            j_p_k = coo_matrix((data, (row, col)), shape=(self.states.dimension , self.states.dimension), dtype = complex)
            j_p_k.eliminate_zeros()
            self.j_p.append(j_p_k.tocsc())
            
        #
        self.j_x = []
        self.j_y = []
        for k in range(self.states.num_modes):
            self.j_x.append((self.j_p[k] + self.j_m[k]) / 2)
            self.j_y.append((self.j_p[k] - self.j_m[k]) / 2 / 1j)
            
        #
        self.j_z = []
        for k in range(self.states.num_modes):
            row = np.zeros(self.states.dimension)
            col = np.arange(self.states.dimension, dtype=int)
            data = np.zeros(self.states.dimension)
            for i in range(self.states.dimension):
                in_state = self.states.all_states[i]
                ind = self.states.index_of(in_state)
                row[i] = ind
                m = in_state[k] - self.j
                data[i] = m
            j_z_k = coo_matrix((data, (row, col)), shape=(self.states.dimension , self.states.dimension), dtype = complex)
            j_z_k.eliminate_zeros()
            self.j_z.append(j_z_k.tocsc())
    def vac(self):
        state = np.zeros(self.states.dimension, dtype = complex)
        state[0] = 1.0
        return state

# bipartite system as a kron of left 'L' and right 'R' parts

# for fermions, given set of left states |Lk>, k = 1 ... L_dimension
# and a set of right states |Rl>, l = 1 ... R_dimension
# Also given left canonical operators a_L , a_L_dag and right canonical operators a_R, a_R_dag
# We construct the "product" state space as
# |kl> = |Lk>|Rl> for k, l running in the respective ranges/
# The canonical operators are embedded into the "product" state space as
# a_L -> a_L_ = a_L * identity_R, a_L_dag -> a_L_dag_ = a_L_dag * identity_R
# a_R -> a_R_ = parity_L * a_R, a_R_dag -> a_R_dag_ = parity_L * a_R_dag
# where parity_L * |Lk> gives the parity of number of particles in |Lk>
# the anticommutation relations are ensured:
# (parity_L * a_R) * (a_L * identity_R) = parity_L * a_L * a_R * identity_R
#  = - a_L * parity_L * a_R * identity_R 
#  = - (a_L * identity_R) * (parity_L * a_R)

# Given two fermion systems with states
# |L> = f(a_L_dag) |0>_L and |R> = g(a_R_dag) |0>_R
# the product state is 
# |LR> = f(a_L_dag_) g(a_R_dag_) |0>_L |0>_R

# Implementation:
# if |L> = f(a_L_dag) |0>_L 
#        = \sum_{n, p} f{n, p} * a_L_dag[0]^p[0] ... a_L_dag[n]^p[n] * |0>_L
#        = \sum_{n, p} f{n, p} * |{n,p}>_L
# and
#    |R> = g(a_R_dag) |0>_R 
#        = \sum_{n, p} g{n, p} * a_R_dag[0]^p[0] ... a_R_dag[n]^p[n] * |0>_R
#        = \sum_{n, p} g{n, p} * |{n,p}>_R
# then
#    |LR> = sum_{n,m,p,q} f{n, p} * f{m, q} 
#           * a_L_dag_[0]^p[0] ... a_L_dag_[n]^p[n]
#           * a_R_dag_[0]^q[0] ... a_R_dag_[m]^q[m] * |0>_L * |0>_R (note underscores in operators)
#         = sum_{n,m,p,q} f{n, p} * g{m, q} |{n,p,m,q}>_LR
#         = sum_{ind} f{ind_L_n, ind_L_p} * g{ind_R_m, ind_R_q} * |{ind_L_n,ind_L_p,ind_L_m,ind_L_q}>_LR
# here ind is the numeration of the "product" basis in the order consistent with the matrix kron 

# The partial trace operation should satisfy the following
# physical constraints:
# 1). 
# For |Psi> = f(a_L_dag_) g(a_R_dag_) |0>_L |0>_R
# rho_L = Tr_R[ |Psi><Psi| ] =  g(a_R_dag) |0>_R <0|_R g^*(a_R)
# rho_R = Tr_L[ |Psi><Psi| ] =  f(a_L_dag) |0>_L <0|_L f^*(a_L)
# provided |R> =  g(a_R_dag) |0>_R is normalized and
#          |L> =  f(a_L_dag) |0>_L is normalized
# 2).
# For |Psi> = \sum_k c_k f_k(a_L_dag_) g_k(a_R_dag_) |0>_L |0>_R
# rho_L = Tr_R[ |Psi><Psi| ] =  \sum_k |c_k|^2 * g_k(a_R_dag) |0>_R <0|_R g_k^*(a_R)
# rho_R = Tr_L[ |Psi><Psi| ] =   \sum_k |c_k|^2 * f_k(a_L_dag) |0>_L <0|_L f_k^*(a_L)
# provided <Li|Lj> = delta_ij and <Ri|Rj> = delta_ij
# where |Lj> = f_j(a_L_dag) |0>_L
#       |Rj> = g_j(a_R_dag) |0>_R
#
# We implement this operation as
# rho_R = Tr_L[ |Psi><Psi| ] 
#    = \sum_{n,m,p,q} rho_R{n,m,p,q} * a_R_dag_[0]^p[0] ... a_R_dag_[n]^p[n]
#                 * |0>_R <0|_R
#                 * a_R_[m]^q[m] ... a_R_[0]^q[0]  (note inverse ordering!)
# where
#    rho_R{n,m,p,q} = \sum{n_L,m_L,p_L,q_L} Psi{n_L,p_L,n,p} Psi^*{m_L,q_L,m,q}
# for |Psi> =  \sum{n_L,p_L,n,p} Psi{n_L,p_L,n,p} |{n_L,p_L,n,p}>_LR
#  which satisfy requirements 1) and 2)                     

class bipartite:
    def __init__(self, L_dimension, R_dimension):
        self.L_dimension = L_dimension
        self.R_dimension = R_dimension
        self.all_states = list(self.state_generator())
        self.enumerated_states = {state: idx for idx, state in enumerate(self.all_states)}
        self.dimension = len(self.all_states)
        
    def state_generator(self):
        for l in range(self.L_dimension):
            for r in range(self.R_dimension):
                yield (l, r)
                
    def index_of(self, state):
        return self.enumerated_states[state]
    
    def vector_with(self, state):
        v = np.zeros(self.dimension, dtype = complex)
        v[self.index_of(state)] = 1.0
        return v
    
    # given wavefunction of the left part psi_L
    # and wavefunction of the right part psi_R,
    # construct their tensor product
    # (i.e. joint quantum state of the bipartite
    #  psi_L \otimes psi_R)
    def kron(self, psi_L, psi_R):
        psi = np.zeros(self.dimension, dtype = complex)
        
        for i in range(self.L_dimension):
            for j in range(self.R_dimension):
                ind = self.index_of((i, j))
                psi[ind] = psi_L[i] * psi_R[j]
                
        return psi
        
    # vacuum state
    # we assume that left and right parts 
    # enumerate their vacuum as zero's state:
    # index_of(vac) == 0
    def vac(self):
        state = np.zeros(self.dimension, dtype = complex)
        state[0] = 1.0
        return state
        
    # reduce the state of a bipartite system by tracing out left part
    def trace_out_L(self, psi):
        rho_R = np.zeros((self.R_dimension, self.R_dimension), dtype = complex)
        
        for i in range(self.R_dimension):
            for j in range(self.R_dimension):
                rho_ij = 0
                for k in range(self.L_dimension):
                    ind_1 = self.index_of((k, i))
                    ind_2 = self.index_of((k, j))
                    rho_ij += psi[ind_1].conj() * psi[ind_2]
                rho_R[i, j] = rho_ij
                
        return rho_R
    
    # reduce the state of a bipartite system by tracing out right part
    def trace_out_R(self, psi):
        rho_L = np.zeros((self.L_dimension, self.L_dimension), dtype = complex)
        
        for i in range(self.L_dimension):
            for j in range(self.L_dimension):
                rho_ij = 0
                for k in range(self.R_dimension):
                    ind_1 = self.index_of((i, k))
                    ind_2 = self.index_of((j, k))
                    rho_ij += psi[ind_1].conj() * psi[ind_2]
                rho_L[i, j] = rho_ij
                
        return rho_L