#!/usr/bin/env python
# coding: utf-8

import numpy as np
from numpy import linalg as LA
from scipy import sparse
from scipy.sparse import coo_matrix
from scipy.sparse import identity
from scipy.sparse import diags



class fock_space:
    def __init__(self, num_modes, max_total_occupation, statistics, max_local_occupations = None):

        self.global_exc = max_total_occupation #internal parameter
        self.statistics = statistics #internal parameter
            
        if statistics == 'Bose':
            
            if max_local_occupations is None:
        #1)
                self.modes = num_modes
        #2)
                self.local_exc = np.full(self.modes, max_total_occupation)
                self.j = np.full(self.modes, max_total_occupation)/2
                #self.j = (np.full(self.modes, max_total_occupation) - 1)/2
            else:
        #1)
                self.modes = num_modes #len(about_excitations)
        #2)
                self.local_exc = np.array(max_local_occupations)
                self.j = np.array(max_local_occupations)/2
        
        elif statistics == 'Fermi':
            if max_local_occupations is None:
        #1)
                self.modes = num_modes
        #2)
                self.local_exc = np.full(self.modes, 1)
            else:
        #1)
                self.modes = num_modes
        #2)
                self.local_exc = np.array(max_local_occupations)
                
        else:
            raise Exception('Сhoose the statistics: Bose or Fermi') 
            
        if not max_local_occupations is None and not len(max_local_occupations) == num_modes:
            raise Exception("Number of modes should be equal to number of local occupation constrains")
        
        self.local_exc1 = np.array(self.local_exc+1) #internal parameter
        
        #3)
        self.states_list = list(self.states_generator())
        
        #4)
        self.find_index = {state: idx for idx, state in enumerate(self.states_list)}
        
        #5)
        self.dimension  = len(self.states_list)
        
        #6)
        self.emptyH = coo_matrix((self.dimension , self.dimension ), dtype = complex).tocsc()
        
        #7)
        self.eye = sparse.eye(self.dimension ).tocsc()
        
        
        if statistics == 'Bose':
            
        #7.5) angular momentum lowering operator
            self.j_m=[]
            current_state = []
            #m = []
            for k in range(self.modes):
                row = np.zeros(self.dimension )
                col = np.arange(self.dimension , dtype=int)
                data = np.zeros(self.dimension )
                for i in range(self.dimension ):
                    
                    
                    
                    if self.states_list[i][k]==0:
                        row[i] = i
                    else:
                        current_state = list(self.states_list[i])
                        current_state[k]-=1
                        p = self.index(current_state)
                        row[i] = p
                        
                        m = self.states_list[i][k] - self.j[k]
                        
                        data[i]=(self.j[k] * (self.j[k] + 1) - m * (m - 1))**0.5
                A = coo_matrix((data, (row, col)), shape=(self.dimension , self.dimension ), dtype = complex)
                self.j_m.append(A.tocsc())
            
        #8)    
            self.annihilate=[]
            current_state = []
            for k in range(self.modes):
                row = np.zeros(self.dimension )
                col = np.arange(self.dimension , dtype=int)
                data = np.zeros(self.dimension )
                for i in range(self.dimension ):
                    if self.states_list[i][k]==0:
                        row[i] = i
                    else:
                        current_state = list(self.states_list[i])
                        current_state[k]-=1
                        p = self.index(current_state)
                        row[i] = p
                        data[i]=(self.states_list[i][k])**0.5
                A = coo_matrix((data, (row, col)), shape=(self.dimension , self.dimension ), dtype = complex)
                self.annihilate.append(A.tocsc()) 
        
        #8.5) 
        
            self.j_p=[]
            current_state = []
            for k in range(self.modes):
                row = np.zeros(self.dimension )
                col = np.arange(self.dimension , dtype=int)
                data = np.zeros(self.dimension )
                for i in range(self.dimension ):
                    if self.states_list[i][k] == self.local_exc[k] or sum(self.states_list[i]) == self.global_exc:#!!!!!!!!!!!!!!!!!!!!!!
                        row[i] = i 
                    else:
                        current_state = list(self.states_list[i])
                        current_state[k]+=1
                        p = self.index(current_state)
                        row[i] = p
                        
                        m = self.states_list[i][k] - self.j[k]
                        data[i]=(self.j[k] * (self.j[k] + 1) - m * (m + 1))**0.5
                A = coo_matrix((data, (row, col)), shape=(self.dimension , self.dimension ), dtype = complex)
                self.j_p.append(A.tocsc())
                
        #8.6)
        
            self.j_x = []
            self.j_y = []
            for k in range(self.modes):
                self.j_x.append((self.j_p[k] + self.j_m[k]) / 2)
                self.j_y.append((self.j_p[k] - self.j_m[k]) / 2 / 1j)
            
        #8.7)
            self.j_z = []
            current_state = []
            for k in range(self.modes):
                row = np.zeros(self.dimension)
                col = np.arange(self.dimension, dtype=int)
                data = np.zeros(self.dimension)
                for i in range(self.dimension):
                    current_state = list(self.states_list[i])
                    p = self.index(current_state)
                    row[i] = p
                    m = self.states_list[i][k] - self.j[k]
                    data[i] = m
                A = coo_matrix((data, (row, col)), shape=(self.dimension , self.dimension ), dtype = complex)
                self.j_z.append(A.tocsc())
        
        #8.8)
        
        
        
        #9)
            self.create=[]
            current_state = []
            for k in range(self.modes):
                row = np.zeros(self.dimension )
                col = np.arange(self.dimension , dtype=int)
                data = np.zeros(self.dimension )
                for i in range(self.dimension ):
                    if self.states_list[i][k] == self.local_exc[k] or sum(self.states_list[i]) == self.global_exc:#!!!!!!!!!!!!!!!!!!!!!!
                        row[i] = i 
                    else:
                        current_state = list(self.states_list[i])
                        current_state[k]+=1
                        p = self.index(current_state)
                        row[i] = p
                        data[i]=(self.states_list[i][k]+1)**0.5
                A = coo_matrix((data, (row, col)), shape=(self.dimension , self.dimension ), dtype = complex)
                self.create.append(A.tocsc())
       
        elif statistics == 'Fermi':
        
        #8)
            self.annihilate=[]
            current_state = []
            for k in range(self.modes):
                row = np.zeros(self.dimension )
                col = np.arange(self.dimension , dtype=int)
                data = np.zeros(self.dimension )
                for i in range(self.dimension ):
                    if self.states_list[i][k]==0:
                        row[i] = i
                    else:
                        current_state = list(self.states_list[i])
                        y = sum(current_state[:k])
                        current_state[k] = 0
                        p = self.index(current_state)
                        row[i] = p
                        data[i]=(-1)**y*(self.states_list[i][k])**0.5
                A = coo_matrix((data, (row, col)), shape=(self.dimension , self.dimension ), dtype = complex)
                self.annihilate.append(A.tocsc())
                              
         #9)       
            self.create=[]
            current_state = []
            for k in range(self.modes):
                row = np.zeros(self.dimension )
                col = np.arange(self.dimension , dtype=int)
                data = np.zeros(self.dimension )
                for i in range(self.dimension ):
                    if self.states_list[i][k] == self.local_exc[k] or sum(self.states_list[i]) == self.global_exc:
                        row[i] = i 
                    else:
                        current_state = list(self.states_list[i])
                        y = sum(current_state[:k])
                        current_state[k] = 1
                        p = self.index(current_state)
                        row[i] = p
                        data[i]=(-1)**y*(self.states_list[i][k]+1)**0.5
                A = coo_matrix((data, (row, col)), shape=(self.dimension , self.dimension ), dtype = complex)
                self.create.append(A.tocsc())
            
        else:
            print('Сhoose the statistics: Bose or Fermi')
                  
                
    def sigmax(self, i):

        return(self.annihilate[i] + self.create[i])
    
    
    def sigmay(self, i):
 
        return(-1j*(-self.annihilate[i] + self.create[i]))
    
    
    def sigmaz(self, i):

        return(- self.annihilate[i]@self.create[i] + self.create[i]@self.annihilate[i])
  
        
    def states_generator(self):
        #Generates all the possible states for given Fock space
        current_state = (0,)*len(self.local_exc1)
        n = 0
        while True:
            yield current_state
            j = len(self.local_exc1) - 1
            current_state = current_state[:j] + (current_state[j]+1,)
            n += 1
            while n > self.global_exc or current_state[j] >= self.local_exc1[j]:
                j -= 1
                if j < 0:
                    return
                n -= current_state[j+1] - 1
                current_state = current_state[:j] + (current_state[j]+1, 0) + current_state[j+2:]
      
        
    def occupations(self,i):
     
        if i >= self.dimension :
            print('the number is out of range')
            
        else:
            return(np.array(self.states_list[i]))
    
    
    def index(self, state):
       
        if len(state) != self.modes:
            print('incorrect size of an array')
        else:
            s = tuple(state)
            return(self.find_index[s])
    


        
class fock_space_kron:
    def __init__(self, f1, f2):

        self.f1 = f1
        self.f2 = f2
        
        #1)
        self.modes = f1.modes + f2.modes

        #2)
        self.dimension  = f1.dimension * f2.dimension

        #3)
        self.emptyH = coo_matrix((self.dimension , self.dimension ), dtype = complex).tocsc()

        #4)
        self.eye = sparse.eye(self.dimension ).tocsc()
       
        
        
        if (f1.statistics=='Fermi') and (f2.statistics=='Fermi'):
                       
            data1 = np.zeros(f1.dimension)
            data2 = np.zeros(f2.dimension)
            
            for i in range(f1.dimension):
                k = sum(f1.occupations(i))
                data1[i] = (-1)**k
            for i in range(f2.dimension):
                k = sum(f2.occupations(i))
                data2[i] = (-1)**k
        
        #4.1)
            self.parity_f1 = diags(data1)
        #4.2)
            self.parity_f2 = diags(data2)
        
        #5)
            self.annihilate=[]
        
            for k in range(f1.modes):
                self.annihilate.append(sparse.kron(f1.annihilate[k], f2.eye).tocsc())
            for k in range(f1.modes, self.modes):
                self.annihilate.append(sparse.kron(self.parity_f1, f2.annihilate[k-f1.modes]).tocsc())
        #6)
            self.create=[]
        
            for k in range(f1.modes):
                self.create.append(sparse.kron(f1.create[k], f2.eye).tocsc())
            for k in range(f1.modes, self.modes):
                self.create.append(sparse.kron(self.parity_f1, f2.create[k-f1.modes]).tocsc())
        
        else:
            
        #5)
            self.annihilate=[]
        
            for k in range(f1.modes):
                self.annihilate.append(sparse.kron(f1.annihilate[k], f2.eye).tocsc())
            for k in range(f1.modes, self.modes):
                self.annihilate.append(sparse.kron(f1.eye, f2.annihilate[k-f1.modes]).tocsc())
        #6)
            self.create=[]
        
            for k in range(f1.modes):
                self.create.append(sparse.kron(f1.create[k], f2.eye).tocsc())
            for k in range(f1.modes, self.modes):
                self.create.append(sparse.kron(f1.eye, f2.create[k-f1.modes]).tocsc())
                
        #7)
        if f1.statistics=='Bose':
            
            self.j_m=[]
            for k in range(f1.modes):
                self.j_m.append(sparse.kron(f1.j_m[k], f2.eye).tocsc())
                
            self.j_p=[]
            for k in range(f1.modes):
                self.j_p.append(sparse.kron(f1.j_p[k], f2.eye).tocsc())
                
            self.j_x=[]
            for k in range(f1.modes):
                self.j_x.append(sparse.kron(f1.j_x[k], f2.eye).tocsc())
                
            self.j_y=[]
            for k in range(f1.modes):
                self.j_y.append(sparse.kron(f1.j_y[k], f2.eye).tocsc())
            
            self.j_z=[]
            for k in range(f1.modes):
                self.j_z.append(sparse.kron(f1.j_z[k], f2.eye).tocsc())
        
            self.j = np.copy(f1.j)
            
    def sigmax(self, i):
        if (i<self.f1.modes):
             return(sparse.kron(self.f1.sigmax(i), self.f2.eye).tocsc())
        else:
            return(sparse.kron(self.f1.eye, self.f2.sigmax(i-self.f1.modes)).tocsc())

        
    def sigmay(self, i):
       
        if (i<self.f1.modes):
             return(sparse.kron(self.f1.sigmay(i), self.f2.eye).tocsc())
        else:
            return(sparse.kron(self.f1.eye, self.f2.sigmay(i-self.f1.modes)).tocsc())

        
    def sigmaz(self, i):
        
        if (i<self.f1.modes):
             return(sparse.kron(self.f1.sigmaz(i), self.f2.eye).tocsc())
        else:
            return(sparse.kron(self.f1.eye, self.f2.sigmaz(i-self.f1.modes)).tocsc())
    
    
    def occupations(self,i):
        
        if i >= self.dimension :
            print('the number is out of range')
        else:
            state = np.zeros(self.modes, dtype = int)
            state[:self.f1.modes] = self.f1.occupations(i//self.f2.dimension)
            state[self.f1.modes:] = self.f2.occupations(i%self.f2.dimension)
            return(state)
        
        
    def index(self,state):
        
        if state.size != self.modes:
            print('incorrect size of an array')
        else:
            s = list(state)
            return(self.f1.index(state[:self.f1.modes]) * self.f2.dimension + self.f2.index(state[self.f1.modes:]))
        
def real_time_solver(psi0, dt, tmax, H, Q = None, final_state = None):
    K = psi0.size
    Nt = int(tmax/dt)+1
    psi = [psi0, psi0]
    
    if callable(H) == False:
        H_const = H
        def H(t):
            return(H_const)
     
    if Q == None:
        for i in range(1,Nt):
            psi_iter_old = psi0
            psi_iter = np.zeros(K)
            psi_compare = psi0
            while (LA.norm(psi_iter-psi_compare)>10**(-6)):
                s = psi[1]+psi_iter_old
                psi_iter = psi[1]+ dt/2*(-1j) * H(i*dt+dt/2)@s
                psi_compare = psi_iter_old
                psi_iter_old = psi_iter
            psi[1] = psi_iter
            
        return (psi[1])
        
    elif type(Q) == list:
        l = len(Q)
        results= np.zeros((l,Nt))
        for j in range(l):
            results[j,0] = abs(np.conj(psi0) @ Q[j] @ psi0)
        
        for i in range(1,Nt):
            psi_iter_old = psi0
            psi_iter = np.zeros(K)
            psi_compare = psi0
            while (LA.norm(psi_iter-psi_compare)>10**(-6)):
                s = psi[1]+psi_iter_old
                psi_iter = psi[1]+ dt/2*(-1j) * H(i*dt+dt/2)@s
                psi_compare = psi_iter_old
                psi_iter_old = psi_iter
            psi[1] = psi_iter
            
            for j in range(l):
                results[j,i] = abs(np.conj(psi[1]) @ Q[j] @ psi[1])
        
    else:
        results = np.zeros(Nt)
        results[0] = abs(np.conj(psi0) @ Q @ psi0)
        
        for i in range(1,Nt):
            psi_iter_old = psi0
            psi_iter = np.zeros(K)
            psi_compare = psi0
            while (LA.norm(psi_iter-psi_compare)>10**(-6)):
                s = psi[1]+psi_iter_old
                psi_iter = psi[1]+ dt/2*(-1j) * H(i*dt+dt/2)@s
                psi_compare = psi_iter_old
                psi_iter_old = psi_iter
            psi[1] = psi_iter
            results[i] = abs(np.conj(psi[1]) @ Q @ psi[1])
    if final_state != None :
        return (results, psi[1])
    else:
        return(results)
    
        


# In[ ]:




