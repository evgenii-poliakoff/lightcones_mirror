import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import sys
import numpy as np
import tools_changed_by_me
from tools_changed_by_me import mv
import time
from evolution_chained2_kicked import evolution_chained2_kicked
import math
import random
import pathlib
import scipy.sparse
from numpy import linalg as LA

class bath:
    
    def __init__(self):
        self.intervals = None
        self.couplings = None
        self.couplings_full = None
        self.rotations = None
        self.m_max = None
        self.n_rel = None
        self.t = None
        self.dt = None
        self.nt = None
        
    def get_jump_times(self):
        
        n_out = 0
        jt = []
        
        for i in self.intervals:
            n_out_ = i[2]
            if n_out_ > n_out:
                n_out = n_out_
                jt.append(self.t[i[0]])
                
        return jt
            
    def save_to_file(self, fout):
        
        p = os.path.join(sys.path[0], fout, "time.txt")
        os.makedirs(os.path.dirname(p), exist_ok = True)
        with open(p, "w") as f:
            for i in range(self.nt):
                print(str(self.t[i]), file = f)
        
        p = os.path.join(sys.path[0], fout, "intervals.txt")
        os.makedirs(os.path.dirname(p), exist_ok = True)
        with open(p, "w") as f:
            for i in self.intervals:
                a = i[2]
                b = i[3]
                print(i[0], "\t", i[1], "\t", a, "\t", b, file = f)
                
        np.set_printoptions(linewidth=np.inf)
                
        p = os.path.join(sys.path[0], fout, "couplings.txt")
        os.makedirs(os.path.dirname(p), exist_ok = True)
        with open(p, "w") as f:
            for i in self.intervals:
                b = i[2]
                a = i[3]
                for ti in range(i[0], i[1]):
                    clp = self.couplings[ti]
                    print('\t'.join(map(str, clp.real)), file = f)
                    print('\t'.join(map(str, clp.imag)), file = f) 

        p = os.path.join(sys.path[0], fout, "rotations.txt")
        os.makedirs(os.path.dirname(p), exist_ok = True)
        with open(p, "w") as f:

            ii = 0
            for i in self.intervals:

                a = i[2]
                b = i[3]
                
                w = self.rotations[ii]
                #w = i[4]

                for p in range(b - a):
                    if not w is None:
                        print('\t'.join(map(str, w[p, :].real)), file = f)
                    else:
                        print('\t'.join(map(str, [0.0]*(b - a))), file = f)

                for p in range(b - a):
                    if not w is None:
                        print('\t'.join(map(str, w[p, :].imag)), file = f)
                    else:
                        print('\t'.join(map(str, [0.0]*(b - a))), file = f)
        
                ii = ii + 1
        
    @classmethod
    def load_from_file(cls, fin):
            
        bth = cls() 
            
        #----------------------------------------------------

        intervals = []

        try:
            p = os.path.join(sys.path[0], fin, "intervals.txt")
            with open(p) as f:
                while True:
                    l = [int(e) for e in next(f).split()] 
                    intervals.append(l)
        except StopIteration as e:
            pass

        bth.intervals = intervals
        
        #----------------------------------------------------

        couplings = []

        try:
            p = os.path.join(sys.path[0], fin, "couplings.txt")
            with open(p) as f:
                while True:
                    re = np.asarray([float(e) for e in next(f).split()])
                    im = np.asarray([float(e) for e in next(f).split()])
                    couplings.append(re + 1j * im)
        except StopIteration as e:
            pass
        
        bth.couplings = couplings

        #----------------------------------------------------

        rotations = []

        m_max = 0
        n_rel = 0

        p = os.path.join(sys.path[0], fin, "rotations.txt")
        with open(p) as f:

            for i in intervals:

                a = i[2]
                b = i[3]

                n_rel = max(n_rel, b)
                m_max = max(m_max, b - a)

                re = np.zeros((b - a, b - a), dtype = np.cdouble)
                for p in range(b - a):
                    re[p, :] = np.asarray([float(e) for e in next(f).split()])

                im = np.zeros((b - a, b - a), dtype = np.cdouble)
                for p in range(b - a):
                    im[p, :] = np.asarray([float(e) for e in next(f).split()])

                w = re + 1j * im

                rotations.append(w)

        bth.rotations = rotations
        bth.m_max = m_max
        bth.n_rel = n_rel

        #----------------------------------------------------

        p = os.path.join(sys.path[0], fin, "time.txt")
        with open(p) as f:
            tg = np.loadtxt(f)
        ntg = tg.size
        dt = tg[1] - tg[0]

        bth.t = tg
        bth.dt = dt
        bth.nt = ntg
        
        return bth
    
    @classmethod
    def from_chain(cls, chain, n_sites = -1, rel_tol = 10**(-3), ring_size = 4, t_max = 100, dt = 0.01, visualize = False):
        
        bth = cls() 
            
        #----------------------------------------------------
        
        es = np.array(chain[0][:])
        hs = np.array(chain[1][:])
        
        if not es.size == hs.size + 1:
            raise Exception("One should have nh + 1 = ne, where nh = number of hoppings, ne = number of on-site energies")

        if n_sites < 0:
            n_sites = es.size
        else:
            es = es[:n_sites]
            hs = hs[:n_sites-1]
            
        t = t_max

        bth.n_sites = n_sites
        
        tg = np.arange(0, t + dt, dt)
        ntg = tg.size
        
        bth.t = tg
        bth.dt = dt
        bth.nt = ntg
        
        #-------------------------------------------------------------------------------------------------
        
        H = tools_changed_by_me.tridiag(es, hs)
        
        psi0 = np.zeros(n_sites, dtype = np.cdouble)
        psi0[0] = 1

        psi_lc = np.zeros((n_sites, ntg), dtype = np.cdouble)

        def Ht(t):
            return H

        if visualize:
            bth.Ht = Ht

        for i, psi in tools_changed_by_me.evolutionpy(start_index = 0, end_index = ntg, H = Ht, dt = dt, initial_state = psi0):
            psi_lc[:, i] = np.copy(psi)

        if visualize:
            bth.couplings_chain = np.copy(psi_lc)

            H_ = H.todense()
            e_, U_ = tools_changed_by_me.find_eigs_ascending(H_)
            bth.couplings_star = np.zeros((n_sites, ntg), dtype = complex)
            for i in range(ntg):
                bth.couplings_star[:, i] = U_.T.conj() @ psi_lc[:, i]
            
        #-------------------------------------------------------------------------------------------------
        
        n_guard = 3
        revival_tolerance = 10**(-9)
        revival = np.amax(np.abs(psi_lc[-n_guard :, :]))

        if revival > revival_tolerance:
            raise Exception('Revival: Increase the length of chain')
        
        #-------------------------------------------------------------------------------------------------

        rho_lc = np.zeros((n_sites, n_sites), dtype = np.cdouble)

        for i in range(0, ntg):
            psi = tools_changed_by_me.as_column_vector(psi_lc[:, i])
            rho_lc += tools_changed_by_me.dyad(psi, psi) * dt

        tools_changed_by_me.make_hermitean(rho_lc)

        #--------------------------------------------------------------------------------------------------
        
        pi, U_rel = tools_changed_by_me.find_largest_eigs(rho_lc)

        lr_metric = pi - rel_tol * pi[0]
        inside_lightcone = lr_metric > 0

        pi_rel = pi[inside_lightcone]
        n_rel =  np.size(pi_rel)
        bth.n_rel = n_rel
        U_rel_full = np.copy(U_rel)
        U_rel = U_rel[:, inside_lightcone]

        rho_lc_rel = np.diag(pi_rel.astype('cdouble'))

        #--------------------------------------------------------------------------------------------------

        psi_lc_rel = U_rel.T.conj() @ psi_lc

        #--------------------------------------------------------------------------------------------------

        rho_ret = np.copy(rho_lc_rel)

        #--------------------------------------------------------------------------------------------------

        times_in = []
        n_in = [n_rel]

        U_min = np.eye(n_rel, dtype = np.cdouble)

        n = n_rel

        for i in reversed(range(0, ntg)):

            pi_min, _ = tools_changed_by_me.find_smallest_eigs(rho_ret, 1)
            pi_max, _ = tools_changed_by_me.find_largest_eigs(rho_ret, 1)

            lr_metric = pi_min - rel_tol * pi_max
            outside_lightcone = lr_metric < 0

            if outside_lightcone:
                pi, U = tools_changed_by_me.find_eigs_descending(rho_ret)
                psi_lc_rel[: n, :] = U.T.conj() @ psi_lc_rel[: n, :]
                U_min[: n, :] = U.T.conj() @ U_min[: n, :]
                rho_ret = np.diag(pi[: -1].astype('cdouble'))
                times_in.insert(0, i + 1)
                n = n_rel - len(times_in)
                n_in.insert(0, n) 

            psi = tools_changed_by_me.as_column_vector(psi_lc_rel[: n, i])
            rho_ret -= tools_changed_by_me.dyad(psi, psi) * dt

            tools_changed_by_me.make_hermitean(rho_ret)

        #-------------------------------------------------------------------------------------------------
        
        if visualize:
            bth.couplings_min = np.copy(psi_lc_rel)
            bth.U_min =  np.copy(U_rel_full.T.conj())
            bth.U_min[:n_rel, :] = U_min @ bth.U_min[:n_rel, :] 
        
        #-------------------------------------------------------------------------------------------------

        intervals_in = []

        i_left = 0

        for i_right, n in zip(times_in + [ntg], n_in):
            intervals_in.append((i_left, i_right, n))
            i_left = i_right

        #--------------------------------------------------------------------------------------------------
        if visualize:
            bth.times_in = []
            bth.itimes_in = []
            for i in intervals_in:
                bth.times_in.append(tg[i[0]])
                bth.itimes_in.append(i[0])
        
        #--------------------------------------------------------------------------------------------------

        rho_ret =  np.zeros((n_rel, n_rel), dtype = np.cdouble)

        max_n_coupled = ring_size - 1

        rho_adv = U_min @ np.copy(rho_lc_rel) @ U_min.T.conj()
        tools_changed_by_me.make_hermitean(rho_adv)

        psi_lc_out = np.copy(psi_lc_rel)

        if visualize:
            U_cdia = np.copy(bth.U_min)
        
        intervals_out = []

        n_out = 0

        for i in intervals_in:

            begin = i[0]
            end = i[1]
            n_in = i[2]

            n_coupled = n_in - n_out

            w = None

            n_out_new = n_out

            if n_coupled > max_n_coupled:

                rho_cdi = rho_adv[n_out : n_in, n_out : n_in]
                pi, U = tools_changed_by_me.find_eigs_ascending(rho_cdi)
                psi_lc_out[n_out : n_in, i[0] :] = U.T.conj() @ psi_lc_out[n_out : n_in, i[0] :]

                if visualize:
                    U_cdia[n_out : n_in, :] = U.T.conj() @ U_cdia[n_out : n_in, :]
                
                rho_adv[n_out : n_in, n_out : ] = U.T.conj() @ rho_adv[n_out : n_in, n_out : ] 
                rho_adv[n_out : , n_out : n_in] = rho_adv[n_out : , n_out : n_in] @ U 

                n_out_new = n_in - max_n_coupled

                w = U.T.conj()

            intervals_out.append((begin, end, n_in, n_out, w))

            n_out = n_out_new

            for j in range(i[0], i[1]):

                psi = tools_changed_by_me.as_column_vector(psi_lc_out[n_out : , j])
                rho_adv[n_out : , n_out : ] -= tools_changed_by_me.dyad(psi, psi) * dt

        if visualize:
            bth.U_cdia = U_cdia
            bth.times_out = []
            bth.itimes_out = []
            n_out = 0
            for i in intervals_out:
                if i[3] > n_out:
                    bth.times_out.append(tg[i[0]])
                    bth.itimes_out.append(i[0])
                    n_out = i[3]
        
        #----------------------------------------------------------------------------------

        from scipy.linalg import block_diag

        intervals_out_c = []

        min_duration = 0.5

        for i in intervals_out:

            if (i[4] is None):
                intervals_out_c.append(i)
                continue

            duration = (i[1] - i[0]) * dt

            while duration < min_duration:

                i_ = intervals_out_c.pop()

                u = i[4]

                if i_[3] < i[3]:
                    d = i[3] - i_[3]
                    u = block_diag(np.eye(d, dtype = np.cdouble), u)

                if not i_[4] is None:

                    u_ = i_[4]

                    if i_[2] < i[2]:
                        d = i[2] - i_[2]
                        u_ = block_diag(u_, np.eye(d, dtype = np.cdouble))    

                    u = u @ u_  

                i = (i_[0], i[1], i[2], i_[3], u)

                duration = (i[1] - i[0]) * dt

            intervals_out_c.append(i)   

        #-------------------------------------------------------------------

        from scipy.linalg import eig

        intervals_r = []

        for i in intervals_out_c:

            u = i[4]
            if (not u is None):
                duration = (i[1] - i[0]) * dt

                e_, v_ = eig(u)
                e_ = np.log(e_ + 0j) / duration
                u = v_ @ np.diag(e_) @ v_.conj().T
                
            intervals_r.append((i[0], i[1], i[2], i[3], u))

        #print('---------------------')
        #print(intervals_r)
        #print('---------------------')
            
        #--------------------------------------------------------------------

        from scipy.linalg import expm

        couplings = np.copy(psi_lc_rel)

        u = np.eye(n_rel, dtype = np.cdouble)

        for i in intervals_r:
            for j in range(i[0], i[1]):
                couplings[:, j] = u @ couplings[:, j]

                w = i[4]
                if not w is None:
                    du = expm(dt * w)
                    a = i[2]
                    b = i[3]
                    u[b : a, :] = du @ u[b : a, :]

        #----------------------------------------------------------------------

        m_max = 0
        
        bth.intervals = []
        
        for i in intervals_r:

            a = i[2]
            b = i[3]

            m_max = max(m_max, a - b)
            
            bth.intervals.append([i[0], i[1], b, a])
            
        bth.m_max = m_max
            
        #---------------------------------------------------------------------
        
        bth.couplings = []
        for i in bth.intervals:
            a = i[2]
            b = i[3]
            #print('range: ', i[0], ' ', i[1])
            #print('a = ', a, 'b = ', b)
            for ti in range(i[0], i[1]):
                bth.couplings.append(couplings[a : b, ti])

        if visualize:
            bth.couplings_cdia = couplings
        
        #---------------------------------------------------------------------
        
        rotations = []
        for i in intervals_r:
            b = i[2]
            a = i[3]
            r = b - a
            w = i[4]
            if not w is None:
                w = w[0 : r, :]
            else:
                w = np.zeros((r, r), dtype = complex)
            rotations.append(w)
            
        bth.rotations = rotations
        
        #---------------------------------------------------------------------
        
        return bth

class job_opts:
    def __init__(self):
        self.init = None
        self.Hs = None
        self.Vs = None
        self.Os = None
        self.bth = None
        self.csize = None
        self.tmax = None
        self.fout = None
        self.compute_probs  = False
        self.compute_dmatrices = False
        self.model = None
    
def job(image, opts):
    
    #
    
    Hs = opts.Hs
    Vs = opts.Vs
    Os = opts.Os 
    bth = opts.bth 
    csize = opts.csize
    nquanta = opts.model.nquanta
    tmax = opts.tmax
    fout = opts.fout
    compute_probs = opts.compute_probs
    compute_dmatrices = opts.compute_dmatrices
    quench = opts.quench
    
    #-----------------------------------------------------

    t = bth.t
    nt = bth.nt
    dt = bth.dt
    nt_ = np.where(t <= tmax)[0][-1]
    t_ = t[0 : nt_]

    #-----------------------------------------------------

    p = os.path.join(sys.path[0], fout, "time.txt")
    os.makedirs(os.path.dirname(p), exist_ok = True)

    with open(p, "w") as f:
        for i in range(nt_):
            print(str(t[i]), file = f)
            
    #-----------------------------------------------------

    #print('computing sparse matrices...', flush = True)

    Hs_ = Hs(0)
    dim_s = Hs_.shape[0]
    id_s = scipy.sparse.eye(dim_s).tocsc()
    
    m_max = bth.m_max
    
    model = opts.model
    
    m = model.m
    lo = model.lo

    id_e = model.id_e
    
    #la = tools.kron(id_s, lo.a())
    #la_dag = tools.kron(id_s, lo.a_dag())
    
    a_hat = model.a
    a_hat_dag = model.a_dag
    
    O = model.O
    
    #Hse = model.Hs
    
    #V_dag = model.V_dag
    
    def Hse(ti):
        return tools_changed_by_me.kron(Hs(ti * dt), id_e)
    
    def V_dag(ti):
        return tools_changed_by_me.kron(Vs(ti * dt), id_e).conj().T
    
    zero_op = model.zero_op
 
    dimension = model.dimension 
    
    with lock:
        print('running chunk ', image, ": m_max = ", m_max ,  ", dim = ", dimension,  " ... ", flush = True)
    
    #-----------------------------------------------------

    #with lock:
    #    print('...done', flush = True)
    
    #### Initial condition

    psi_ini = np.zeros(dimension, dtype = complex)
    psi_ini[0] = 1

    ####
    n_rel = bth.n_rel
    
    to_ring = [ _ % m_max for _ in range(0, n_rel)]

    ###

    start_time = time.time()
    
    seed = 1000 * math.pi * image + pow(image, 2) 
    random.seed(seed)
    
    av_O = [(np.zeros(nt_, dtype = complex), o_) for o_ in O]
    j_probs_chunk = []
    dm_chunk = []
    dms = []

    psi = np.zeros(dimension, dtype = complex)
    psi_mid = np.zeros(dimension, dtype = complex)
    psi_mid_next = np.zeros(dimension, dtype = complex)
    psi_buff = np.zeros(dimension, dtype = complex)

    def eval_o(ti, psi):
        for av in av_O:
            mv(av[1], psi, psi_buff)
            av[0][ti] = av[0][ti] + np.vdot(psi, psi_buff)

    def eval():
        
        
        j_probs = []
        
        if compute_probs:
            #print('adding jump for ' + str(i[0]))
            j_probs_chunk.append(j_probs)
            #print(j_probs)     
        
        intervals = bth.intervals
        couplings = bth.couplings
        rotations = bth.rotations
    
        psi_begin = np.copy(psi_ini)
    
        first_in_chain = 1
    
        ni = 0
    
        for (i, i_) in zip(intervals, [None] + intervals[:-1]):
    
            if i[0] >= nt_:
                return
        
            #print('enter ' + str(i[0]))
        
            i1 = min(i[1], nt_-1)
    
            b = i[3]
            a = i[2]
    
            if not i_ is None:
                a_ = i_[2]
                for q in range(a_, a):
                    measured_mode = to_ring[q] #+ 1
                    psi_begin, j_prob, _, j_index, dm  = lo.quantum_jumpEx(psi_begin, measured_mode)
                    
                    if compute_probs:
                        #print('adding jump for ' + str(i[0]))
                        j_probs.append(j_prob[j_index])
                        #print(j_probs)
                    #psi_begin = lo.quantum_jump(psi_begin, measured_mode)
                    
                    if compute_dmatrices:
                        if len(dm_chunk) == 0:
                            dms.append(dm)
                        else:
                            dm_chunk[ni] += dm
                
            w = rotations[ni]
    
            a_ring = [a_hat[to_ring[_]] for _ in range(a, b)]
            a_ring_dag = [a_hat_dag[to_ring[_]] for _ in range(a, b)]
    
            Hw = zero_op.copy()
            if not w is None:
                for p in range(b - a):
                    for q in range(b - a):
                        Hw += 1j * a_ring_dag[q] @ a_ring[p] * w[q, p].conj()
    
            def Vint(ti):
            
                #print("a")
                #tt_ = sum(couplings[ti] * a_ring)
                #print(couplings[ti])
                #print(tt_)
                #print("b")
                #ttt_ = V_dag(ti) @ tt_
                #print("c")
                V_ = V_dag(ti) @ sum(couplings[ti] * a_ring)
                V_ = V_ + V_.conj().transpose()
                return(V_)
    

            def Ht(ti):
                return Hse(ti) + Vint(ti) + Hw
    
            def Hwt(ti):
                return Hw
    
            eval.Ht_ = None
    
            def begin_step(ti, psi):
            
                if not quench is None:
                    quench(ti * dt, psi)
                
                eval.Ht_ = Ht(ti)
            
                #if (beta > 0):
                #    eval.Ht_ = eval.Ht_ + Vth(ti)
    
            def apply_H(ti, psi_in, psi_out):
                mv(eval.Ht_, psi_in, psi_out)
    
            #for ti, psi in tools.evolution(start_index = i[0], end_index = i[1], H = Ht, dt = dt, initial_state = psi0):
            #    pass
       
            evolution_chained2_kicked(i[0], i1, dt, begin_step, apply_H, eval_o, psi_begin, psi, psi_mid, psi_mid_next, first_in_chain)
            first_in_chain = 0
    
            psi_begin = np.copy(psi)
    
            ni = ni + 1
        
        #print("here!")
        #if compute_probs:
        #    j_probs_chunk.append(j_probs)
        #    print("finished trajectory: ")
        #    print(j_probs_chunk)
       
    for i in range(csize):
        eval()
        
        if compute_dmatrices:
            if len(dm_chunk) == 0:
                dm_chunk[:] = dms
            dms.clear()
            
        #print("hmm!")
    
    end_time = time.time()

    with lock:
        print(' ... chunk ', image, ' execution time: ', end_time-start_time, ' sec', flush = True)

    for i in range(len(av_O)):
        path = os.path.join(sys.path[0], fout, Os[i][0] + "_" + str(image) + ".txt")
        with open(path, "w") as f:
            av = av_O[i]
            print(str(csize) + " " + str(0), file = f)
            for i in range(nt_):
                print(str(av[0][i].real) + " " + str(av[0][i].imag), file = f)
                
    if compute_probs:
        #print('saving probs:')
        #print(j_probs_chunk)
        pa = np.array(j_probs_chunk).T
        path = os.path.join(sys.path[0], fout, "probs" + "_" + str(image) + ".txt")
        np.savetxt(path, pa)
        
    if compute_dmatrices:
        path = os.path.join(sys.path[0], fout, "dmatrices" + "_" + str(image) + ".txt")
        with open(path, "w") as f:
            dm = dm_chunk[0]
            dm_ = dm.flatten()
            ds = len(dm_)
            hh = np.zeros(ds)
            hh[0] = csize
            print('\t'.join(map(str, hh)), file = f)
            
            for dm in dm_chunk:
                dm_ = dm.flatten()
                print('\t'.join(map(str, dm_.real)), file = f)
                print('\t'.join(map(str, dm_.imag)), file = f)
            
        

def job_(i, opts):
    try:
        job(i, opts)
    except:
        import traceback
        traceback.print_exception(*sys.exc_info())
        raise()

def worker_process_initializer(pool_lock):
    global lock
    lock = pool_lock

    #run(cfrom, cto, maxcores, o)
#def run(Hs, Vs, Os, bth, cfrom = 1, cto = 1, csize = 1, maxcores = 4, nquanta = 3, tmax = 100.0, fout = 'results'):
def run(cfrom, cto, maxcores, o):

    Hs = o.Hs
    Vs = o.Vs
    Os = o.Os
    #nquanta = o.nquanta
    tmax = o.tmax
    fout = o.fout
    csize = o.csize
    

    print("Wellcome to lightcone impurity solver version 04.10.2023-23.56")
    
    print("Will compute ", (cto - cfrom + 1) * csize, " trajectories")
    print("on max ", maxcores, " CPU cores")

    print("Will save results into ", fout)
    
    print("Will simulate up to time", tmax)
    
    #print("Max number of coupled quanta: ", nquanta)
             
    def progress_indicator(lock, f):
        global progress
        progress += 1
        with lock:
            print('finished job ', progress, ' out of ', cto - cfrom + 1, flush = True)

    #if __name__ == '__main__':

    import functools
    from multiprocessing import Lock
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures import wait

    lock = Lock()
    
    with ProcessPoolExecutor(max_workers = maxcores, initializer = functools.partial(worker_process_initializer, lock)) as executor:

        with lock:
            print('submitting jobs to pool ...', flush = True)
     
        futures = [executor.submit(job_, i, o) for i in range(cfrom, cto + 1)]
        global progress
        progress = 0
        for f in futures:
            f.add_done_callback(functools.partial(progress_indicator, lock))
        
        wait(futures)
    
        for f in futures:
            exception = f.exception()
            if not exception is None:
                import traceback
                print("Exception:")
                print(exception)
                traceback.print_exc()
        
    print('\nDone!')


class load_probs:
    
    def __init__(self, arange, folder):
        
        chunk_from = arange[0] 
        chunk_to = arange[1]

        pa = None
        
        for i in range(chunk_from, chunk_to):
            p = os.path.join(os.getcwd(), folder,  "probs" + "_" + str(i) + ".txt")
            pa_ = np.loadtxt(p)
            if pa is None:
                pa = pa_
            else:
                pa = np.concatenate((pa, pa_), axis = 1)
            
        self.probabilities = pa
        
class load_dmatrices:
    
    def __init__(self, arange, folder):
        
        nsamples = 0

        chunk_from = arange[0]
        chunk_to = arange[1]

        o = None
        
        for i in range(chunk_from, chunk_to):
            p = os.path.join(os.getcwd(), folder, "dmatrices_" + str(i) + ".txt")
            p = pathlib.Path(p)

            if not p.exists():
                print("Warning: no chunk #", i)
                continue

            with p.open() as f:
                oc = np.loadtxt(f)
                
                if o is None:
                    o = np.zeros(((oc.shape[0] - 1) // 2, oc.shape[1]), dtype = complex)
                
                nsamples += oc[0, 0]
                o += oc[1::2, :] + 1j * oc[2::2, :] 

        if nsamples == 0:
            raise Exception('No simulation data for ' + name + ' in the range ' + str(chunk_from) + ' ... ' + str(chunk_to))
                
        o = o / nsamples
        
        dmatrices = []
        dim = round(np.sqrt(o.shape[1]))
        for i in range(o.shape[0]):
            dmatrices.append(np.reshape(o[i, :], (dim, dim)))
        
        self.dmatrices = dmatrices
        self.nsamples = nsamples
    
class load_results:

    def __init__(self, name, arange, folder):

        # path = os.path.join(sys.path[0], folder, "time.txt")
        path = os.path.join(sys.path[0], folder, "time.txt")
        with open(path, "w") as f:
            np.savetxt(f, np.array(tg).real)

        # with pathlib.Path(os.path.join(os.getcwd(), folder, "time.txt")).open() as f:
            # tg = np.loadtxt(f)

        self.t = tg
            
        ntg = tg.size
        self.nt = ntg

        o = np.zeros(ntg, dtype = complex)
        
        nsamples = 0

        chunk_from = arange[0]
        chunk_to = arange[1]

        for i in range(chunk_from, chunk_to):
            p = os.path.join(os.getcwd(), folder,  name + "_" + str(i) + ".txt")
            p = pathlib.Path(p)

            if not p.exists():
                print("Warning: no chunk #", i)
                continue

            with p.open() as f:
                oc = np.loadtxt(f)    
                nsamples += oc[0, 0]
                o[0 : oc.shape[0] - 1] += oc[1:, 0] + 1j * oc[1:, 1] 

        if nsamples == 0:
            raise Exception('No simulation data for ' + name + ' in the range ' + str(chunk_from) + ' ... ' + str(chunk_to))
                
        o = o / nsamples
        
        self.o = o
        self.nsamples = nsamples
    
class Results(dict):
    pass
    
def compute_entropy(results):
    
    pa = results.jump_p
    sh = pa.shape
    nj = sh[0]
    ns = 1
    try:
        ns = sh[1]
    except:
        pa = tools_changed_by_me.as_column_vector(pa)
        #print('oops!')
        pass
    #print('pops!')
    S = np.zeros(nj)
    log_p = np.zeros(ns)
    for i in range(nj):
        log_p = log_p + np.log(pa[i, :])
        S[i] = sum(log_p) / ns
        
    setattr(results, 'jump_S', -S)
    
def compute_dm_entropy(results):
                       
    dm = results.jump_rho
    n = len(dm)
    r = dm[0]
    
    S = np.zeros(n)
    
    #print('n=', n)
    #print(S)
    
    for i in range(n):
        r = dm[i]
        probs, _ = LA.eigh(r)
        #probs = np.array(probs)
        S[i] = -sum(probs * np.log(probs))
        
    setattr(results, 'jump_rho_S', S)
    
class Model:
    def __init__(self):
        self.nquanta = None
        self.dimension = None
        self.dim_s = None
        self.zero_op = None
        self.O = None
        self.a = None
        self.a_dag = None
        self.id_e = None
        self.lo = None
        self.m = None
        self.m_max = None
        self.id_s = None
    
class problem:
    
    def __init__(self):
        self.impurity = None
        self.baths = []
        self.quench = None
        self.impurity_observables = []
        self.dim_s = None
        self.id_e = None
        self.id_s = None
        self.id = None
        self.model = None
        self.compute_probs = False
        self.compute_dmatrices = False
        
    def set_impurity(self, Hs):
        self.Hs = Hs
        Hs_ = Hs(0)
        dim_s = Hs_.shape[0]
        id_s = scipy.sparse.eye(dim_s).tocsc()
        self.dim_s = dim_s
        self.id_s = id_s
        
    def add_bath(self, bath, Vs):
        self.baths.append((bath, Vs))
        
    def add_impurity_observable(self, name, Os):
        self.impurity_observables.append((name, Os))
        
    def construct_model(self, max_num_quanta):
        
        self.model = Model()
        
        #-----------------------------------------------------
        
        Hs_ = self.Hs(0)
        dim_s = Hs_.shape[0]
        id_s = scipy.sparse.eye(dim_s).tocsc()
    
        m_max = self.baths[0][0].m_max
    
        nquanta = max_num_quanta
        self.model.nquanta = nquanta
        m = tools_changed_by_me.boson_chain(num_modes = m_max, max_num_quanta = nquanta, id_s = id_s)
        lo = m.get_local_observables()

        id_e = m.space.eye
     
        a_hat = [tools_changed_by_me.kron(id_s, a_) for a_ in m.a]
        a_hat_dag = [a_.T.conj() for a_ in a_hat]
    
        O = [tools_changed_by_me.kron(o_[1], id_e) for o_ in self.impurity_observables]
    
        #def Hse(ti):
        #    return tools.kron(Hs(ti * dt), id_e)
    
        #def V_dag(ti):
        #    return tools.kron(Vs(ti * dt), id_e).conj().T

        zero_op = tools_changed_by_me.kron(scipy.sparse.csc_array((dim_s, dim_s), dtype = complex), m.space.emptyH) 
    
        dimension = dim_s * m.dimension
        
        #-----------------------------------------------------
        
        self.model.dimension = dimension
        self.model.dim_s = dim_s
        self.model.zero_op = zero_op
        #self.model.V_dag = V_dag
        #self.model.Hs = Hse
        self.model.O = O
        self.model.a = a_hat
        self.model.a_dag = a_hat_dag
        self.model.id_e = id_e
        self.model.lo = lo
        self.model.m = m
        self.model.m_max = m_max
        self.model.id_s = id_s
        
    def set_quench(self, quench):
        self.quench = quench
        
    def solve(self, cfrom = 1, cto = 1, csize = 1, maxcores = 4, tmax = 100.0, fout = 'results'):
        
        o = job_opts()
        o.Hs = self.Hs
        o.Vs = self.baths[0][1]
        o.Os = self.impurity_observables
        o.bth = self.baths[0][0]
        o.csize = csize
        o.tmax = tmax
        o.fout = fout
        o.compute_probs = self.compute_probs
        o.model = self.model
        o.quench = self.quench
        o.compute_dmatrices = self.compute_dmatrices
        
        run(cfrom, cto, maxcores, o)
        
        results = Results()
        for o_ in self.impurity_observables:
            res = load_results(o_[0], [cfrom, cto + 1], fout)
            results[o_[0]] = res.o
            
            try:
                getattr(results, "t")
            except AttributeError:
                setattr(results, 't', res.t)
                
        if self.compute_probs:
            pa = load_probs([cfrom, cto + 1], fout).probabilities
            jt = o.bth.get_jump_times()
            setattr(results, 'jump_t', np.array(jt))
            setattr(results, 'jump_p', pa)
            compute_entropy(results)
            results.jump_t = results.jump_t[:len(results.jump_S)]
            
        if self.compute_dmatrices:
            dm = load_dmatrices([cfrom, cto + 1], fout).dmatrices
            setattr(results, 'jump_rho', dm)
            compute_dm_entropy(results)
            
        return results
    
