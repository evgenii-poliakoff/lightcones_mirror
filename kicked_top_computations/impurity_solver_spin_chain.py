import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import sys
import numpy as np
import tools
from tools import mv
import time
from evolution_chained2_kicked import evolution_chained2_kicked
from evolution import evolution
from evolution2 import evolution2

import math
import random
import pathlib
from scipy.linalg import eigh, eigvalsh
from scipy import sparse
from scipy.sparse import coo_matrix
from scipy.sparse import identity
from scipy.sparse import diags



print("Wellcome to lightcone impurity solver version 10.03.2023-11.51")
print("Will read the lightcone info from the current directory")

#### execution parameters

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--fout', type=str, default = "results")
parser.add_argument('--cfrom', type=int, default = 1)
parser.add_argument('--cto', type=int, default = 10)
parser.add_argument('--csize', type=int, default = 10)
parser.add_argument('--maxcores', type=int, default = 1)
parser.add_argument('--nquanta', type=int, default = 3)
parser.add_argument('--tmax', type=float, default = 100)
parser.add_argument('--nspins', type=float, default = 10)
parser.add_argument('--lambd', type=float, default = 0)
parser.add_argument('--beta', type=str, default = "inf")
# parser.add_argument('--psi_file', type=str, default="psi_file.txt", help='Path to initial psi file')
parser.add_argument('--intervals', type=str, default="intervals.txt", help='Path to intervals')
parser.add_argument('--couplings', type=str, default="couplings.txt", help='Path to couplings')
parser.add_argument('--rotations', type=str, default="rotations.txt", help='Path to rotations')
parser.add_argument('--times', type=str, default="times.txt", help='Path to times')
parser.add_argument('--star_out', type=str, default="star_out.txt", help='Path to star_out')

args = parser.parse_args()

chunk_from = args.cfrom
chunk_to = args.cto # inclusive
trajectories_per_chunk = args.csize
max_cores_used = args.maxcores

print("Will compute ", (chunk_to - chunk_from + 1) * trajectories_per_chunk, " trajectories")
print("on max ", max_cores_used, " CPU cores")

#### where to save the results

folder = args.fout

print("Will save results into ", folder)

#### Max simulation time and time step

tmax = args.tmax

print("Will simulate up to time", tmax)

####

if args.beta.strip() == "inf":
    beta = -1
    print("Environment at zero themperature")
else:
    beta = float(args.beta)
    print("Environment beta = ", beta)

lambda_ = args.lambd

print("Parameter: ", lambda_)

n_max = args.nquanta

print("Max number of coupled quanta: ", n_max)

n_spin = int(args.nspins)

print("n spins: ", n_spin)

#----------------------------------------------------

intervals = []

try:
    with pathlib.Path(args.intervals).open() as f:
        while True:
            l = [int(e) for e in next(f).split()] 
            intervals.append(l)
except StopIteration as e:
    pass

#----------------------------------------------------

couplings = []

try:
    with pathlib.Path(args.couplings).open() as f:
        while True:
            re = np.asarray([float(e) for e in next(f).split()])
            im = np.asarray([float(e) for e in next(f).split()])
            couplings.append(re + 1j * im)
except StopIteration as e:
    pass

#----------------------------------------------------

rotations = []

m_max = 0
n_rel = 0

with pathlib.Path(args.rotations).open() as f:
    
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

print("Max number of coupled modes: ", m_max)

#----------------------------------------------------

with pathlib.Path(args.times).open() as f:
    tg = np.loadtxt(f)
ntg = tg.size
dt = tg[1] - tg[0]

#-----------------------------------------------------

t = tg
nt = ntg
nt_ = np.where(t <= tmax)[0][-1]
t_ = t[0 : nt_]

#-----------------------------------------------------

p = os.path.join(sys.path[0], folder, "time.txt")
os.makedirs(os.path.dirname(p), exist_ok = True)

with open(p, "w") as f:
    for i in range(nt_):
        print(str(t[i]), file = f)

#-----------------------------------------------------

print('computing sparse matrices...', flush = True)


m = tools.spin_chain_boson_model(n_spin, m_max, n_max)
lo = m.get_local_observables()

m_spin = tools.spin_chain(n_spin)
#-----------------------------------------------------

print('...done', flush = True)

### Initial condition one spin up and vacuum

# psi_ini = np.zeros(m.dimension, dtype = complex)
# psi_ini[0] = 1

to_ring = [ _ % m_max for _ in range(0, n_rel)]

n_spin_ = n_spin - 1
J = 1
r = 0.5
h = J/lambda_

H_spin_ = coo_matrix((m.dimension , m.dimension ), dtype = complex).tocsc()
for i in range(n_spin - 1):
    H_spin_ -= J * (
        (1 + r)/2 * m.sx[i] @ m.sx[i+1] +
        (1 - r)/2 * m.sy[i] @ m.sy[i+1])
# periodic boundary condition    
H_spin_ -= J * (
    (1 + r)/2 * m.sx[n_spin - 1] @ m.sx[0] +
    (1 - r)/2 * m.sy[n_spin - 1] @ m.sy[0])
for i in range(n_spin):
    H_spin_ -= h * m.sz[i]


H_spi = np.zeros((2**n_spin, 2**n_spin), dtype='complex')
for i in range(n_spin - 1):
    H_spi -= J * ((1 + r)/2 * m_spin.sx[i] @ m_spin.sx[i+1] +
                (1 - r)/2 * m_spin.sy[i] @ m_spin.sy[i+1])
H_spi -= J * ((1 + r)/2 * m_spin.sx[n_spin - 1] @ m_spin.sx[0] +
                (1 - r)/2 * m_spin.sy[n_spin - 1] @ m_spin.sy[0])
for i in range(n_spin):
    H_spi -= h * m_spin.sz[i]

eigenvalues, eigenvectors = eigh(H_spi)

psi_boson_vac = np.zeros(m.fs_chain.dimension)
psi_boson_vac[0] = 1.0  # vacuum bosons

psi_ini = np.kron(eigenvectors[:,0], psi_boson_vac)

# psi_spin = np.zeros(m.space.dimension) 
# psi_spin[3] = 1.0
# psi_ini = np.kron(psi_spin, psi_boson_vac)

dim_A = m.space.dimension
dim_B = m.fs_chain.dimension

def H_spin(ti):
    return H_spin_

# def H_spin(ti):
#     return np.kron(H_spi, np.eye(252))


la = lo.a()
la_dag = lo.a_dag()

####

def job(image):
    
    print('running chunk ', image, ', of ', trajectories_per_chunk, ' trajectories', flush = True)
    
    start_time = time.time()
    
    seed = 1000 * math.pi * image + pow(image, 2) 
    random.seed(seed)
    
    nqx = np.zeros(nt_)
    nqy = np.zeros(nt_)
    nqz = np.zeros(nt_)
    nqE = np.zeros(nt_)

    j_probs = []
    j_disps = []
    j_inds = []
    
    j_av_disp = []

    rho_part = []
    
    psi = np.zeros(m.dimension, dtype = complex)
    psi_mid = np.zeros(m.dimension, dtype = complex)
    psi_mid_next = np.zeros(m.dimension, dtype = complex)
    psi_buff = np.zeros(m.dimension, dtype = complex)
    
    job.eta_th = None
    
    
    ####

    def eval_o(ti, psi):
        
        nqx[ti] = 0
        nqy[ti] = 0
        nqz[ti] = 0
        mv(sum(m.sx[i] for i in range(n_spin))/n_spin, psi, psi_buff)
        nqx[ti] = nqx[ti] + np.vdot(psi, psi_buff).real
            
        mv(sum(m.sy[i] for i in range(n_spin))/n_spin, psi, psi_buff)
        nqy[ti] = nqy[ti] + np.vdot(psi, psi_buff).real
        
        mv(sum(m.sz[i] for i in range(n_spin))/n_spin, psi, psi_buff)
        nqz[ti] = nqz[ti] + np.vdot(psi, psi_buff).real
        
    
    def eval():
        
        
        psi_begin = np.copy(psi_ini)

        first_in_chain = 1

        ni = 0

        for (i, i_) in zip(intervals, [None] + intervals[:-1]):

            if i[0] >= nt_:
                return
            
            i1 = min(i[1], nt_-1)

            b = i[3]
            a = i[2]

            if not i_ is None:
                a_ = i_[2]
                for q in range(a_, a):
                    
                    measured_mode = to_ring[q] #+ 1

                    psi_begin, j_prob, j_psi, j_ind, _  = lo.quantum_jumpEx(psi_begin, measured_mode)
                    
                    # psi_begin_ = psi_begin.reshape((dim_A, dim_B))
                    rho_full = np.outer(psi_begin, np.conj(psi_begin))
                    rho_full = rho_full.reshape((dim_A, dim_B, dim_A, dim_B))
                    rho_sys = np.einsum('iaja->ij', rho_full)
                    rho_sys_flat = rho_sys.flatten()
                    rho_part.append(rho_sys_flat)
                    
                                    
                    j_inds.append(j_ind)
                    j_probs.append(j_prob)
                    
                    dd = np.zeros(j_psi.shape[1], dtype = complex)
                    
                    for r in range(j_psi.shape[1]):
                        dd[r] = np.vdot(j_psi[:, r], la @ j_psi[:, r])
                        
                    j_disps.append(dd)
                    
                    ad = 0
                    
                    for r in range(j_psi.shape[1]):
                        ad = ad + j_prob[r] * np.vdot(j_psi[:, r], la @ j_psi[:, r])
                    
                    j_av_disp.append(ad)
                    
                        
            w = rotations[ni]

            #try:

            a_ring = [m.a[to_ring[_]] for _ in range(a, b)]
            a_ring_dag = [m.a_dag[to_ring[_]] for _ in range(a, b)]

            Hw = m.hs_joint.emptyH
            if not w is None:
                for p in range(b - a):
                    for q in range(b - a):
                        Hw += 1j * a_ring_dag[q] @ a_ring[p] * w[q, p].conj()


            def Vint(ti):
                
                V = 0*sum(couplings[ti] * a_ring) @ sum(m.sx[i] for i in range(n_spin))/n_spin
                    
                V = V + V.conj().transpose()
                return(V)
            
            def Ht(ti):
                return H_spin(ti)  #+ Hw #Vint(ti) + Hw


            eval.Ht_ = None

            def begin_step(ti, psi):
                eval.Ht_ = Ht(ti)
                
             
            def apply_H(ti, psi_in, psi_out):
                mv(eval.Ht_, psi_in, psi_out)

            evolution_chained2_kicked(i[0], i1, dt, begin_step, apply_H, eval_o, psi_begin, psi, psi_mid, psi_mid_next, first_in_chain)
            first_in_chain = 0
            
            psi_begin = np.copy(psi)


            ni = ni + 1
# store results 
    
    for i in range(trajectories_per_chunk):
        eval()

    end_time = time.time()
    
    print('chunk ', image, ' execution time: ', end_time-start_time, ' sec', flush = True)

    ####
    
    path = os.path.join(sys.path[0], folder, "x_" + str(image) + ".txt")
    
    with open(path, "w") as f:
        print(str(trajectories_per_chunk) + " " + str(0), file = f)
        for i in range(nt_):
            print(str(nqx[i].real) + " " + str(nqx[i].imag), file = f)
   
    path = os.path.join(sys.path[0], folder, "y_" + str(image) + ".txt")
    
    with open(path, "w") as f:
        print(str(trajectories_per_chunk) + " " + str(0), file = f)
        for i in range(nt_):
            print(str(nqy[i].real) + " " + str(nqy[i].imag), file = f)
    

    path = os.path.join(sys.path[0], folder, "z_" + str(image) + ".txt")
    
    with open(path, "w") as f:
        print(str(trajectories_per_chunk) + " " + str(0), file = f)
        for i in range(nt_):
            print(str(nqz[i].real) + " " + str(nqz[i].imag), file = f)
    
    path = os.path.join(sys.path[0], folder, "E_" + str(image) + ".txt")
    
    with open(path, "w") as f:
        print(str(trajectories_per_chunk) + " " + str(0), file = f)
        for i in range(nt_):
            print(str(nqE[i].real) + " " + str(nqE[i].imag), file = f)
    
    path = os.path.join(sys.path[0], folder, "probs_" + str(image) + ".txt")
    
    with open(path, "w") as f:
        np.savetxt(f, j_probs)

    path = os.path.join(sys.path[0], folder, "rhos_" + str(image) + ".txt")
    
    with open(path, "w") as f:
        np.savetxt(f, rho_part)
        
    path = os.path.join(sys.path[0], folder, "inds_" + str(image) + ".txt")
    
    with open(path, "w") as f:
        np.savetxt(f, j_inds)
        
    path = os.path.join(sys.path[0], folder, "disps_re_" + str(image) + ".txt")
    
    with open(path, "w") as f:
        np.savetxt(f, np.array(j_disps).real)
        
    path = os.path.join(sys.path[0], folder, "disps_im_" + str(image) + ".txt")
    
    with open(path, "w") as f:
        np.savetxt(f, np.array(j_disps).imag)
        
    path = os.path.join(sys.path[0], folder, "av_disps_re_" + str(image) + ".txt")
    
    with open(path, "w") as f:
        np.savetxt(f, np.array(j_av_disp).real)
        
    path = os.path.join(sys.path[0], folder, "av_disps_im_" + str(image) + ".txt")
    
    with open(path, "w") as f:
        np.savetxt(f, np.array(j_av_disp).imag)
        

def job_(image):
    try:
        job(image)
    except:
        import traceback
        traceback.print_exception(*sys.exc_info())
        raise()


def progress_indicator(f):
    global progress
    progress += 1
    print('finished ', progress, 'job  out of ', chunk_to - chunk_from + 1, flush = True)

if __name__ == '__main__':
    
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures import wait
        
    with ProcessPoolExecutor(max_workers = max_cores_used) as executor:
        
        print('submitting jobs to pool ...', flush = True)
        
        futures = [executor.submit(job_, i) for i in range(chunk_from, chunk_to + 1)]
        global progress
        progress = 0
        for f in futures:
            f.add_done_callback(progress_indicator)
            
        wait(futures)
        
        for f in futures:
            exception = f.exception()
            if not exception is None:
                import traceback
                print("Exception:")
                print(exception)
                traceback.print_exc()
            
    print('\nDone!')
