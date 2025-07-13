import time
start_time = time.time()

#---------------------------------------------------------------------

import tools 
import numpy as np

#---------------------------------------------------------------------

import argparse
import pathlib
import math

parser = argparse.ArgumentParser()
parser.add_argument('--chain', type=pathlib.Path, default = "chain.txt")
parser.add_argument('--dt', type=float, default = 0.01)
parser.add_argument('--ns', type=int, default = -1)
parser.add_argument('--tmax', type=float, default = 400)
parser.add_argument('--rtol', type=float, default = 10**(-3))
parser.add_argument('--ringmax', type=int, default = 4)
args = parser.parse_args()

print("Wellcome to lightcone constructor version 01.03.2023-14.44")
print("Semiinfinite chain file: ", args.chain)

data = np.loadtxt(str(args.chain))

es = data[:, 0]
hs = data[:, 1]

    
if not es.size == hs.size:
    raise Exception("One should have nh = ne, where nh = number of hoppings, ne = number of on-site energies")

if args.ns < 0:
    ns = es.size
else:
    ns = args.ns
    es = es[:ns]
    hs = hs[:ns]
print('Length of chain: ', ns)

rel_tol = args.rtol
print('Relative significance treshold for forward lightcone boundary: ', rel_tol)

ring_max = args.ringmax
print('Maximal size of ring: ', ring_max)

t = args.tmax
print('Maximal time: ', t)

dt = args.dt
print('Time step: ', dt)

tg = np.arange(0, t + dt, dt)
ntg = tg.size

#-------------------------------------------------------------------------------------------------

# save time grid

import os
import sys

# Извлечение плашки без 'chain'
chain_stem = args.chain.stem  # Например, "chain_time=1000_h=0.1_c=0.025"
if chain_stem.startswith("chain_"):
    chain_stem = chain_stem[len("chain_"):]  # Удаляем префикс "chain_"

def get_output_path(name):
    output_dir = pathlib.Path(f"lightcone_{chain_stem}")
    output_dir.mkdir(exist_ok=True)
    return output_dir / name

# Формирование имени файла

time_filename = f"time.txt"
star_out_filename = f"star_out.txt"  
intervals_filename = f"intervals.txt" 
rotations_filename = f"rotations.txt"
couplings_filename = f"couplings.txt"


# Запись в файл
with open(get_output_path(time_filename), "w") as f:
    for i in range(ntg):
        print(str(tg[i]), file=f)

#-------------------------------------------------------------------------------------------------

scoupling = hs[0]
print('Coupling to impurity: ', scoupling)
H = tools.tridiag(es, hs[1:])

#--------------------------------------------------------------------------------------------------

H_dense = H.todense()
w, modes = tools.find_eigs_ascending(H_dense)

with open(get_output_path(star_out_filename), "w") as f:
    for i in range(ns):
        print(str(w[i]), "\t", modes[0, i].real * scoupling, "\t", modes[0, i].imag * scoupling, file = f)

#--------------------------------------------------------------------------------------------------

psi0 = np.zeros(ns, dtype = np.cdouble)
psi0[0] = 1

psi_lc = np.zeros((ns, ntg), dtype = np.cdouble)

def Ht(t):
    return H

for i, psi in tools.evolutionpy(start_index = 0, end_index = ntg, H = Ht, dt = dt, initial_state = psi0):
    psi_lc[:, i] = np.copy(psi)

#-------------------------------------------------------------------------------------------------

n_guard = 3
revival_tolerance = 10**(-9)
revival = np.amax(np.abs(psi_lc[-n_guard :, :]))

if revival > revival_tolerance:
    raise Exception('Revival: Increase the length of chain')

#-------------------------------------------------------------------------------------------------

rho_lc = np.zeros((ns, ns), dtype = np.cdouble)

for i in range(0, ntg):
    psi = tools.as_column_vector(psi_lc[:, i])
    rho_lc += tools.dyad(psi, psi) * dt

tools.make_hermitean(rho_lc)

#--------------------------------------------------------------------------------------------------


pi, U_rel = tools.find_largest_eigs(rho_lc)

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

U_min = np.eye(n_rel, dtype = np.cdouble)

n = n_rel

for i in reversed(range(0, ntg)):
    
    pi_min, _ = tools.find_smallest_eigs(rho_ret, 1)
    pi_max, _ = tools.find_largest_eigs(rho_ret, 1)

    lr_metric = pi_min - rel_tol * pi_max
    outside_lightcone = lr_metric < 0

    if outside_lightcone:
        pi, U = tools.find_eigs_descending(rho_ret)
        psi_lc_rel[: n, :] = U.T.conj() @ psi_lc_rel[: n, :]
        U_min[: n, :] = U.T.conj() @ U_min[: n, :]
        rho_ret = np.diag(pi[: -1].astype('cdouble'))
        times_in.insert(0, i + 1)
        n = n_rel - len(times_in)
        n_in.insert(0, n) 

    psi = tools.as_column_vector(psi_lc_rel[: n, i])
    rho_ret -= tools.dyad(psi, psi) * dt

    tools.make_hermitean(rho_ret)

#-------------------------------------------------------------------------------------------------

intervals_in = []

i_left = 0

for i_right, n in zip(times_in + [ntg], n_in):

    intervals_in.append((i_left, i_right, n))
    i_left = i_right

#--------------------------------------------------------------------------------------------------

rho_ret =  np.zeros((n_rel, n_rel), dtype = np.cdouble)

max_n_coupled = ring_max - 1

rho_adv = U_min @ np.copy(rho_lc_rel) @ U_min.T.conj()
tools.make_hermitean(rho_adv)

psi_lc_out = np.copy(psi_lc_rel)

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
        pi, U = tools.find_eigs_ascending(rho_cdi)
        psi_lc_out[n_out : n_in, i[0] :] = U.T.conj() @ psi_lc_out[n_out : n_in, i[0] :]

        rho_adv[n_out : n_in, n_out : ] = U.T.conj() @ rho_adv[n_out : n_in, n_out : ] 
        rho_adv[n_out : , n_out : n_in] = rho_adv[n_out : , n_out : n_in] @ U 

        n_out_new = n_in - max_n_coupled

        w = U.T.conj()

    intervals_out.append((begin, end, n_in, n_out, w))

    n_out = n_out_new

    for j in range(i[0], i[1]):

        psi = tools.as_column_vector(psi_lc_out[n_out : , j])
        rho_adv[n_out : , n_out : ] -= tools.dyad(psi, psi) * dt
    
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

for i in intervals_r:

    a = i[2]
    b = i[3]

    m_max = max(m_max, a - b)

#-------------------------------------------------------------------------

to_ring = [ _ % m_max + 1 for _ in range(0, n_rel)]

#---------------------------------------------------------------------


# Запись в файл

with open(get_output_path(intervals_filename), "w") as f:

    for i in intervals_r:

        b = i[2]
        a = i[3]

        print(i[0], "\t", i[1], "\t", a, "\t", b, file = f)

np.set_printoptions(linewidth=np.inf)

couplings = couplings * scoupling


with open(get_output_path(couplings_filename), "w") as f:
    for i in intervals_r:

        b = i[2]
        a = i[3]

        for ti in range(i[0], i[1]):

             print('\t'.join(map(str, couplings[a : b, ti].real)), file = f)
             print('\t'.join(map(str, couplings[a : b, ti].imag)), file = f) 

with open(get_output_path(rotations_filename), "w") as f:

    for i in intervals_r:

        b = i[2]
        a = i[3]

        w = i[4]

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

        

end_time = time.time()

#--------------------------------------------------------------------------------

print("The execution time :",
      (end_time-start_time), " sec")