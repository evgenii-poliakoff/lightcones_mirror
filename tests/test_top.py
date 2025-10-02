import numpy as np
import math
import pytest
import lightcones.linalg as ll
import lightcones.space as sp
from lightcones.models import top

tol = 1e-10

def test_top():
    
    # case 1
    
    j = 11   
    t = top(j)
    
    # j_m
    j_m = t.j_m
    data = [ 4.69041575982+0.j,  6.48074069841+0.j,  7.74596669241+0.j, \
        8.71779788708+0.j,  9.48683298051+0.j, 10.09950493836+0.j, \
        10.58300524426+0.j, 10.9544511501 +0.j, 11.22497216032+0.j, \
        11.40175425099+0.j, 11.48912529308+0.j, 11.48912529308+0.j, \
        11.40175425099+0.j, 11.22497216032+0.j, 10.9544511501 +0.j, \
        10.58300524426+0.j, 10.09950493836+0.j,  9.48683298051+0.j, \
        8.71779788708+0.j,  7.74596669241+0.j,  6.48074069841+0.j, \
        4.69041575982+0.j ]
    assert np.allclose(j_m.data, data, atol=tol), \
        f"data for j_m"
    indices = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, \
       17, 18, 19, 20, 21]
    assert np.array_equal(j_m.indices, indices), \
        f"indices for j_m"
    indptr = [ 0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, \
       16, 17, 18, 19, 20, 21, 22]
    assert np.array_equal(j_m.indptr, indptr), \
        f"indptr for j_m"
        
    # j_p
    j_p = t.j_p
    data = [ 4.69041575982+0.j,  6.48074069841+0.j,  7.74596669241+0.j,
        8.71779788708+0.j,  9.48683298051+0.j, 10.09950493836+0.j,
       10.58300524426+0.j, 10.9544511501 +0.j, 11.22497216032+0.j,
       11.40175425099+0.j, 11.48912529308+0.j, 11.48912529308+0.j,
       11.40175425099+0.j, 11.22497216032+0.j, 10.9544511501 +0.j,
       10.58300524426+0.j, 10.09950493836+0.j,  9.48683298051+0.j,
        8.71779788708+0.j,  7.74596669241+0.j,  6.48074069841+0.j,
        4.69041575982+0.j ]
    assert np.allclose(j_p.data, data, atol=tol), \
        f"data for j_p"
    indices = [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, \
       18, 19, 20, 21, 22]
    assert np.array_equal(j_p.indices, indices), \
        f"indices for j_p"
    indptr = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 22]
    assert np.array_equal(j_p.indptr, indptr), \
        f"indptr for j_p"
        
    # j_x
    j_x = t.j_x
    data = [2.34520787991+0.j, 2.34520787991+0.j, 3.2403703492 +0.j,
       3.2403703492 +0.j, 3.87298334621+0.j, 3.87298334621+0.j,
       4.35889894354+0.j, 4.35889894354+0.j, 4.74341649025+0.j,
       4.74341649025+0.j, 5.04975246918+0.j, 5.04975246918+0.j,
       5.29150262213+0.j, 5.29150262213+0.j, 5.47722557505+0.j,
       5.47722557505+0.j, 5.61248608016+0.j, 5.61248608016+0.j,
       5.7008771255 +0.j, 5.7008771255 +0.j, 5.74456264654+0.j,
       5.74456264654+0.j, 5.74456264654+0.j, 5.74456264654+0.j,
       5.7008771255 +0.j, 5.7008771255 +0.j, 5.61248608016+0.j,
       5.61248608016+0.j, 5.47722557505+0.j, 5.47722557505+0.j,
       5.29150262213+0.j, 5.29150262213+0.j, 5.04975246918+0.j,
       5.04975246918+0.j, 4.74341649025+0.j, 4.74341649025+0.j,
       4.35889894354+0.j, 4.35889894354+0.j, 3.87298334621+0.j,
       3.87298334621+0.j, 3.2403703492 +0.j, 3.2403703492 +0.j,
       2.34520787991+0.j, 2.34520787991+0.j]
    assert np.allclose(j_x.data, data, atol=tol), \
        f"data for j_x"
    indices = [ 1,  0,  2,  1,  3,  2,  4,  3,  5,  4,  6,  5,  7,  6,  8,  7,  9, \
        8, 10,  9, 11, 10, 12, 11, 13, 12, 14, 13, 15, 14, 16, 15, 17, 16, \
       18, 17, 19, 18, 20, 19, 21, 20, 22, 21]
    assert np.array_equal(j_x.indices, indices), \
        f"indices for j_x"
    indptr = [ 0,  1,  3,  5,  7,  9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, \
       33, 35, 37, 39, 41, 43, 44]
    assert np.array_equal(j_x.indptr, indptr), \
        f"indptr for j_x"
        
    # j_y
    j_y = t.j_y
    data = [0.-2.34520787991j, 0.+2.34520787991j, 0.-3.2403703492j ,
       0.+3.2403703492j , 0.-3.87298334621j, 0.+3.87298334621j,
       0.-4.35889894354j, 0.+4.35889894354j, 0.-4.74341649025j,
       0.+4.74341649025j, 0.-5.04975246918j, 0.+5.04975246918j,
       0.-5.29150262213j, 0.+5.29150262213j, 0.-5.47722557505j,
       0.+5.47722557505j, 0.-5.61248608016j, 0.+5.61248608016j,
       0.-5.7008771255j , 0.+5.7008771255j , 0.-5.74456264654j,
       0.+5.74456264654j, 0.-5.74456264654j, 0.+5.74456264654j,
       0.-5.7008771255j , 0.+5.7008771255j , 0.-5.61248608016j,
       0.+5.61248608016j, 0.-5.47722557505j, 0.+5.47722557505j,
       0.-5.29150262213j, 0.+5.29150262213j, 0.-5.04975246918j,
       0.+5.04975246918j, 0.-4.74341649025j, 0.+4.74341649025j,
       0.-4.35889894354j, 0.+4.35889894354j, 0.-3.87298334621j,
       0.+3.87298334621j, 0.-3.2403703492j , 0.+3.2403703492j ,
       0.-2.34520787991j, 0.+2.34520787991j]
    assert np.allclose(j_y.data, data, atol=tol), \
        f"data for j_y"
    indices = [ 1,  0,  2,  1,  3,  2,  4,  3,  5,  4,  6,  5,  7,  6,  8,  7,  9, \
        8, 10,  9, 11, 10, 12, 11, 13, 12, 14, 13, 15, 14, 16, 15, 17, 16, \
       18, 17, 19, 18, 20, 19, 21, 20, 22, 21]
    assert np.array_equal(j_y.indices, indices), \
        f"indices for j_y"
    indptr = [ 0,  1,  3,  5,  7,  9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, \
       33, 35, 37, 39, 41, 43, 44]
    assert np.array_equal(j_y.indptr, indptr), \
        f"indptr for j_y"
        
    # j_z
    j_z = t.j_z
    data = [-11.+0.j, -10.+0.j,  -9.+0.j,  -8.+0.j,  -7.+0.j,  -6.+0.j,
        -5.+0.j,  -4.+0.j,  -3.+0.j,  -2.+0.j,  -1.+0.j,   1.+0.j,
         2.+0.j,   3.+0.j,   4.+0.j,   5.+0.j,   6.+0.j,   7.+0.j,
         8.+0.j,   9.+0.j,  10.+0.j,  11.+0.j]
    assert np.allclose(j_z.data, data, atol=tol), \
        f"data for j_z"
    indices = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 12, 13, 14, 15, 16, 17,
       18, 19, 20, 21, 22]
    assert np.array_equal(j_z.indices, indices), \
        f"indices for j_z"
    indptr = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 11, 12, 13, 14, 15,
       16, 17, 18, 19, 20, 21, 22]
    assert np.array_equal(j_z.indptr, indptr), \
        f"indptr for j_z"
        
def test_top_vac():
        
    t = top(11)
    
    vac_actual = t.vac()
    
    vac_expected = np.zeros(23, dtype = complex)
    vac_expected[0] = 1.0
    
    assert np.allclose(vac_actual, vac_expected, atol=tol), \
        f"vac does not match"
        
def test_top_eye():
    
    t = top(11)
    
    # eye
    eye = t.eye()
    data = [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1.]
    assert np.allclose(eye.data, data, atol=tol), \
        f"data for eye"
    indices = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22]
    assert np.array_equal(eye.indices, indices), \
        f"indices for eye"
    indptr = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23]
    assert np.array_equal(eye.indptr, indptr), \
        f"indptr for eye"
        

def test_top_zero():
    
    t = top(11)
    
    # zero
    z = t.zero()
    data = []
    assert np.allclose(z.data, data, atol=tol), \
        f"data for zero"
    indices = []
    assert np.array_equal(z.indices, indices), \
        f"indices for zero"
    indptr =[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0]
    assert np.array_equal(z.indptr, indptr), \
        f"indptr for zero"
        
def test_top_state_with():
    
     t = top(11)
     state = t.state_with(j_z=5)
     j_z = np.vdot(state, t.j_z @ state).real
     assert abs(j_z - 5) < tol

def test_top_state_with_half_spin():

    t = top(0.5)
    # should return vacuum state
    state = t.state_with(j_z=-0.5)
    vac = t.vac()
    infidelity = 1.0 - np.vdot(state, vac)
    assert abs(infidelity) < tol
    # should return excited state
    state = t.state_with(j_z=0.5)
    vac = t.j_p @ t.vac()
    infidelity = 1.0 - np.vdot(state, vac)
    assert abs(infidelity) < tol

    #
    j = 5.5 
    t = top(j)
    # should return vacuum state
    state = t.state_with(j_z=-j)
    vac = t.vac()
    infidelity = 1.0 - np.vdot(state, vac)
    assert abs(infidelity) < tol
    # should return excited state
    state = t.state_with(j_z=-j + 3)
    state_expected = t.vac()
    for m in [-j, -j + 1, -j + 2]:
        state_expected = t.j_p @ state_expected / math.sqrt((j - m) * (j + m + 1))
    infidelity = 1.0 - np.vdot(state, state_expected)
    assert abs(infidelity) < tol
