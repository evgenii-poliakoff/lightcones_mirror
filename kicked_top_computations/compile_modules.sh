#!/usr/bin/env bash
source ~/.bash_profile
export FFLAGS="-ffree-line-length-512"
python -m numpy.f2py  -c fastmul.f90 -m fastmul 
python -m numpy.f2py  -c fastmuli.f90 -m fastmuli 
python -m numpy.f2py  -c evolution_chained2.f90 -m evolution_chained2  
python -m numpy.f2py  -c evolution_chained2_kicked.f90 -m evolution_chained2_kicked 
python -m numpy.f2py  -c evolution.f90 -m evolution 
python -m numpy.f2py  -c evolution2.f90 -m evolution2  
python -m numpy.f2py  -c local_op.f90 -m local_op
python -m numpy.f2py  -c solve.f90 -m solve
python -m numpy.f2py  -c outer.f90 -m _outer
python -m numpy.f2py  -c dlancz.f -m _dlancz