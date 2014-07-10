__author__ = 'j_brun16'


import numpy as np
#from pymor.algorithms.timestepping import explicit_euler

from pymor.algorithms.timestepping import ExplicitEulerTimeStepper

print np.array([2,2])

A=solve(ExplicitEulerTimeStepper,0.,2.,np.array([2.,2.]), )