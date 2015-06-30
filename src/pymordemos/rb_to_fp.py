#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

'''Proof of concept for solving the poisson equation in 1D using linear finite elements and our grid interface

Usage:
    rb_to_fp.py


Options:

    -h, --help    this message
'''

from __future__ import absolute_import, division, print_function

import numpy as np
from pymor.core import getLogger
from pymor.discretizers.ellipticplus import discretize_elliptic_cg_plus
from pymor.analyticalproblems.fokkerplanck_rb import Fokkerplanck_V
import pickle
from pymor.la import NumpyVectorArray
from pymor.la.pod import pod
from datetime import datetime as date
import time



getLogger('pymor.discretizations').setLevel('INFO')


def pod_from_snapshots(snapshots, n_grid=500, rb_size=25, problemname='SourceBeam',save_snapshots=False, compute_rb=True):
    assert problemname == 'SourceBeam'


    #print('Setup Problem ...')


    problem=Fokkerplanck_V(problem=problemname, delta=0, quadrature_count=(1,1),P_parameter_range=(0.01,1.2),
                           dxP_parameter_range=(-5.4,0.9),dtP_parameter_range=(0,5))

    #print('Discretize ...')


    discretization, _ = discretize_elliptic_cg_plus(problem, diameter=2 / n_grid)


    np.random.seed()
    tic = time.time()

    d=date.now()
    V = discretization.type_solution.empty(discretization.dim_solution, reserve=snapshots)
    j=0
    for mu in problem.parameter_space.sample_randomly(snapshots):
        j+=1


        if (problemname == 'SourceBeam' and mu['qxpoint'] >=1) or not (problemname == 'SourceBeam'):
            try:
                V.append(discretization.solve(mu))
                print('Nr. {}'.format(j))
            except:
                print('Fehler bei mu={}'.format(mu))
                V.append(NumpyVectorArray(np.zeros(discretization.dim_solution)))
        else:
            V.append(NumpyVectorArray(np.zeros(discretization.dim_solution)))




    if save_snapshots:
        pickle.dump(V,open( "rb-daten {}, n={} {}.p".format(snapshots,n_grid,d.strftime("%y-%m-%d %H:%M:%S")), "wb" ))



    rb,sw=pod(V,modes=rb_size,orthonormalize=True,product=discretization.products['l2'],check_tol=0.1)
    #discretization.visualize(rb)
    pickle.dump(V,open( "rb {}, n={} {}.p".format(snapshots,n_grid,d.strftime("%y-%m-%d %H:%M:%S")), "wb" ))


    return rb,sw,discretization






if __name__ == '__main__':
    #args = docopt(__doc__)
    pod_from_snapshots(20,n_grid=200,rb_size=15)