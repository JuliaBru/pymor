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


def rb_solutions(problemname='SourceBeam', rb_size=50, return_rb=False, picklen=False,compute_rb=True):


    print('Setup Problem ...')


    problem=Fokkerplanck_V(problem=problemname, delta=0, quadrature_count=(1,1),P_parameter_range=(0.01,1.2),
                           dxP_parameter_range=(-5.4,0.9),dtP_parameter_range=(0,5))


    print('Discretize ...')

    n=250

    discretization, _ = discretize_elliptic_cg_plus(problem, diameter=1 / n)


    print('The parameter type is {}'.format(discretization.parameter_type))

    if picklen == False:
        snapshots=100

        np.random.seed()
        tic = time.time()

        d=date.now()
        for i in range(1):
            print(i)
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


            pickle.dump(V,open( "rb-daten {}, n={} Nr. {} {}.p".format(snapshots,n,i,d.strftime("%y-%m-%d %H:%M:%S")), "wb" ))


        print('Solving took {}s'.format(time.time() - tic))


        if compute_rb==True:
            rb,sw=pod(V,modes=50,orthonormalize=True,product=discretization.products['l2'],check_tol=0.1)
            discretization.visualize(rb)




    if picklen == True:
        rbvoll,sw =pickle.load(open("rb 1000000 15-02-02 10:47:28.p",'rb'))

        rb=NumpyVectorArray(rbvoll.data[0:rb_size,:])


    if return_rb==True:
        return  rb, discretization






if __name__ == '__main__':
    #args = docopt(__doc__)
    rb_solutions()