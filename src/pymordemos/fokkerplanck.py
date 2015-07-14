# -*- coding: utf-8 -*-

# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

'''Solves system derived from Fokker-Planck equation


'''

import sys
import csv
import time
from functools import partial
import numpy as np
import pymor.core as core
import pickle

from pymor.discretizers.advection import discretize_nonlinear_instationary_advection_fv_ndim
from pymor.domaindiscretizers import discretize_domain_default
from pymor.analyticalproblems.fokkerplanck import FPProblem
from pymor.grids import OnedGrid
from datetime import datetime as date



def fp_system(m, problem_name='SourceBeam', n_grid=500, num_flux='godunov_upwind',
              basis_type='Leg',basis_pl_discr=None,
              CFL_type='Auto',CFL=None,
              save_pickled=False, save_csv=False, save_time=False):


    assert num_flux in ('godunov_upwind')
    assert basis_type in ('Leg', 'RB')
    assert (basis_type == 'Leg' and basis_pl_discr == None) or (basis_type == 'RB' and basis_pl_discr is not None)
    assert CFL_type in ('Auto', 'Manual')
    assert (CFL_type == 'Auto' and CFL == None) or (CFL_type == 'Manual' and CFL is not None)

    print('Setup Problem ...')
    domain_discretizer = partial(discretize_domain_default, grid_type=OnedGrid)
    problem = FPProblem(sysdim=m, problem=problem_name,
                        basis_type=basis_type, basis_pl_discr=basis_pl_discr)


    print('Discretize ...')
    discretizer = discretize_nonlinear_instationary_advection_fv_ndim
    xwidth = problem.domain.domain[1] - problem.domain.domain[0]



    core.cache.clear_caches()
    mu = problem.basis_dict
    mu.update({'m': m})

    #sys.stdout.flush()
    tic = time.time()



    if CFL_type =='Auto':
        Lambda,W =np.linalg.eig(problem.flux_matrix)
        CFL=min(1./(2*np.max(np.abs(Lambda))),5.)

    print('Solve ...')
    print('CFL is {}'.format(CFL))
    while True:
        try:
            discretization, data = discretizer(problem, m, diameter=float(xwidth) / n_grid,
                                               num_flux=num_flux,
                                               CFL=CFL, domain_discretizer=domain_discretizer, num_values=1000)

            U, tvec = discretization.solve(mu)
            break
        except ValueError:
            if CFL_type == 'Manual':
                raise ValueError('Manual CFL set to {} is too large. Try again with smaller CFL.'.format(CFL))
            CFL *= 0.75
            print('CFL was automatically decreased. New CFL is {}'.format(CFL))

    V = U[0] * 0

    for j in range(m):
        V.axpy(mu['basis_werte'][0, j], U[j])

    print('Solving took {}s'.format(time.time() - tic))



    if save_csv == True:
        d=date.now()
        with open('{} {} {} m={}.csv'.format(problem_name,d.strftime("%y-%m-%d %H:%M:%S"),basis_type ,m),'w') as csvfile:
            writer=csv.writer(csvfile)
            for j in range(np.shape(V.data)[0]):
                writer.writerow(V.data[j,:])

    if save_time == True:
        d=date.now()
        with open('Time Steps {} {} {} m={}.csv'.format(problem_name,d.strftime("%y-%m-%d %H:%M:%S"),basis_type ,m),'w') as csvfile:
            writer=csv.writer(csvfile)
            writer.writerow(tvec)

    if save_pickled == True:
        d=date.now()
        pickle.dump((V,tvec) ,open( '{} {} {} m={}.p'.format(problem_name, d.strftime("%y-%m-%d %H:%M:%S"),basis_type ,m), "wb" ))


    return V, discretization





