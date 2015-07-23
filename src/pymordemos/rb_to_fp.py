#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

'''
Collection of different functions needed for FP-RB basis generation

'''

from __future__ import absolute_import, division, print_function

import numpy as np
import math
import pickle
import csv
from datetime import datetime as date
from pymor.core import getLogger
from pymor.discretizers.ellipticplus import discretize_elliptic_cg_plus
from pymor.analyticalproblems.fokkerplanck_rb import Fokkerplanck_V
from pymordemos.fokkerplanck import fp_system
from pymor.functions import GenericFunction
from pymor.la.gram_schmidt import gram_schmidt
from pymor.la import NumpyVectorArray
from pymor.la.pod import pod
from pymor.parameters import CubicParameterSpace

getLogger('pymor.discretizations').setLevel('INFO')


def fperror(V, FPLoes):
    # Computes error between reduced solution (given as NumpyVectorArray) and imported reference solution (given as np.array)
    Vd = V.data
    (nt, nx) = Vd.shape
    (ntf, nxf) = FPLoes.shape
    Vneu = np.zeros((ntf, nxf))
    for i in range(nxf):
        for j in range(ntf):
            Vneu[j, i] = Vd[math.floor(j*nt/ntf), math.floor(i*nx/nxf)]
    error = np.sum(np.abs(FPLoes - Vneu))/np.sum(np.abs(FPLoes))
    return error


def fp_random_snapshots(n_train, n_grid=200, test_case='SourceBeam', save_snapshots=False, visualize=False):
    assert test_case == 'SourceBeam'
    logger = getLogger('pymordemos.rb_to_fp.fp_random_snapshots')

    problem = Fokkerplanck_V(test_case=test_case, quadrature_count=(1, 1), P_parameter_range=(0.01, 1.2),
                             dxP_parameter_range=(-5.4, 0.9), dtP_parameter_range=(0, 5))

    discretization, _ = discretize_elliptic_cg_plus(problem, diameter=2. / n_grid)

    np.random.seed()

    d = date.now()
    V = discretization.type_solution.empty(discretization.dim_solution, reserve=n_train)
    j = 0

    logger.info('Solve transverse PDE for {} random parameter values'.format(n_train))

    for mu in problem.parameter_space.sample_randomly(n_train):
        j += 1
        if (test_case == 'SourceBeam' and mu['qxpoint'] >= 1) or not (test_case == 'SourceBeam'):
            try:
                V.append(discretization.solve(mu))
                logger.info('No. {}'.format(j))
            except:
                logger.info('Error at computing snapshot for mu={}'.format(mu))
                V.append(NumpyVectorArray(np.zeros(discretization.dim_solution)))
        else:
            V.append(NumpyVectorArray(np.zeros(discretization.dim_solution)))

    if save_snapshots:
        pickle.dump((V, discretization), open("Snapshots {}, n={} {}.p".format(n_train, n_grid, d.strftime("%y-%m-%d %H:%M:%S")), "wb"))

    if visualize:
        discretization.visualize(V)

    return V, discretization


def pod_from_snapshots(snapshots, basis_size, discretization=None):

    if discretization == None:
        (n_train, n_grid) = snapshots.data.shape
        problem = Fokkerplanck_V(test_case='SourceBeam', quadrature_count=(1, 1), P_parameter_range=(0.01, 1.2),
                           dxP_parameter_range=(-5.4, 0.9), dtP_parameter_range=(0, 5))
        discretization, _ = discretize_elliptic_cg_plus(problem, diameter=2. / (n_grid-1))

    basis = pod(snapshots, modes=basis_size, product=discretization.products['l2'], orthonormalize=True)

    return basis, discretization


def basis_plus_boundary(basis, boundary_type, v_discr):

    if boundary_type == 'p=0.1':
        def delta(x):
            return np.exp(-(x-1)**2/0.1)
    if boundary_type == 'p=0.01':
        def delta(x):
            return np.exp(-(x-1)**2/0.01)
    if boundary_type == 'p=0.001':
        def delta(x):
            return np.exp(-(x-1)**2/0.001)
    if boundary_type == 'p=0.0001':
        def delta(x):
            return np.exp(-(x-1)**2/0.0001)
    if boundary_type == 'peak':
        def delta(x):
            return x == 1

    Delta = GenericFunction(delta, shape_range=(1,))
    dirich = Delta.evaluate(v_discr.visualizer.grid.centers(1))[:, 0]
    dirich = dirich/(np.sqrt(v_discr.products['l2'].apply2(NumpyVectorArray(dirich), NumpyVectorArray(dirich), pairwise=False)))

    bound_pl_basis = NumpyVectorArray(dirich)
    bound_pl_basis.append(basis)
    bound_pl_basis = gram_schmidt(bound_pl_basis, product=v_discr.products['l2'], offset=0, check=True)
    bound_pl_basis = NumpyVectorArray(bound_pl_basis.data[0:-1, :])

    return bound_pl_basis, v_discr


def proximity_func(mu1, mu2):
    # Proximity function used in greedy algorithms
    d = (1./1.19*(mu1['P'][0, 0] - mu2['P'][0, 0]))**2
    d += (1./6.3*(mu1['dxP'][0, 0] - mu2['dxP'][0, 0]))**2
    d += (1./5.*(mu1['dtP'][0, 0] - mu2['dtP'][0, 0]))**2
    d += 0*(1./4.*(mu1['qtpoint'] - mu2['qtpoint']))**2
    d += (1./2.*(mu1['qxpoint'] - mu2['qxpoint']))**2
    d += (mu1['dirich'][0] - mu2['dirich'][0])**2
    d += (mu1['dirich'][1] - mu2['dirich'][1])**2
    return np.sqrt(d)


def greedy_fp(m_max, i_refine, sample, test_grid, seed=None, start_basis=None):

    logger = getLogger('pymordemos.rb_to_fp.greedy_fp')

    StartB = dict.fromkeys(range(m_max + 1))
    StartB[0] = NumpyVectorArray.empty(501)
    np.random.seed(seed)
    mstart = 0

    if start_basis is not None:
        mstart = start_basis._len
        StartB[mstart] = start_basis

    problem = Fokkerplanck_V(test_case='SourceBeam', quadrature_count=(1, 1), P_parameter_range=(0.01, 1.2),
                           dxP_parameter_range=(-5.4, 0.9), dtP_parameter_range=(0, 5))

    n = 250
    discretization, _ = discretize_elliptic_cg_plus(problem, diameter=1 / n)

    FDRef = np.zeros((1000, 500))

    with open('FD_reference_solution.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        i = 0
        for row in reader:
            for j in range(500):
                FDRef[i, j] = float(row[j])
            i += 1

    for m_ind in range(mstart, m_max):
        m = m_ind+1
        B = dict.fromkeys(range(i_refine*sample))
        Basis = dict.fromkeys(range(i_refine*sample))
        mudict = dict.fromkeys(range(i_refine*sample))
        V = dict.fromkeys(range(i_refine*sample))
        relerror = np.ones(i_refine*sample)*10000.

        sample_ind = 0
        logger.info('m_ind={}'.format(m_ind))

        snapshot_min_ind = np.ma.argmin(relerror)

        for i_ind in range(i_refine):

            logger.info('i_ind={}'.format(i_ind))
            logger.info('Sample random parameter values')

            if i_ind == 0:
                # In first iteration random parameter values from the whole domain
                for mu in problem.parameter_space.sample_randomly(sample):
                    mudict[sample_ind] = mu
                    sample_ind += 1
            else:
                # Search for parameter values near the best value from the first iterations
                mudiff = np.zeros(i_ind*sample)
                for sample_ind in range(i_ind*sample, (i_ind+1)*sample):

                    while True:
                        for mu in problem.parameter_space.sample_randomly(1):
                            for test_ind in range(i_ind*sample):
                                mudiff[test_ind] = proximity_func(mudict[test_ind], mu)
                            test_ind_min = np.ma.argmin(mudiff)

                        if test_ind_min == snapshot_min_ind:
                            mudict[sample_ind] = mu
                            break

            logger.info('Compute snapshots and errors for new parameter values')
            for sample_ind in range(i_ind*sample, (i_ind+1)*sample):
                mu = mudict[sample_ind]
                B[sample_ind] = NumpyVectorArray(StartB[m_ind].data)
                B[sample_ind].append(discretization.solve(mu))
                Basis[sample_ind] = gram_schmidt(B[sample_ind], discretization.products['l2'])

                V[sample_ind], discr = fp_system(m=m, n_grid=test_grid, basis_type='RB', basis_pl_discr=(Basis[sample_ind], discretization))

                relerror[sample_ind] = fperror(V[sample_ind], FDRef)

        # Choice of best basis for next m_ind
        snapshot_min_ind = np.ma.argmin(relerror)
        StartB[m_ind+1] = Basis[snapshot_min_ind]

        logger.info('Error for best solution at iteration m_ind={} : {}'.format(m_ind, relerror))

    return StartB[m_max], discretization


def greedy_fp_pod(m_max, i_refine, sample, test_grid, basis_out, seed=None):
    logger = getLogger('pymordemos.rb_to_fp.greedy_fp_pod')
    np.random.seed(seed)

    StartB = dict.fromkeys(range(m_max))
    StartB[0] = NumpyVectorArray.empty(501)

    problem = Fokkerplanck_V(test_case='SourceBeam', quadrature_count=(1, 1), P_parameter_range=(0.01, 1.2),
                           dxP_parameter_range=(-5.4, 0.9), dtP_parameter_range=(0, 5))

    n = 250
    v_discr, _ = discretize_elliptic_cg_plus(problem, diameter=1 / n)

    FDRef = np.zeros((1000, 500))
    with open('FD_reference_solution.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        i = 0
        for row in reader:
            for j in range(500):
                FDRef[i, j] = float(row[j])
            i += 1

    B = NumpyVectorArray.empty(v_discr.dim_solution, reserve=i_refine*sample*m_max)
    mudict = dict.fromkeys(range(i_refine*sample*m_max))
    relerror = dict.fromkeys(range(m_max))

    logger.info('Compute snapshots for initial parameter values')

    snapshot_ind = 0
    for mu in problem.parameter_space.sample_randomly(sample):
        snapshot = v_discr.solve(mu)
        B.append(snapshot)
        mudict[snapshot_ind] = mu
        snapshot_ind += 1

    for m_ind in range(m_max):
        logger.info('m_ind={}'.format(m_ind))
        m = m_ind+1
        relerror[m_ind] = np.ones(sample*(i_refine*(m_ind+1)+1))*10000

        logger.info('Compute errors for present snapshots')
        for ind in range(snapshot_ind):
            Basis = NumpyVectorArray(StartB[m_ind].data)
            Basis.append(NumpyVectorArray(B.data[ind, :]))
            Basis = gram_schmidt(Basis, v_discr.products['l2'])

            V, fp_discr = fp_system(m=m, basis_type='RB', n_grid=test_grid, basis_pl_discr=(Basis, v_discr))

            relerror[m_ind][ind] = fperror(V, FDRef)

        snapshot_min_ind = np.ma.argmin(relerror[m_ind])

        for i_ind in range(i_refine):
            mudiff = np.zeros(snapshot_ind)

            s_now = snapshot_ind

            logger.info('Generate new parameter values near best resolved parameter value')
            logger.info('i_ind={}'.format(i_ind))
            para_space = problem.parameter_space
            xi = 0.8

            for ind in range(sample):
                i = 0

                while True:
                    if i >= 1000:
                        # If necessary, diminish parameter space for random search
                        ranges_old = para_space.ranges
                        ranges_new = {'P': (float(ranges_old['P'][0] + (mudict[snapshot_min_ind]['P'][0] - ranges_old['P'][0])*xi),
                                                float(ranges_old['P'][1] - (ranges_old['P'][1] - mudict[snapshot_min_ind]['P'])*xi)),
                                        'dxP': (float(ranges_old['dxP'][0] + (mudict[snapshot_min_ind]['dxP'] - ranges_old['dxP'][0])*xi),
                                                float(ranges_old['dxP'][1] - (ranges_old['dxP'][1] - mudict[snapshot_min_ind]['dxP'])*xi)),
                                        'dtP': (float(ranges_old['dtP'][0] + (mudict[snapshot_min_ind]['dtP'] - ranges_old['dtP'][0])*xi),
                                                float(ranges_old['dtP'][1] - (ranges_old['dtP'][1] - mudict[snapshot_min_ind]['dtP'])*xi)),
                                        'dirich': (float(ranges_old['dirich'][0] + (min(mudict[snapshot_min_ind]['dirich']) - ranges_old['dirich'][0])*xi),
                                                float(ranges_old['dirich'][1] - (ranges_old['dirich'][1] - max(mudict[snapshot_min_ind]['dirich']))*xi)),
                                        'qxpoint': (float(ranges_old['qxpoint'][0] + (mudict[snapshot_min_ind]['qxpoint'] - ranges_old['qxpoint'][0])*xi),
                                                float(ranges_old['qxpoint'][1] - (ranges_old['qxpoint'][1] - mudict[snapshot_min_ind]['qxpoint'])*xi)),
                                        'qtpoint': ranges_old['qtpoint']}
                        para_space = CubicParameterSpace({'P': (1, 1),
                                               'dxP': (1, 1),
                                               'dtP': (1, 1),
                                               'dirich': 2,
                                               'qxpoint': 0,
                                               'qtpoint': 0},
                                              ranges=ranges_new)
                        i = 0
                    for mu in para_space.sample_randomly(1):
                        for test_ind in range(s_now):
                            mudiff[test_ind] = proximity_func(mudict[test_ind], mu)
                        test_ind_min = np.ma.argmin(mudiff)

                    if test_ind_min == snapshot_min_ind:
                        mudict[snapshot_ind] = mu
                        snapshot_ind += 1
                        break
                    else:
                        i += 1

                snapshot = v_discr.solve(mu)
                B.append(snapshot)

                # Compute errors for new snapshots
                Basis = NumpyVectorArray(StartB[m_ind].data)
                Basis.append(snapshot)
                Basis = gram_schmidt(Basis, v_discr.products['l2'])

                V, fp_discr = fp_system(m=m, basis_type='RB', n_grid=test_grid, basis_pl_discr=(Basis, v_discr))

                relerror[m_ind][snapshot_ind-1] = fperror(V, FDRef)

            # Determine anchor point for next grid refinement
            snapshot_min_ind = np.ma.argmin(relerror)

        logger.info('Compute new basis as starting point for next iteration')
        StartB[m_ind+1] = pod(B, modes=m_ind+1, product=v_discr.products['l2'])

    logger.info('Compute final basis')
    FinalPOD = pod(B, modes=basis_out, product=v_discr.products['l2'])

    return FinalPOD, v_discr
