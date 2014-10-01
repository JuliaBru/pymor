#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

'''Proof of concept for solving the poisson equation in 1D using linear finite elements and our grid interface

Usage:
    cg_oned.py PROBLEM-NUMBER N PLOT

Arguments:
    PROBLEM-NUMBER    {0,1}, selects the problem to solve

    N                 Grid interval count

    PLOT              plot solution after solve?

Options:
    -h, --help    this message
'''

from __future__ import absolute_import, division, print_function

from docopt import docopt
import numpy as np

from pymor.analyticalproblems import EllipticProblem
from pymor.analyticalproblems.ellipticplus import EllipticPlusProblem
from pymor.core import getLogger
from pymor.discretizers import discretize_elliptic_cg
from pymor.discretizers.ellipticplus import discretize_elliptic_cg_plus
from pymor.domaindescriptions import LineDomain
from pymor.functions import GenericFunction, ConstantFunction
from pymor.parameters import CubicParameterSpace, ProjectionParameterFunctional, GenericParameterFunctional
from pymor.operators.cg import L2ProductP1

getLogger('pymor.discretizations').setLevel('INFO')


def rb_solutions(nrhs, n, domainint):
    rhs0 = GenericFunction(lambda X: np.ones(X.shape[:-1]) * 10, dim_domain=1)          # NOQA
    rhs1 = GenericFunction(lambda X: (X[..., 0] - 0.5) ** 2 * 1000, dim_domain=1)       # NOQA

    assert 0 <= nrhs <= 1, ValueError('Invalid rhs number.')
    rhs = eval('rhs{}'.format(nrhs))

    d0 = GenericFunction(lambda V: (1 - V[..., 0]**2), dim_domain=1)

    def a0func(V):
        #para=mu['diffusionl']
        return V[...,0]
    #a0 = GenericFunction(lambda V,mu: mu*V[...,0], dim_domain=1,parameter_type={'diffusionl':0})
    a0 =GenericFunction(a0func,dim_domain=1)#,parameter_type={'diffusionl':0})


    parameter_space = CubicParameterSpace({'diffusionl': 0}, 0.1, 1)
    f0 = ProjectionParameterFunctional('diffusionl', 0)
    #f1 = GenericParameterFunctional(lambda mu: 1, {})

    #print('Solving on OnedGrid(({0},{0}))'.format(n))

    #print('Setup Problem ...')
    problem = EllipticPlusProblem(domain=LineDomain(domainint), rhs=rhs, diffusion_functions=(d0,),
                              diffusion_functionals=(f0,), absorb_function=a0, dirichlet_data=None,
                              name='1DProblem')

    #print('Discretize ...')
    discretization, _ = discretize_elliptic_cg_plus(problem, diameter=1 / n)

    #print('The parameter type is {}'.format(discretization.parameter_type))

    #U = discretization.type_solution.empty(discretization.dim_solution)
    #for mu in parameter_space.sample_uniformly(10):
    #    U.append(discretization.solve(mu))

    #print('Computing System Matrices ...')
    #m=U._len
    #prod=discretization.l2_product
    #M=np.zeros([m,m])
    #for i in range(m):
    #    for j in range(m):
    #        M[i,j]=prod.apply2(U,U,i,j,True)
    #print(M)

    #M=prod.apply2(U,U,False)
    #print(M)
    #diag=prod.apply2(U,U,True)
    #print(diag)


    #if plot:
    #    print('Plot ...')
    #    discretization.visualize(U, title='Solution for diffusionl in [0.1, 1]')
    return discretization

if __name__ == '__main__':
    args = docopt(__doc__)
    nrhs = int(args['PROBLEM-NUMBER'])
    n = int(args['N'])
    plot = bool(args['PLOT'])
    rb_solutions(nrhs, n, [-1,1])
