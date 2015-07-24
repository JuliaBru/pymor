# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
#
# Author: Julia Brunken

"""
Recursive Definition of the Legendre polynomials and corresponding discretization.
Used by analyticalproblems.fokkerplanck.FPProblem
"""


import numpy as np
from pymor.la import NumpyVectorArray
from pymor.analyticalproblems.ellipticplus import EllipticPlusProblem
from pymor.discretizers.ellipticplus import discretize_elliptic_cg_plus
from pymor.domaindescriptions import LineDomain
from pymor.functions import GenericFunction, ConstantFunction
from pymor.parameters import ProjectionParameterFunctional

__author__ = 'j_brun16'

def legpol(V, m):
    L = np.array(np.zeros((m,) + np.shape(V)))
    L[0, ...] = 1.
    if m >= 2:
        L[1, ...] = V
        if m >= 3:
            L[2, ...] = 1./2.*(3.*V**2. - 1.)
            if m >= 4:
                for i in range(3, m):
                    L[i, ...] = 1./i*((2.*i-1.)*V*L[i-1, ...] - (i-1.)*L[i-2, ...])
    for i in range(m):
        # L[i, ...] *= np.sqrt((2.*i + 1.)/2.)*(V >= -1)*(V <= 1) # L2-Normed
        L[i, ...] *= (V >= -1)*(V <= 1)
    return NumpyVectorArray(L)


def basis_discr(n, domainint):
    rhs = ConstantFunction(dim_domain=1)

    d0 = GenericFunction(lambda V: (1 - V[..., 0]**2), dim_domain=1)

    def a0func(V):
        return V[..., 0]
    a0 = GenericFunction(a0func, dim_domain=1)

    f0 = ProjectionParameterFunctional('diffusionl', 0)

    problem = EllipticPlusProblem(domain=LineDomain(domainint), rhs=rhs, diffusion_functions=(d0,),
                                  diffusion_functionals=(f0,), absorb_functions=(a0,), dirichlet_data=None,
                                  name='1DProblem')

    discretization, _ = discretize_elliptic_cg_plus(problem, diameter=1. / n)

    return discretization
