# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function
import numpy as np

from pymor.analyticalproblems.advection import InstationaryAdvectionProblem
from pymor.core import Unpicklable, inject_sid
from pymor.domaindescriptions import LineDomain
from pymor.functions import ConstantFunction, GenericFunction
from pymor.parameters.spaces import CubicParameterSpace


class FPProblem(InstationaryAdvectionProblem, Unpicklable):
    '''One-dimensional Fokker-Planck problem.

    The problem is to solve ::

        ∂_t p(x, t)  +  A* ∂_x p(x, t)) = - (sigma(x,t) I + 1/2 T(x,t) S) p(x,t) + q(x,t)
                                       p(x, 0) = p_0(x)

    for p \in R^m with t in [a, b], x in [0, 2].

    Parameters
    ----------
    v
        The velocity v.
    circle
        If `True` impose periodic boundary conditions. Otherwise Dirichlet left,
        outflow right.
    initial_data_type
        Type of initial data (`'sin'` or `'bump'`).
    parameter_range
        The interval in which μ is allowed to vary.
    '''

    def __init__(self, problem='2Beams',sysdim=1):

        self.m=sysdim

        assert problem in ('2Beams')

        def fp_flux(U):
            #Matrizen fuer Legendre-Polynome
            #m=5
            Minv=np.diag((2.*np.array(range(m+1))+1.)/2.)
            A = np.diag(np.array(range(1,m+1))/(1.+2.*np.array(range(0,m))),1) +np.diag(np.array(range(1,m+1))/(1.+2.*np.array(range(1,m+1))),-1)
            return np.dot(Minv,A)

        #inject_sid(burgers_flux, str(BurgersProblem) + '.burgers_flux', v)

        #def burgers_flux_derivative(U, mu):
        #    U_exp = mu['exponent'] * (np.sign(U) * np.power(np.abs(U), mu['exponent']-1))
        #    R = U_exp * v
        #    return R
        #inject_sid(burgers_flux_derivative, str(BurgersProblem) + '.burgers_flux_derivative', v)

        flux_function = GenericFunction(fp_flux, dim_domain=1, shape_range=(m,),
                                        name='burgers_flux')

        #flux_function_derivative = GenericFunction(burgers_flux_derivative, dim_domain=1, shape_range=(1,),
        #                                           parameter_type={'exponent': 0},
        #                                           name='burgers_flux')

        if problem == '2Beams':
            def initial_data(x):
                return  10.**(-4)+x[...,0]*0
            # return 0.5 * (np.sin(2 * np.pi * x[..., 0]) + 1.)
            inject_sid(initial_data, str(FPProblem) + '.initial_data_2Beams')
            dirichlet_data = ConstantFunction(dim_domain=1, value=100.)
        #else:
        #    def initial_data(x):
        #        return (x[..., 0] >= 0.5) * (x[..., 0] <= 1) * 1
        #    inject_sid(initial_data, str(BurgersProblem) + '.initial_data_bump')
        #    dirichlet_data = ConstantFunction(dim_domain=1, value=0)

        initial_data = GenericFunction(initial_data, dim_domain=1)


        domain = LineDomain([-0.5, 0.5])

        super(FPProblem, self).__init__(domain=domain,
                                             rhs=None,
                                             flux_function=flux_function,
                                             initial_data=initial_data,
                                             dirichlet_data=dirichlet_data,
                                             T=4, name='FPProblem')

        self.parameter_space = CubicParameterSpace({'mu' : 0.},minimum=0., maximum=1.)




