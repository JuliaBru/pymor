# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

# Author: Julia Brunken


from __future__ import absolute_import, division, print_function
import numpy as np
from pymor.analyticalproblems.advection import InstationaryAdvectionProblem
from pymor.core import Unpicklable
from pymor.domaindescriptions import LineDomain
from pymor.functions import GenericFunction
from pymor.parameters.spaces import CubicParameterSpace
from pymor.analyticalproblems import Legendre
from pymor.la import NumpyVectorArray


class FPProblem(InstationaryAdvectionProblem, Unpicklable):
    """One-dimensional Fokker-Planck problem.

    The problem is to solve :

        ∂_t p(x, t)  +  A* ∂_x p(x, t)) = - (sigma(x,t) I + 1/2 T(x,t) S) p(x,t) + q(x,t)
                                p(x, 0) = p_0(x)

    for p \in R^m

    Parameters
    ----------
    sysdim
        Dimension m of the system
    test_case
        Name of the test case to solve
    basis_type
        'Leg' or 'RB'
    basis_pl_discr
        Has to be provided if basis_type == 'RB', tuple of basis (NumpyVectorArray) and corresponding
        discretization object
    """

    def __init__(self, sysdim, test_case, basis_type='Leg', basis_pl_discr=None):

        assert basis_type in ('Leg', 'RB')
        if basis_type == 'Leg':
            assert test_case in ('2Beams', '2Pulses', 'SourceBeam', 'SourceBeamNeu', 'RectIC')
        else:
            assert test_case == 'SourceBeam'
            assert basis_pl_discr is not None

        def basis_generation(type):
            if basis_pl_discr is not None:
                (basis, discr) = basis_pl_discr
                assert basis._len == sysdim
                grid = discr.visualizer.grid
            elif type == 'Leg':
                discr = Legendre.basis_discr(1000, [-1, 1])
                grid = discr.visualizer.grid
                basis = Legendre.legpol(grid.quadrature_points(1,order=2)[:, 0, 0], sysdim)

            mprod = discr.l2_product
            M = mprod.apply2(basis, basis, False)
            Minv = np.linalg.inv(M)
            dprod = discr.absorb_product
            D = dprod.apply2(basis, basis, False)
            MinvD = np.dot(Minv, D)
            sprod = discr.h1_product
            S = sprod.apply2(basis, basis, False)
            MinvS = np.dot(Minv, S)

            basis_werte = mprod.apply2(NumpyVectorArray(np.ones(np.shape(grid.quadrature_points(1, order=2)[:, 0, 0]))), basis, False)
            basis_rand_l = basis.data[:, 0]
            basis_rand_r = basis.data[:, -1]
            return dict({'Minv': Minv, 'MinvD': MinvD, 'MinvS': MinvS,
                         'basis_werte': basis_werte, 'basis_rand_l': basis_rand_l, 'basis_rand_r': basis_rand_r})

        def fp_flux(U, mu):
            return U
        flux_function = GenericFunction(fp_flux, dim_domain=1, shape_range=(1,), parameter_type={'m': 0})

        if test_case == '2Pulses':

            domain = LineDomain([0., 7.])
            stoptime = 7.

            def IC(x):
                return 10.**(-4)
            def BCfuncl(t):
                return 0
            def BCfuncr(t):
                return 0
            def BCdeltal(t):
                return 100*np.exp(-0.5*(t-1)**2)
            def BCdeltar(t):
                return 100*np.exp(-0.5*(t-1)**2)

            def Tfunc(x):
                return 0
            def absorbfunc(x):
                return 0
            def Qfunc(x):
                return 0

        if test_case == '2Beams':

            domain = LineDomain([-0.5, 0.5])
            stoptime = 2.

            def IC(x):
                return 10.**(-4)
            def BCfuncl(t):
                return 0
            def BCfuncr(t):
                return 0
            def BCdeltal(t):
                return 100.
            def BCdeltar(t):
                return 100.

            def Tfunc(x):
                return 0
            def absorbfunc(x):
                return 4.
            def Qfunc(x):
                return 0

        if test_case == 'SourceBeam':

            domain = LineDomain([0., 3.])
            stoptime = 4.

            def IC(x):
                return 10.**(-4)
            def BCfuncl(t):
                return 0
            def BCfuncr(t):
                return 10**(-4)
            def BCdeltal(t):
                return 1
            def BCdeltar(t):
                return 0

            def Qfunc(x):
                return (x[..., 0] >= 1)*(x[..., 0] <= 1.5)
            def Tfunc(x):
                return 2.*(x > 1)*(x <= 2) + 10.*(x > 2)
            def absorbfunc(x):
                return 1.*(x <= 2)

        if test_case == 'RectIC':

            domain = LineDomain([0., 7.])
            stoptime = 8.

            def IC(x):
                return 10.**(-4)*(x[..., 0] < 3) + 10.**(-4)*(x[..., 0] > 4) + 10*(x[..., 0] >= 3)*(x[..., 0] <= 4)
            def BCfuncl(t):
                return 10**(-4)
            def BCfuncr(t):
                return 10**(-4)
            def BCdeltal(t):
                return 0
            def BCdeltar(t):
                return 0

            def Qfunc(x):
                return 0
            def Tfunc(x):
                return 1.
            def absorbfunc(x):
                return 0

        def initfunc(x, mu):
            basis_werte = np.dot(mu['Minv'], mu['basis_werte'][0, :])
            initial = IC(x)
            return initial*basis_werte[mu['komp']] + x[..., 0]*0
        initial_data = GenericFunction(initfunc, dim_domain=1, parameter_type={'komp': 0, 'basis_werte': (1, sysdim)})

        def dirichfunc(x, mu):
            basis_werte = np.dot(mu['Minv'], mu['basis_werte'][0, :])
            basis_rand_l = np.dot(mu['Minv'], mu['basis_rand_l'])
            basis_rand_r = np.dot(mu['Minv'], mu['basis_rand_r'])
            dirichlet = BCfuncl(mu['_t'])*basis_werte[mu['komp']]*(x[..., 0] <= 0) + BCfuncr(mu['_t'])*basis_werte[mu['komp']]*(x[..., 0] > 0)
            wl = BCdeltal(mu['_t'])
            wr = BCdeltar(mu['_t'])
            dirichlet += (wl*basis_rand_r[mu['komp']]*(x[..., 0] <= 0)
                           + wr*basis_rand_l[mu['komp']]*(x[..., 0] > 0))*0.5  # 0.5 because of delta distribution on the boundary
            return dirichlet
        dirich_data = GenericFunction(dirichfunc, dim_domain=1, parameter_type={'m': 0, '_t': 0, 'komp': 0})

        def low_ord(UX, mu):
            lo = 0*UX[..., 0]
            Tx = Tfunc(UX[..., 0])/2.
            MinvS = mu['MinvS']
            SU = np.dot(MinvS, UX[..., 1:sysdim+1].T)
            lo += Tx*SU[mu['komp'], :]
            absorb = absorbfunc(UX[..., 0])
            lo += absorb*UX[:, mu['komp'] + 1]
            return lo
        low_order = GenericFunction(low_ord, dim_domain=sysdim+1, parameter_type={'komp': 0})

        def source_func(x, mu):
            basis_werte = mu['basis_werte']
            Minv = mu['Minv']
            Minvphi = np.dot(Minv, basis_werte[0, :])
            Q = Qfunc(x)
            return Q*Minvphi[mu['komp']] + x[..., 0]*0
        source_data = GenericFunction(source_func, dim_domain=1, parameter_type={'komp': 0})
        self.rhs = source_data

        basis_dict = basis_generation(basis_type)
        flux_matrix = basis_dict['MinvD']

        super(FPProblem, self).__init__(domain=domain,
                                             rhs=source_data,
                                             flux_function=flux_function,
                                             initial_data=initial_data,
                                             dirichlet_data=dirich_data,
                                             T=stoptime, name='FPProblem')

        self.parameter_space = CubicParameterSpace({'komp': 0,
                                                    'm': 0,
                                                    'Minv': (sysdim, sysdim),
                                                    'MinvS': (sysdim, sysdim),
                                                    'MinvD': (sysdim, sysdim),
                                                    'basis_werte': (1, sysdim),
                                                    'basis_rand_l': (sysdim,),
                                                    'basis_rand_r': (sysdim,)},
                                                    minimum=0, maximum=20)
        self.basis_dict = basis_dict
        self.parameter_range = (0, 20)
        self.flux_matrix = flux_matrix
        self.low_order = low_order
