# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
__author__ = 'j_brun16'

#from __future__ import absolute_import, division, print_function

from itertools import product

from pymor.analyticalproblems.ellipticplus import EllipticPlusProblem
from pymor.core import Unpicklable, inject_sid
from pymor.domaindescriptions import LineDomain
from pymor.functions import GenericFunction, ConstantFunction
from pymor.parameters import CubicParameterSpace, GenericParameterFunctional
import numpy as np




class Fokkerplanck_V(EllipticPlusProblem, Unpicklable):
    '''Analytical description of the 1D velocity part of the Fokkerplanck equation.

    This problem is to solve the elliptic equation ::

     - a(μ)  d_v (1-v²) d_v phi(v) + b(μ) v phi(v) + c(mu) phi(v) = Q(v,μ)

    on the domain [-1-delta , 1 + delta] with Dirichlet zero boundary values.
    The parameter functionals are evaluated with a quadrature formula dependent on problem-specific functions and parameters



    The Problem is implemented as an |EllipticProblem|

    Parameters
    ----------
    problem
        The name of the problem to solve
    delta
        Computational domain will be [-1-delta, 1+delta]
    quadrature_count
        A tuple (nx,nt). Number of quadrature points in x and t directions
    P_parameter_range
        A tuple (P_min, P_max). The values of P on each quadrature point lies in [P_min,P_max]
    dxP_parameter_range
        A tuple (dxP_min, dxP_max). The values of d_x(P) on each quadrature point lie in [dxP_min,dxP_max]
    dtP_parameter_range
        A tuple (dtP_min, dtP_max). The values of d_t(P) on each quadrature point lie in [dtP_min,dtP_max]
    rhs
        The |Function| f(x, μ).
    '''

    def __init__(self, delta=0.5 , problem='SourceBeam', quadrature_count=(3,3), P_parameter_range=(0, 10), dxP_parameter_range=(0, 10),
                 dtP_parameter_range=(0, 10)):

        self.delta=delta
        assert delta >= 0
        self.domain = LineDomain((-1-delta,1+delta))

        if problem == 'SourceBeam':
            xdomain=(0.,3.)
            tdomain=(0.,4.)

            def IC(x,t):
                return 10.**(-4)
            def BCl(x,t):
                return 1
            def BCr(x,t):
                return 10**(-4)
            def Qfunc(x,t):
                return (x >= 1)*(x <= 1.5)
            def Tfunc(x,t):
                return 2.*(x > 1)*(x <=2) + 10.*(x > 2)
            def absorbfunc(x,t):
                return 1.*(x<=2)

        def xt_quadrature(f,xdomain,tdomain):
            assert f.ndim == 2
            xlength=xdomain[1]-xdomain[0]
            tlength=tdomain[1]-tdomain[0]
            nx=f.shape[0]
            nt=f.shape[1]
            dx=xlength/(nx-1)
            dt=tlength/(nt-1)
            Ix= dt*(np.sum(f,axis=1)- 0.5*(f[:,0]+f[:,-1]))
            I=dx*(np.sum(Ix)-0.5*(Ix[0]+Ix[-1]))
            return I

        parameter_range={'P': P_parameter_range, 'dxP': dxP_parameter_range, 'dtP': dtP_parameter_range, 'dirich':(0,0.5)}
        parameter_space = CubicParameterSpace({'P': (quadrature_count[0], quadrature_count[1]),
                                               'dxP':(quadrature_count[0], quadrature_count[1]),
                                               'dtP':(quadrature_count[0], quadrature_count[1]),
                                               'dirich':(2)},
                                              ranges=parameter_range)

        xpoints=np.linspace(xdomain[0],xdomain[1],quadrature_count[0])
        tpoints=np.linspace(tdomain[0],tdomain[1],quadrature_count[1])




        #compute the parameter-dependent parts of the PDE
        def param_a(mu):
            # (T/2* P,P)_x_t
            P=mu['P']
            Tmatr=np.zeros(quadrature_count)
            assert P.shape == Tmatr.shape
            for i in range(quadrature_count[0]):
                for j in range(quadrature_count[1]):
                    Tmatr[i,j]=Tfunc(xpoints[i],tpoints[j])
            F=np.multiply(np.multiply(P,P),Tmatr)
            ret=xt_quadrature(F,xdomain,tdomain)
            print('param_a={}'.format(ret))
            return ret

        self.diffusion_functionals=[GenericParameterFunctional(param_a,{'P':(quadrature_count[0], quadrature_count[1]),
                                                                        'dxP':(quadrature_count[0], quadrature_count[1]),
                                                                        'dtP':(quadrature_count[0], quadrature_count[1]),}),]
        func= lambda v:(1.-v[...,0]**2)
        self.diffusion_functions= [GenericFunction(func,dim_domain=1),]

        def a0func(V):
            #para=mu['diffusionl']
            return V[...,0]

        def param_b(mu):
            # (d_xP,P)_x_t
            P=mu['P']
            dxP=mu['dxP']
            F=np.multiply(dxP,P)
            ret=xt_quadrature(F,xdomain,tdomain)
            print('param_b={}'.format(ret))
            return ret

        def param_c(mu):
            # (d_tP,P)_x_t + (sigma_a P,P)_xt
            P=mu['P']
            dtP=mu['dtP']
            sigmamatr=np.zeros(quadrature_count)
            assert P.shape == sigmamatr.shape
            for i in range(quadrature_count[0]):
                for j in range(quadrature_count[1]):
                    sigmamatr[i,j]=absorbfunc(xpoints[i],tpoints[j])
            F=np.multiply(dtP,P)+np.multiply(np.multiply(P,P),sigmamatr)
            ret=xt_quadrature(F,xdomain,tdomain)
            print('param_c={}'.format(ret))
            return ret


        def absorb_func(v,mu):
            return a0func(v)*param_b(mu)+param_c(mu)
        self.absorb_function=GenericFunction(absorb_func,dim_domain=1, parameter_type={'P':(quadrature_count[0], quadrature_count[1]),
                                                                        'dxP':(quadrature_count[0], quadrature_count[1]),
                                                                        'dtP':(quadrature_count[0], quadrature_count[1]),})


        def rhs_func(v,mu):
            P=mu['P']
            Qmatr=np.zeros(quadrature_count)
            for i in range(quadrature_count[0]):
                for j in range(quadrature_count[1]):
                    Qmatr[i,j]=Qfunc(xpoints[i],tpoints[j])
            F=np.multiply(P,Qmatr)
            ret= xt_quadrature(F,xdomain,tdomain)+v[...,0]*0
            print('rhs={}'.format(ret))
            return ret
        self.rhs=GenericFunction(rhs_func,dim_domain=1, parameter_type={'P':(quadrature_count[0],quadrature_count[1])})

        def dirich_func(v,mu):
            dirich_values=mu['dirich']
            return dirich_values[0]*(v[...,0]<=0) + dirich_values[1]*(v[...,0]>0)
        self.dirichlet_data=GenericFunction(dirich_func,dim_domain=1, parameter_type={'dirich':(2)})







        # creating the id-string once for every diffusion function reduces the size of the pickled sid
        #diffusion_function_id = str(ThermalBlockProblem) + '.diffusion_function'

        #def diffusion_function_factory(x, y):
        #    func = lambda X: (1. * (X[..., 0] >= x * dx) * (X[..., 0] < (x + 1) * dx)
        #                         * (X[..., 1] >= y * dy) * (X[..., 1] < (y + 1) * dy))
        #    inject_sid(func, diffusion_function_id, x, y, dx, dy)
        #    return GenericFunction(func, dim_domain=2, name='diffusion_function_{}_{}'.format(x, y))

        #def parameter_functional_factory(x, y):
        #    return ProjectionParameterFunctional(component_name='diffusion',
         #                                        component_shape=(num_blocks[1], num_blocks[0]),
         #                                        coordinates=(num_blocks[1] - y - 1, x),
        #                                         name='diffusion_{}_{}'.format(x, y))

        #diffusion_functions = tuple(diffusion_function_factory(x, y)
        #                            for x, y in product(xrange(num_blocks[0]), xrange(num_blocks[1])))
        #parameter_functionals = tuple(parameter_functional_factory(x, y)
        #                              for x, y in product(xrange(num_blocks[0]), xrange(num_blocks[1])))

        #super(ThermalBlockProblem, self).__init__(domain, rhs, diffusion_functions, parameter_functionals,
        #                                          name='ThermalBlock')
        self.parameter_space = parameter_space
        self.parameter_range = parameter_range