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

    def __init__(self, delta=0.5 , problem='SourceBeam', quadrature_count=(1,1), P_parameter_range=(0.2, 2), dxP_parameter_range=(0, 10),
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

        if problem == 'SourceBeamNeu':
            xdomain=(0.,3.)
            tdomain=(0.,4.)

            def IC(x,t):
                return 10.**(-4)
            def BCr(x,t):
                return 1
            def BCl(x,t):
                return 10**(-4)
            def Qfunc(x,t):
                return (x >= 1)*(x <= 1.5)
            def Tfunc(x,t):
                #return (2.*(x > 1)*(x <=2) + 10.*(x > 2)  )
                return x
            def absorbfunc(x,t):
                #return 1.*(x <= 2)
                return 0.

        if problem == 'RectIC':

            xdomain = (0.,7.)
            tdomain=(0.,8.)

            def IC(x,t):
                return 10.**(-4)*(x<3)+ 10.**(-4)*(x>4) + 10*(x>=3)*(x<=4)

            def BCfuncl(t):
                return 10**(-4)
            def BCfuncr(t):
                return 10**(-4)
            def BCdeltal(t):
                return 0
            def BCdeltar(t):
                return 0


            def Qfunc(x,t):
                return 0
            def Tfunc(x,t):
                return 1.

            def absorbfunc(x,t):
                return 0


        def xt_quadrature(f,xdomain,tdomain):
            xlength=xdomain[1]-xdomain[0]
            tlength=tdomain[1]-tdomain[0]
            if f.ndim == 2:
                nx=f.shape[0]
                nt=f.shape[1]
            if f.ndim == 1 or (nx == 1 and nt == 1):
                I=xlength*tlength*f
                I=I[0,0]
            else:
                dx=xlength/(nx-1)
                dt=tlength/(nt-1)
                Ix= dt*(np.sum(f,axis=1)- 0.5*(f[:,0]+f[:,-1]))
                I=dx*(np.sum(Ix)-0.5*(Ix[0]+Ix[-1]))
            return I



        if problem == 'SourceBeam':
            parameter_range={'P': P_parameter_range,
                             'dxP': dxP_parameter_range,
                             'dtP': dtP_parameter_range,
                             'dirich':(0,1),
                             'qxpoint':(1.,3.),
                             'qtpoint':tdomain}

        else:
            parameter_range={'P': P_parameter_range,
                             'dxP': dxP_parameter_range,
                             'dtP': dtP_parameter_range,
                             'dirich':(0,1),
                             'qxpoint':xdomain,
                             'qtpoint':tdomain}



        parameter_space = CubicParameterSpace({'P': (quadrature_count[0], quadrature_count[1]),
                                               'dxP':(quadrature_count[0], quadrature_count[1]),
                                               'dtP':(quadrature_count[0], quadrature_count[1]),
                                               'dirich':(2),
                                               'qxpoint':0,
                                               'qtpoint':0},
                                              ranges=parameter_range)

        def points(quadrature_count,mu):
            if quadrature_count == (1,1):
                xpoints=(mu['qxpoint'],)
                tpoints=(mu['qtpoint'],)
            else:
                xpoints=np.linspace(xdomain[0],xdomain[1],quadrature_count[0])
                tpoints=np.linspace(tdomain[0],tdomain[1],quadrature_count[1])
            return xpoints,tpoints




        #compute the parameter-dependent parts of the PDE
        def param_a(mu):
            # (T/2* P,P)_x_t
            P=mu['P']
            Tmatr=np.zeros(quadrature_count)
            xpoints,tpoints=points(quadrature_count,mu)
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
                                                                        'dtP':(quadrature_count[0], quadrature_count[1]),
                                                                        'qxpoint':0,'qtpoint':0}),]
        func= lambda v:(1.-v[...,0]**2)
        self.diffusion_functions= [GenericFunction(func,dim_domain=1),]

        def a0func(V):
            #para=mu['diffusionl']
            return V[...,0]
        def a1func(V):
            return V[...,0]*0+1.

        def param_b(mu):
            # (d_xP,P)_x_t
            P=mu['P']
            dxP=mu['dxP']
            F=np.multiply(dxP,P)
            ret=xt_quadrature(F,xdomain,tdomain)
            print('b={}'.format(ret))
            return ret

        def param_c(mu):
            # (d_tP,P)_x_t + (sigma_a P,P)_xt
            P=mu['P']
            dtP=mu['dtP']
            xpoints,tpoints=points(quadrature_count,mu)
            sigmamatr=np.zeros(quadrature_count)
            assert P.shape == sigmamatr.shape
            for i in range(quadrature_count[0]):
                for j in range(quadrature_count[1]):
                    sigmamatr[i,j]=absorbfunc(xpoints[i],tpoints[j])
            F=np.multiply(dtP,P)+np.multiply(np.multiply(P,P),sigmamatr)
            ret=xt_quadrature(F,xdomain,tdomain)
            print('c={}'.format(ret))
            return ret


        self.absorb_functions=(GenericFunction(a0func,dim_domain=1),GenericFunction(a1func,dim_domain=1))

        self.absorb_functionals=(GenericParameterFunctional(param_b,{'P':(quadrature_count[0], quadrature_count[1]),
                                                                        'dxP' : (quadrature_count[0], quadrature_count[1]),
                                                                        'dtP' : (quadrature_count[0], quadrature_count[1]),
                                                                        'qxpoint':0,'qtpoint':0}),
                                 GenericParameterFunctional(param_c,{'P':(quadrature_count[0], quadrature_count[1]),
                                                                        'dxP' : (quadrature_count[0], quadrature_count[1]),
                                                                        'dtP' : (quadrature_count[0], quadrature_count[1]),
                                                                        'qxpoint' : 0,'qtpoint' : 0}))

        def rhs_func(v,mu):
            P=mu['P']
            Qmatr=np.zeros(quadrature_count)
            xpoints,tpoints=points(quadrature_count,mu)
            for i in range(quadrature_count[0]):
                for j in range(quadrature_count[1]):
                    Qmatr[i,j]=Qfunc(xpoints[i],tpoints[j])
            F=np.multiply(P,Qmatr)
            ret= xt_quadrature(F,xdomain,tdomain)+v[...,0]*0
            return ret
        self.rhs=GenericFunction(rhs_func,dim_domain=1, parameter_type={'P':(quadrature_count[0],quadrature_count[1]),'qxpoint':0,'qtpoint':0})

        def dirich_func(v,mu):
            dirich_values=mu['dirich']
            return dirich_values[0]*(v[...,0]<=0) + dirich_values[1]*(v[...,0]>0)
        self.dirichlet_data=GenericFunction(dirich_func,dim_domain=1, parameter_type={'dirich':(2)})


        self.parameter_space = parameter_space
        self.parameter_range = parameter_range