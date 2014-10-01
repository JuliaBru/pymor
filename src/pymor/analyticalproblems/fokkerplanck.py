# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function
import numpy as np


from pymor.analyticalproblems.advection import InstationaryAdvectionProblem
from pymor.core import Unpicklable, inject_sid
from pymor.domaindescriptions import LineDomain
from pymor.functions import GenericFunction
from pymor.parameters.spaces import CubicParameterSpace
from pymor.analyticalproblems import Legendre
from pymor.la import NumpyVectorArray


class FPProblem(InstationaryAdvectionProblem, Unpicklable):
    '''One-dimensional Fokker-Planck problem.

    The problem is to solve ::

        ∂_t p(x, t)  +  A* ∂_x p(x, t)) = - (sigma(x,t) I + 1/2 T(x,t) S) p(x,t) + q(x,t)
                                       p(x, 0) = p_0(x)

    for p \in R^m

    Parameters
    ----------
    initial_data_type
        Type of initial data (`'sin'` or `'bump'`).
    parameter_range
        The interval in which μ is allowed to vary.
    '''

    def __init__(self,sysdim, problem='2Pulses'):



        assert problem in ('2Beams','2Pulses','SourceBeam')


        def fp_flux(U,mu):
            #Matrizen fuer Legendre-Polynome
            m=mu['m']
            Minv=np.diag((2.*np.array(range(m+1))+1.)/2.)
            A = np.diag(np.array(range(1,m+1))/(1.+2.*np.array(range(0,m))),1) +np.diag(np.array(range(1,m+1))/(1.+2.*np.array(range(1,m+1))),-1)
            flux=np.dot(Minv,A)
            if m == 0:
                flux=flux[0,0]

            return flux*U

        inject_sid(fp_flux, str(FPProblem) + '.fp_flux')

        flux_function=GenericFunction(fp_flux, dim_domain=1, shape_range=(1,), parameter_type={'m':0})




        if problem == '2Pulses':
            domain = LineDomain([0., 7.])
            stoptime=7.

            def initfunc(x,mu):
                if mu['komp']==0:
                    return x[...,0]*0+10.**(-4)*np.sqrt(2)
                else:
                    return x[...,0]*0
            initial_data=GenericFunction(initfunc,dim_domain=1,parameter_type={'komp':0})

            def dirichfunc(x,mu):
                dl,dr=Legendre.Sysdirichlet(400,mu['m'])
                A=((x[...,0]<=0.)*dl[mu['komp']] + (x[...,0]>0.)*dr[mu['komp']])*np.exp(-0.5*(mu['_t']-1)**2)*100
                return A
            dirich_data=GenericFunction(dirichfunc,dim_domain=1,parameter_type={'m':0,'_t':0,'komp':0})

            Tfunc=None
            absorbfunc=None
            source_data=None





        if problem == '2Beams':
            domain = LineDomain([-0.5, 0.5])
            stoptime=2.

            def initfunc(x,mu):
                if mu['komp']==0:
                    return x[...,0]*0+10.**(-4)*np.sqrt(2)
                else:
                    return x[...,0]*0
            initial_data=GenericFunction(initfunc,dim_domain=1,parameter_type={'komp':0})

            def dirichfunc(x,mu):
                dl,dr=Legendre.Sysdirichlet(400,mu['m'])
                A=((x[...,0]<=0.)*dl[mu['komp']] + (x[...,0]>0.)*dr[mu['komp']])*100
                return A
            dirich_data=GenericFunction(dirichfunc,dim_domain=1,parameter_type={'m':0,'_t':0,'komp':0})

            def absorbfunc(x):
                return 4.

            Tfunc=None
            source_data=None




        if problem == 'SourceBeam':
            domain = LineDomain([0.,3.])
            stoptime=4.

            def initfunc(x,mu):
                if mu['komp']==0:
                    return x[...,0]*0+10.**(-4)*np.sqrt(2)
                else:
                    return x[...,0]*0
            initial_data=GenericFunction(initfunc,dim_domain=1,parameter_type={'komp':0})

            def dirichfunc(x,mu):
                dl,dr=Legendre.Sysdirichlet(400,mu['m'])
                A=(x[...,0]<=1.)*dl[mu['komp']]*2 + (x[...,0]>1)*10.**(-4)
                return A
            dirich_data=GenericFunction(dirichfunc,dim_domain=1,parameter_type={'m':0,'komp':0})

            def sourcefunc(x,mu):
                if mu['komp']==0:
                    return (x[...,0] >= 1)*(x[...,0]<= 1.5)*np.sqrt(2)*0.5
                else:
                    return x[...,0]*0

            source_data=GenericFunction(sourcefunc,dim_domain=1,parameter_type={'komp':0})

            def Tfunc(x):
                return 2.*(x > 1)*(x <=2) + 10.*(x > 2)

            def absorbfunc(x):
                return 1.*(x<=2)

        M,D,S = Legendre.LegendreMatrices(1000,sysdim)
        flux_matrix = D

        if (Tfunc is not None) or (absorbfunc is not None):
            def low_ord(UX,mu):
                lo=0*UX[...,0]
                if Tfunc is not None:
                    Tx=Tfunc(UX[...,0])/2.
                    SU=np.dot(S,UX[...,1:sysdim+1].T)
                    lo+=Tx*SU[mu['komp'],:]
                if absorbfunc is not None:
                    absorb=absorbfunc(UX[...,0])
                    lo+=absorb*UX[:,mu['komp']+1]
                return lo
            low_order=GenericFunction(low_ord,dim_domain=sysdim+1,parameter_type={'komp':0})







        super(FPProblem, self).__init__(domain=domain,
                                             rhs=source_data,
                                             flux_function=flux_function,
                                             #low_order=low_order,
                                             initial_data=initial_data,
                                             dirichlet_data=dirich_data,
                                             T=stoptime, name='FPProblem')

        self.parameter_space = CubicParameterSpace({'m' : 0, 'komp':0},minimum=0, maximum=20)
        self.parameter_range=(0,20)
        self.flux_matrix=flux_matrix
        self.low_order=low_order





