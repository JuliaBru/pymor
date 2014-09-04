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

    def __init__(self,sysdim, problem='2Pulses'):



        assert problem in ('2Beams','2Pulses')


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




        initial_data=dict.fromkeys(range(sysdim))
        dirichlet_data=dict.fromkeys(range(sysdim))

        if problem == '2Pulses':
            domain = LineDomain([0, 7])
            stoptime=7.
            def dirichlet_data_func_0(x,mu):
                dl,dr=Legendre.Sysdirichlet(100,mu['m'])
                A=((x[...,0]<=0.)*dl[0] + (x[...,0]>0.)*dr[0])*np.exp(-0.5*(mu['_t']-1)**2)
                return A
            dirichlet_data[0]=GenericFunction(dirichlet_data_func_0,dim_domain=1,parameter_type={'m':0,'_t':0})

            if sysdim >= 2:
                def dirichlet_data_func_1(x,mu):
                    dl,dr=Legendre.Sysdirichlet(100,mu['m'])
                    A=((x[...,0]<=0.)*dl[1] + (x[...,0]>0.)*dr[1])*np.exp(-0.5*(mu['_t']-1)**2)
                    return A
                dirichlet_data[1]=GenericFunction(dirichlet_data_func_1,dim_domain=1,parameter_type={'m':0,'_t':0})

                if sysdim >= 3:
                    def dirichlet_data_func_2(x,mu):
                        dl,dr=Legendre.Sysdirichlet(100,mu['m'])
                        A=((x[...,0]<=0.)*dl[2] + (x[...,0]>0.)*dr[2])*np.exp(-0.5*(mu['_t']-1)**2)
                        return A
                    dirichlet_data[2]=GenericFunction(dirichlet_data_func_2,dim_domain=1,parameter_type={'m':0,'_t':0})

                    if sysdim >=4:
                        def dirichlet_data_func_3(x,mu):
                            dl,dr=Legendre.Sysdirichlet(100,mu['m'])
                            A=((x[...,0]<=0.)*dl[3] + (x[...,0]>0.)*dr[3])*np.exp(-0.5*(mu['_t']-1)**2)
                            return A
                        dirichlet_data[3]=GenericFunction(dirichlet_data_func_3,dim_domain=1,parameter_type={'m':0,'_t':0})

                        if sysdim >=5:
                            def dirichlet_data_func_4(x,mu):
                                dl,dr=Legendre.Sysdirichlet(100,mu['m'])
                                A=((x[...,0]<=0.)*dl[4] + (x[...,0]>0.)*dr[4])*np.exp(-0.5*(mu['_t']-1)**2)
                                return A
                            dirichlet_data[4]=GenericFunction(dirichlet_data_func_4,dim_domain=1,parameter_type={'m':0,'_t':0})

                            if sysdim >=6:
                                def dirichlet_data_func_5(x,mu):
                                    dl,dr=Legendre.Sysdirichlet(100,mu['m'])
                                    A=((x[...,0]<=0.)*dl[5] + (x[...,0]>0.)*dr[5])*np.exp(-0.5*(mu['_t']-1)**2)
                                    return A
                                dirichlet_data[5]=GenericFunction(dirichlet_data_func_5,dim_domain=1,parameter_type={'m':0,'_t':0})

                                if sysdim >=7:
                                    def dirichlet_data_func_6(x,mu):
                                        dl,dr=Legendre.Sysdirichlet(100,mu['m'])
                                        A=((x[...,0]<=0.)*dl[6] + (x[...,0]>0.)*dr[6])*np.exp(-0.5*(mu['_t']-1)**2)
                                        return A
                                    dirichlet_data[6]=GenericFunction(dirichlet_data_func_6,dim_domain=1,parameter_type={'m':0,'_t':0})

                                    if sysdim >=8:
                                        def dirichlet_data_func_7(x,mu):
                                            dl,dr=Legendre.Sysdirichlet(100,mu['m'])
                                            A=((x[...,0]<=0.)*dl[7] + (x[...,0]>0.)*dr[7])*np.exp(-0.5*(mu['_t']-1)**2)
                                            return A
                                        dirichlet_data[7]=GenericFunction(dirichlet_data_func_7,dim_domain=1,parameter_type={'m':0,'_t':0})
            def initial_data_1(x):
                return x[...,0]*0+10.**(-4)
            def initial_data_z(x):
                return x[...,0]*0
            initial_data[0]=GenericFunction(initial_data_1,dim_domain=1)
            for j in range(1,sysdim):
                initial_data[j]=GenericFunction(initial_data_z, dim_domain=1)
            low_order=None



        if problem == '2Beams':
            absorb=dict.fromkeys(range(sysdim))
            domain = LineDomain([-0.5, 0.5])
            stoptime=4.
            def dirichlet_data_func_0(x,mu):
                dl,dr=Legendre.Sysdirichlet(100,mu['m'])
                A=((x[...,0]<=0.)*dl[0] + (x[...,0]>0.)*dr[0])
                return A
            def absorb_func_0(u):
                return 4.*u[:,0]
            dirichlet_data[0]=GenericFunction(dirichlet_data_func_0,dim_domain=1,parameter_type={'m':0,'_t':0})
            absorb[0]=GenericFunction(absorb_func_0,dim_domain=sysdim)

            if sysdim >= 2:
                def dirichlet_data_func_1(x,mu):
                    dl,dr=Legendre.Sysdirichlet(100,mu['m'])
                    A=((x[...,0]<=0.)*dl[1] + (x[...,0]>0.)*dr[1])
                    return A
                def absorb_func_1(u):
                    return 4.*u[:,1]
                dirichlet_data[1]=GenericFunction(dirichlet_data_func_1,dim_domain=1,parameter_type={'m':0,'_t':0})
                absorb[1]=GenericFunction(absorb_func_1,dim_domain=sysdim)

                if sysdim >= 3:
                    def dirichlet_data_func_2(x,mu):
                        dl,dr=Legendre.Sysdirichlet(100,mu['m'])
                        A=((x[...,0]<=0.)*dl[2] + (x[...,0]>0.)*dr[2])
                        return A
                    def absorb_func_2(u):
                        return 4.*u[:,2]
                    dirichlet_data[2]=GenericFunction(dirichlet_data_func_2,dim_domain=1,parameter_type={'m':0,'_t':0})
                    absorb[2]=GenericFunction(absorb_func_2,dim_domain=sysdim)

                    if sysdim >=4:
                        def dirichlet_data_func_3(x,mu):
                            dl,dr=Legendre.Sysdirichlet(100,mu['m'])
                            A=((x[...,0]<=0.)*dl[3] + (x[...,0]>0.)*dr[3])
                            return A
                        def absorb_func_3(u):
                            return 4.*u[:,3]
                        dirichlet_data[3]=GenericFunction(dirichlet_data_func_3,dim_domain=1,parameter_type={'m':0,'_t':0})
                        absorb[3]=GenericFunction(absorb_func_3,dim_domain=sysdim)

                        if sysdim >=5:
                            def dirichlet_data_func_4(x,mu):
                                dl,dr=Legendre.Sysdirichlet(100,mu['m'])
                                A=((x[...,0]<=0.)*dl[4] + (x[...,0]>0.)*dr[4])
                                return A
                            def absorb_func_4(u):
                                return 4.*u[:,4]
                            dirichlet_data[4]=GenericFunction(dirichlet_data_func_4,dim_domain=1,parameter_type={'m':0,'_t':0})
                            absorb[4]=GenericFunction(absorb_func_4,dim_domain=sysdim)

                            if sysdim >=6:
                                def dirichlet_data_func_5(x,mu):
                                    dl,dr=Legendre.Sysdirichlet(100,mu['m'])
                                    A=((x[...,0]<=0.)*dl[5] + (x[...,0]>0.)*dr[5])
                                    return A
                                def absorb_func_5(u):
                                    return 4.*u[:,5]
                                dirichlet_data[5]=GenericFunction(dirichlet_data_func_5,dim_domain=1,parameter_type={'m':0,'_t':0})
                                absorb[5]=GenericFunction(absorb_func_5,dim_domain=sysdim)

                                if sysdim >=7:
                                    def dirichlet_data_func_6(x,mu):
                                        dl,dr=Legendre.Sysdirichlet(100,mu['m'])
                                        A=((x[...,0]<=0.)*dl[6] + (x[...,0]>0.)*dr[6])
                                        return A
                                    def absorb_func_6(u):
                                        return 4.*u[:,6]
                                    dirichlet_data[6]=GenericFunction(dirichlet_data_func_6,dim_domain=1,parameter_type={'m':0,'_t':0})
                                    absorb[6]=GenericFunction(absorb_func_6,dim_domain=sysdim)

                                    if sysdim >=8:
                                        def dirichlet_data_func_7(x,mu):
                                            dl,dr=Legendre.Sysdirichlet(100,mu['m'])
                                            A=((x[...,0]<=0.)*dl[7] + (x[...,0]>0.)*dr[7])
                                            return A
                                        def absorb_func_7(u):
                                            return 4.*u[:,7]
                                        dirichlet_data[7]=GenericFunction(dirichlet_data_func_7,dim_domain=1,parameter_type={'m':0,'_t':0})
                                        absorb[7]=GenericFunction(absorb_func_7,dim_domain=sysdim)
            def initial_data_1(x):
                return x[...,0]*0+10.**(-4)
            def initial_data_z(x):
                return x[...,0]*0
            initial_data[0]=GenericFunction(initial_data_1,dim_domain=1)
            for j in range(1,sysdim):
                initial_data[j]=GenericFunction(initial_data_z, dim_domain=1)

            low_order=absorb




 #       for j in range(3):
 #           str='initial_data'+str(j)
 #           def


        flux_matrix=Legendre.Sysmatrix(100,sysdim)





        #dirichlet_data=GenericFunction(dirichlet_data,dim_domain=1,parameter_type={'m':0})




        super(FPProblem, self).__init__(domain=domain,
                                             rhs=None,
                                             flux_function=flux_function,
                                             #low_order=low_order,
                                             initial_data=initial_data,
                                             dirichlet_data=dirichlet_data,
                                             T=stoptime, name='FPProblem')

        self.parameter_space = CubicParameterSpace({'m' : 0},minimum=0, maximum=20)
        self.parameter_range=(0,20)
        self.flux_matrix=flux_matrix
        self.low_order=low_order





