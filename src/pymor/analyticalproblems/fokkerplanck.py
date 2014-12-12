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
from pymordemos.Legendre_Discr import basis_discr
from pymor.parameters.base import Parameter


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

    def __init__(self,sysdim, problem, CFLtype, basis_type='Leg'):

        assert basis_type == 'Leg'

        assert problem in ('2Beams','2Pulses','SourceBeam')

        def basis_generation(type):

            if type=='Leg':
                discr=basis_discr(0,1000,[-1,1])
                grid=discr.visualizer.grid
                basis=Legendre.legpolchar(grid.quadrature_points(1,order=2)[:,0,0],sysdim)

                #def fp_flux(U,mu):
                #    #Matrizen fuer Legendre-Polynome
                #    m=mu['m']
                #    Minv=np.diag((2.*np.array(range(m+1))+1.)/2.)
                #    A = np.diag(np.array(range(1,m+1))/(1.+2.*np.array(range(0,m))),1) +np.diag(np.array(range(1,m+1))/(1.+2.*np.array(range(1,m+1))),-1)
                #    flux=np.dot(Minv,A)
                #    if m == 0:
                #        flux=flux[0,0]
                #    return flux*U

                inject_sid(fp_flux, str(FPProblem) + '.fp_flux')



            mprod=discr.l2_product
            M=mprod.apply2(basis,basis,False)
            dprod=discr.absorb_product
            D=dprod.apply2(basis,basis,False)
            sprod=discr.h1_product
            S=sprod.apply2(basis,basis,False)
            basis_werte=mprod.apply2(NumpyVectorArray(np.ones(np.shape(grid.quadrature_points(1,order=2)[:,0,0]))),basis,False)
            basis_rand_l=basis.data[:,0]
            basis_rand_r=basis.data[:,-1]
            return dict({'M':M, 'D':D,'S':S, 'basis_werte':basis_werte,'basis_rand_l':basis_rand_l,'basis_rand_r':basis_rand_r})


            #if output == 'Matrices':
            #    return M,D,S
            #elif output == 'basis_werte':
            #    return basis_werte
            #elif output == 'fp_flux':
            #    return fp_flux
            #else:
            #     print('no valid output')

        def fp_flux(U,mu):
            return U
        flux_function=GenericFunction(fp_flux, dim_domain=1, shape_range=(1,), parameter_type={'m':0})


        if problem == '2Pulses':
            domain = LineDomain([0., 7.])
            stoptime=7.
            matlabcfl=0.5

            def IC(x):
                return 10.**(-4)

            def BCfuncl(t):
                return 0
            def BCfuncr(t):
                return 0
            def BCdeltal(t):
                return 100*np.exp(-0.5*(t-1)**2)
            def BCdeltar(t):
                return  100*np.exp(-0.5*(t-1)**2)


            def Tfunc(x):
                return 0
            def absorbfunc(x):
                return 0
            def Qfunc(x):
                return 0

        if problem == 'SourceBeam':

            domain = LineDomain([0.,3.])
            stoptime=4.
            matlabcfl=0.1


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
                return (x[...,0] >= 1)*(x[...,0]<= 1.5)
            def Tfunc(x):
                return 2.*(x > 1)*(x <=2) + 10.*(x > 2)

            def absorbfunc(x):
                return 1.*(x<=2)






#####################

        def initfunc(x,mu): #berechnet L2-Produkte mit Basisfunktionen
            basis_werte=mu['basis_werte']
            initial=IC(x)
            return initial*basis_werte[0,mu['komp']]+ x[...,0]*0

        initial_data=GenericFunction(initfunc,dim_domain=1,parameter_type={'komp':0, 'basis_werte':(1,sysdim)})

        def dirichfunc(x,mu):
            basis_werte=mu['basis_werte']
            basis_rand_l=mu['basis_rand_l']
            basis_rand_r=mu['basis_rand_r']
            dirichlet= BCfuncl(mu['_t'])*basis_werte[0,mu['komp']]*(x[...,0]<=0) + BCfuncr(mu['_t'])*basis_werte[0,mu['komp']]*(x[...,0]>0)
            wl=BCdeltal(mu['_t'])
            wr=BCdeltar(mu['_t'])
            dirichlet += wl*basis_rand_r[mu['komp']]*(x[...,0]<=0) +  wr*basis_rand_l[mu['komp']]*(x[...,0]>0)
            return dirichlet
        dirich_data=GenericFunction(dirichfunc,dim_domain=1,parameter_type={'m':0,'_t':0,'komp':0})


        def low_ord(UX,mu):
            lo=0*UX[...,0]
            Tx=Tfunc(UX[...,0])/2.
            S=mu['S']
            SU=np.dot(S,UX[...,1:sysdim+1].T)
            lo+=Tx*SU[mu['komp'],:]
            absorb=absorbfunc(UX[...,0])
            lo+=absorb*UX[:,mu['komp']+1]
            return lo
        low_order=GenericFunction(low_ord,dim_domain=sysdim+1,parameter_type={'komp':0})

        def source_func(x,mu):
            basis_werte=mu['basis_werte']
            Q=Qfunc(x)
            return Q*basis_werte[0,mu['komp']] + x[...,0]*0


        source_data=GenericFunction(source_func,dim_domain=1,parameter_type={'komp':0})
        self.rhs=source_data



        if CFLtype == 'matlab':
            CFL=matlabcfl
        elif CFLtype == 'computed':
            Lambda,W =np.linalg.eig(flux_matrix)
            CFL=1./(2.*np.max(np.abs(Lambda)))
            print('CFL=')
            print(CFL)
        else:
            raise NotImplementedError




        if problem == '2Beams':
            domain = LineDomain([-0.5, 0.5])
            stoptime=2.
            matlabcfl=0.9

            #def initfunc(x,mu):
            #    if mu['komp']==0:
            #        return x[...,0]*0+10.**(-4)*np.sqrt(2)
            #    else:
            #        return x[...,0]*0


            #def dirichfunc(x,mu):
                #dl,dr=Legendre.Sysdirichlet(400,mu['m'])
            #    dl=Legendre.legpolchar(1.,mu['komp']+1)*0.5
            #    dr=Legendre.legpolchar(-1.,mu['komp']+1)*0.5
            #    A=((x[...,0]<=0.)*dl.data[0,-1] + (x[...,0]>0.)*dr.data[0,-1])*100
            #    return A
            #dirich_data=GenericFunction(dirichfunc,dim_domain=1,parameter_type={'m':0,'_t':0,'komp':0})

            def absorbfunc(x):
                return 4.

            Tfunc=None
            source_data=None







        basis_dict=basis_generation(basis_type)
        flux_matrix=basis_dict['D']




        super(FPProblem, self).__init__(domain=domain,
                                             rhs=None,
                                             flux_function=flux_function,
                                             #low_order=low_order,
                                             initial_data=initial_data,
                                             dirichlet_data=dirich_data,
                                             T=stoptime, name='FPProblem')

        self.parameter_space = CubicParameterSpace({'komp':0,
                                                    'm':0,
                                                    'M':(sysdim,sysdim),
                                                    'S':(sysdim,sysdim),
                                                    'D':(sysdim,sysdim),
                                                    'basis_werte':(1,sysdim),
                                                    'basis_rand_l':(sysdim,),
                                                    'basis_rand_r':(sysdim,)}
                                                    ,minimum=0, maximum=20)
        self.basis_dict=basis_dict
        self.parameter_range=(0,20)
        self.flux_matrix=flux_matrix
        self.low_order=low_order
        self.CFL=CFL




