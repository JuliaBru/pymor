#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

'''Proof of concept for solving the poisson equation in 1D using linear finite elements and our grid interface

Usage:
    rb_to_fp.py


Options:

    -h, --help    this message
'''

from __future__ import absolute_import, division, print_function

#from docopt import docopt
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
from pymor.analyticalproblems.fokkerplanck_rb import Fokkerplanck_V
import pickle
from pymor.parameters.base import Parameter
from pymor.la.pod import pod
from pymor import core



getLogger('pymor.discretizations').setLevel('INFO')


def rb_solutions(rb_size=5, return_rb=False):

    core.cache.disable_caching()

    #a0 = GenericFunction(lambda V,mu: mu*V[...,0], dim_domain=1,parameter_type={'diffusionl':0})
    #a0 =GenericFunction(a0func,dim_domain=1)#,parameter_type={'diffusionl':0})


    #parameter_space = CubicParameterSpace({'diffusionl': 0}, 0.1, 1)
    #f0 = ProjectionParameterFunctional('diffusionl', 0)
    #f1 = GenericParameterFunctional(lambda mu: 1, {})

    #print('Solving on OnedGrid(({0},{0}))'.format(n))
    #Uxt=pickle.load(open("saveUxt.p",'rb'))
    #P=Uxt.data.T
    #dxP=np.zeros(P.shape)
    #dtP=np.zeros(P.shape)
    #for i in range(1,P.shape[0]-1):
    #    dxP[i,:]=1./(2.*3./(P.shape[0]-1)) * (P[i+1,:]-P[i-1,:])
    #dxP[0,:]*=0
    #dxP[-1,:]*=0
    #for j in range(1,P.shape[1]-1):
    #    dtP[:,j]=1./(2.*4./(P.shape[1]-1)) * (P[:,j+1]-P[:,j-1])
    #dtP[:,0]*=0
    #dtP[:,-1]*=0
    #Prange=(np.min(P),np.max(P))
    #dxPrange=(np.min(dxP),np.max(dxP))
    #dtPrange=(np.min(dtP),np.max(dtP))
    #print('P-range={}'.format(Prange))
    #print('dxP-range={}'.format(dxPrange))
    #print('dtP-range={}'.format(dtPrange))




    print('Setup Problem ...')
    #problem = EllipticPlusProblem(domain=LineDomain(domainint), rhs=rhs, diffusion_functions=(d0,),
    #                          diffusion_functionals=(f0,), absorb_function=a0, dirichlet_data=None,
    #                          name='1DProblem')

    problem=Fokkerplanck_V(delta=0, quadrature_count=(1,1),P_parameter_range=(0.01,1.2), dxP_parameter_range=(-5.4,0.9),dtP_parameter_range=(0,16.6))


    print('Discretize ...')
    n=200
    discretization, _ = discretize_elliptic_cg_plus(problem, diameter=1 / n)

    #mu=Parameter({'P':P,'dxP':dxP, 'dtP': dtP})#, 'dirich': (1,0)})


    print('The parameter type is {}'.format(discretization.parameter_type))

    U = discretization.type_solution.empty(discretization.dim_solution, reserve=1000)

    param_count=0
    for mu in problem.parameter_space.sample_randomly(500):
        param_count+=1
        mu['dirich']=(0,1)
        print('{}. Parameter'.format(param_count))
        print(mu)
        U.append(discretization.solve(mu))
        param_count+=1
        mu['dirich']=(1,0)
        print('{}. Parameter'.format(param_count))
        print(mu)
        U.append(discretization.solve(mu))



    rb = pod(U,modes=rb_size, orthonormalize=True,product=discretization.products['l2'],check_tol=0.01)




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
    #print('Plot ...')
    #discretization.visualize(rb, title='rb')


    if return_rb==True:
        return rb, discretization




if __name__ == '__main__':
    #args = docopt(__doc__)
    rb_solutions()