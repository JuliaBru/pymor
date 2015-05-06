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

import numpy as np
from pymor.core import getLogger
from pymor.discretizers.ellipticplus import discretize_elliptic_cg_plus
from pymor.analyticalproblems.fokkerplanck_rb import Fokkerplanck_V
import pickle
from pymor.la import NumpyVectorArray
from pymor.la.pod import pod
from datetime import datetime as date
import time
import csv




getLogger('pymor.discretizations').setLevel('INFO')


def rb_solutions(problemname='SourceBeam', rb_size=50, return_rb=False, picklen=False,compute_rb=True):


    #print('Setup Problem ...')


    problem=Fokkerplanck_V(problem=problemname, delta=0, quadrature_count=(1,1),P_parameter_range=(0.01,1.2),
                           dxP_parameter_range=(-5.4,0.9),dtP_parameter_range=(0,5))

    #test_real_solution=True
    #print('Discretize ...')

    n=250

    discretization, _ = discretize_elliptic_cg_plus(problem, diameter=1 / n)

    #FPLoes=np.zeros((1000,500))
    #if test_real_solution == True:
    #    with open('fploes.csv', 'rb') as csvfile:
    #        reader = csv.reader(csvfile, delimiter=',')
    #        i=0
    #        for row in reader:
    #            for j in range(500):
    #                FPLoes[i,j]=float(row[j])
    #            i+=1



    #print(FPLoes)
    #FPLoe=NumpyVectorArray(FPLoes)




    if picklen == False:
        snapshots=2500

        np.random.seed()
        tic = time.time()

        d=date.now()
        for i in range(20):
            print(i)
            V = discretization.type_solution.empty(discretization.dim_solution, reserve=snapshots)
            j=0
            for mu in problem.parameter_space.sample_randomly(snapshots):
                j+=1

                if test_real_solution == True:
                    x=mu['qxpoint']
                    t=mu['qtpoint']
                    nx=int(round(x*500./3.))
                    nt=int(round(t*1000./4.))
                    if nx==0:
                        nx=1
                    if nx>=999:
                        nx=998
                    if nt==0:
                        nt=1
                    if nt>=499:
                        nt=498




                if (problemname == 'SourceBeam' and mu['qxpoint'] >=1) or not (problemname == 'SourceBeam'):
                    mu['dirich']=(1.,0.)
                    try:
                        mu['P']=np.array([[FPLoes[nt,nx],],])
                        mu['dtP']=np.array([[500./(2.*3.)*(FPLoes[nt+1,nx]-FPLoes[nt-1,nx]),],])
                        mu['dxP']=np.array([[1000./(2.*4.)*(FPLoes[nt,nx+1]-FPLoes[nt,nx-1]),],])
                        V.append(discretization.solve(mu))
                        print('Nr. {}'.format(j))
                    except:
                        print('Fehler bei mu={}'.format(mu))
                        V.append(NumpyVectorArray(np.zeros(discretization.dim_solution)))

                else:
                    V.append(NumpyVectorArray(np.zeros(discretization.dim_solution)))


                if (problemname == 'SourceBeam' and mu['qxpoint'] >=1) or not (problemname == 'SourceBeam'):
                    mu['dirich']=(0.,1.)
                    try:
                        mu['P']=np.array([[FPLoes[nt,nx],],])
                        mu['dtP']=np.array([[500./(2.*3.)*(FPLoes[nt+1,nx]-FPLoes[nt-1,nx]),],])
                        mu['dxP']=np.array([[1000./(2.*4.)*(FPLoes[nt,nx+1]-FPLoes[nt,nx-1]),],])
                        V.append(discretization.solve(mu))
                        print('Nr. {}'.format(j))
                    except:
                        print('Fehler bei mu={}'.format(mu))
                        V.append(NumpyVectorArray(np.zeros(discretization.dim_solution)))

                else:
                    V.append(NumpyVectorArray(np.zeros(discretization.dim_solution)))




            pickle.dump(V,open( "rb-daten {}, n={} Nr. {} {}.p".format(2*snapshots,n,i,d.strftime("%y-%m-%d %H:%M:%S")), "wb" ))









        if compute_rb==True:
            rb,sw=pod(V,modes=20,orthonormalize=True,product=discretization.products['l2'],check_tol=0.1)
            discretization.visualize(rb)
            pickle.dump(V,open( "rb {}, n={} {}.p".format(2*snapshots,n,d.strftime("%y-%m-%d %H:%M:%S")), "wb" ))




    if picklen == True:
        #rbvoll,sw =pickle.load(open("rb 50000 15-03-26 09:45:51.p",'rb'))
        a=1
        #rb=NumpyVectorArray(rbvoll.data[0:rb_size,:])


    if return_rb==True:
        return  a, discretization






if __name__ == '__main__':
    #args = docopt(__doc__)
    rb_solutions()