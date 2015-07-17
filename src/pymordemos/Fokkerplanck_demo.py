# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

'''Fokkerplanck demo.

Usage:
  Fokkerplanck_demo.py [--grid=NI] [--help] [--plot_solution]
  [--compute_error] [--save_pickled] [--save_csv] [--save_time]
  MODEL_TYPE_NO MODEL_ORDER


Arguments:
  MODEL_TYPE    Number of reduced model type:
                1: Legendre
                2: FP-RB with POD basis from precomputed snapshots (e.g. MATLAB snapshots)
                3: FP-RB with POD basis from snapshots from random parameter values
                4: FP-RB with Boundary-POD basis
                5: FP-RB with Greedy basis
                6: FP-RB with Greedy-POD basis

  MODEL_ORDER   Size of the reduced system (Number of basis functions, NOT highest polynomial order in Legendre model)


Options:
  --grid=NI             Use grid with NI elements [default: 500].

  --plot_solution       Plot the solution.

  --compute_error       Compute and print the error of the reduced solution

  --save_pickled        Save solution in pickle format

  --save_csv            Save solution in csv file

  --save_time           Save vector of time steps associated with the returned solution in additional csv file

  -h, --help            Show this message.

'''


__author__ = 'j_brun16'


from docopt import docopt
from pymor.core import getLogger
import numpy as np
import csv
from pymor.la import NumpyVectorArray
from pymordemos.fokkerplanck import fp_system
from pymordemos.rb_to_fp import fp_random_snapshots,pod_from_snapshots,basis_plus_boundary,greedy_fp,fperror,greedy_fp_pod

getLogger('pymor.tools').setLevel('ERROR')
getLogger('pymordemos.rb_to_fp').setLevel('INFO')


def fokkerplanck_demo(args):
    assert int(args['MODEL_TYPE_NO']) in (1,2,3,4,5,6)
    type_no = int(args['MODEL_TYPE_NO'])
    m = int(args['MODEL_ORDER'])
    n_grid = int(args['--grid'])

    #---------------Legendre-----------------------------------
    if type_no==1:

        # Legendre solution can be computed for different test cases:
        #---Choose test case---

        #test_case='2Beams'
        #test_case='2Pulses'
        test_case='SourceBeam'
        #test_case='RectIC'

        basis_type='Leg'
        basis_pl_discr=None

        print('\nSolve {} test case with Legendre moments\n'
              'Model order is {}'.format(test_case,m))

    #---------------FP-RB solutions ----------------------------

    else:
        basis_type='RB'
        test_case='SourceBeam'

    #---------------Basis generation----------------------------

    if type_no == 2:

        # Import file containing the precomputed snapshots (for example from MATLAB solution)
        #Specify file name and size:
        file_name='MATLAB_snapshots.csv'
        grid_size=200
        number_of_snapshots=5000

        snapshots=np.zeros((number_of_snapshots,grid_size))
        with open(file_name, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            i=0
            for row in reader:
                for j in range(grid_size):
                    snapshots[i,j]=float(row[j])
                i+=1

        print('\nCompute POD basis from MATLAB snapshots')
        basis_pl_discr=pod_from_snapshots(snapshots=NumpyVectorArray(snapshots),basis_size=m)

        print('\nSolve {} test case with POD basis from MATLAB snapshots\n'
              'Model order is {}'.format(test_case,m))

    if type_no == 3 or type_no ==4:

        #Specify number of computed snapshots
        n_train=100

        print('\nCompute {} Random Snapshots'.format(n_train))
        snapshots,v_discr=fp_random_snapshots(n_train)

        print('\nCompute POD basis')
        basis_pl_discr=pod_from_snapshots(snapshots=snapshots,discretization=v_discr,basis_size=m)

        if type_no == 3:
            print('\nSolve {} test case with POD basis from {} random snapshots \n'
                  'Model order is {}'.format(test_case,n_train,m))

    if type_no == 4:

        #Extend POD basis with boundary function.

        #POD basis:
        basis,v_discr=basis_pl_discr

        #Choose boundary approximation type:
        #boundary_type='p=0.1'
        #boundary_type='p=0.01'
        #boundary_type='p=0.001'
        #boundary_type='p=0.0001'
        boundary_type='peak'

        print('\nExtend basis with boundary basis function')
        basis_pl_discr=basis_plus_boundary(basis,boundary_type,v_discr)

        print('\nSolve {} test case with POD-Boundary basis from {} random snapshots with delta approximation {}.\n '
              'Model order is {}'.format(test_case,n_train,boundary_type,m))

    if type_no == 5:
        #Choose parameters for greedy algorithm:
        imax=1
        sample=2
        test_grid=50
        #mmax has to be chosen as model order m given as argument

        print('\nDo greedy algorithm for basis generation')
        getLogger('pymordemos.rb_to_fp.greedy_fp').setLevel('INFO')
        basis_pl_discr=greedy_fp(mmax=m,imax=imax,sample=sample,test_grid=test_grid,seed=1)
        print('\nSolve {} test case with basis from greedy algorithm. Parameters: imax={}, test_grid={}.\n'
              'Model order is {}'.format(test_case,imax,test_grid,m))

    if type_no == 6:
        #Choose parameters for greedy POD algorithm:
        mmax=3
        imax=1
        sample=2
        test_grid=50

        print('\nDo greedy POD algorithm for basis generation')
        getLogger('pymordemos.rb_to_fp.greedy_fp_pod').setLevel('INFO')
        basis_pl_discr=greedy_fp_pod(mmax=mmax,imax=imax,sample=sample,test_grid=test_grid,basis_out=m,seed=1)
        print('\nSolve {} test case with basis from greedy POD algorithm. Parameters: mmax={}, imax={}, test_grid={}.\n'
              'Model order is {}'.format(test_case,mmax,imax,test_grid,m))

    #-------------------Compute Reduced Solution ---------------

    getLogger('pymor.algorithms').setLevel('INFO')
    getLogger('pymordemos').setLevel('INFO')

    fpsol,x_discr=fp_system(m=m,basis_type=basis_type,basis_pl_discr=basis_pl_discr, test_case=test_case, n_grid=n_grid,
                            save_csv=args['--save_csv'],save_time=args['--save_time'], save_pickled=args['--save_pickled'], CFL_type='Auto')

    #-------------------Error estimation------------------------

    if args['--compute_error']:

        #Import reference solution
        #Specify name and size of reference solution
        FD_file_name='FD_reference_solution.csv'
        FD_file_size=(1000,500)

        FDRef=np.zeros(FD_file_size)
        with open(FD_file_name, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            i=0
            for row in reader:
                for j in range(FD_file_size[1]):
                    FDRef[i,j]=float(row[j])
                i+=1
        print('\nError of reduced solution is {}'.format(fperror(fpsol,FDRef)))

    #------------------Plot solution----------------------------
    if args['--plot_solution']:
        x_discr.visualize(fpsol)



if __name__ == '__main__':
    # parse arguments
    args = docopt(__doc__)
    # run demo
    fokkerplanck_demo(args)