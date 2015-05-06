# -*- coding: utf-8 -*-

# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

'''FP demo.

Usage:
  fokkerplanck.py [-hp] [--grid=NI] [--grid-type=TYPE] [--problem=TYPE] [--CFLtype=TYPE] [--CFL=VALUE]
          [--num-flux=FLUX] [--m=COUNT] [--basis_type=TYPE]


Options:
  --grid=NI              Use grid with NI elements [default: 500].

  --grid-type=TYPE       Type of grid to use (oned) [default: oned].

  --problem=TYPE         Select the problem (2Beams, 2Pulses,SourceBeam, SourceBeamNeu,RectIC) [default: 2Beams].

  --CFLtype=TYPE         Type of CFL to use (matlab,computed, given)  [default: given].

  --CFL=VALUE            Value to use instead of CFL condition [default: 0.5]

  --num-flux=FLUX        Numerical flux to use [default: godunov_upwind].

  --m=COUNT              Dimension of the system [default: 1].

  --basis_type=TYPE      Type of basis to use (Leg,RB,Picklen) [default: Leg].

  -h, --help             Show this message.


'''



import sys
import csv
import time
from functools import partial
import numpy as np
from docopt import docopt
import pymor.core as core
import pickle
from pymor.la import NumpyVectorArray

from pymor.discretizers.advection import discretize_nonlinear_instationary_advection_fv_ndim
from pymor.domaindiscretizers import discretize_domain_default
from pymor.analyticalproblems.fokkerplanck import FPProblem
from pymor.grids import OnedGrid
from datetime import datetime as date

def fp_demo(args, basis_pl_discr=None):
    args['--grid'] = int(args['--grid'])
    args['--grid-type'] = args['--grid-type'].lower()
    args['--basis_type'] = args['--basis_type'].lower()
    assert args['--grid-type'] in ('oned')
    args['--CFL']=float(args['--CFL'])
    args['--m'] = int(args['--m'])
    args['--num-flux'] = args['--num-flux'].lower()
    assert args['--num-flux'] in ('godunov_upwind')
    assert args['--CFLtype'] in ('matlab','computed','given')
    assert args['--basis_type'] in ('leg','rb','picklen','picklen1','picklen2','picklen3')

    #print('Setup Problem ...')
    grid_type_map = {'oned': OnedGrid}
    domain_discretizer = partial(discretize_domain_default, grid_type=grid_type_map[args['--grid-type']])
    problem = FPProblem(sysdim=args['--m'], problem=args['--problem'],CFLtype=args['--CFLtype'],basis_type=args['--basis_type'],basis_pl_discr=basis_pl_discr)


    #print('Discretize ...')
    discretizer = discretize_nonlinear_instationary_advection_fv_ndim
    xwidth=problem.domain.domain[1]-problem.domain.domain[0]

    if args['--CFLtype']=='given':
        CFL=args['--CFL']
    else:
        CFL=problem.CFL




    core.cache.clear_caches()
    mu=problem.basis_dict
    mu.update({'m':args['--m'] })

    sys.stdout.flush()
    tic = time.time()

    while True:
        try:
            discretization, data = discretizer(problem, args['--m'], diameter=float(xwidth) / args['--grid'],
                                       num_flux=args['--num-flux'],
                                       CFL=CFL, domain_discretizer=domain_discretizer, num_values=1000)
            U,tvec = discretization.solve(mu)
            break
        except ValueError:
            CFL*=0.75
            print('neue CFL={}'.format(CFL))


    V=U[0]*0

    for j in range(args['--m']):
        V.axpy(mu['basis_werte'][0,j],U[j])


    #print('Solving took {}s'.format(time.time() - tic))

    # FPLoes=np.zeros((1000,500))
    # if True:
    #     with open('fploes.csv', 'rb') as csvfile:
    #         reader = csv.reader(csvfile, delimiter=',')
    #         i=0
    #         for row in reader:
    #             for j in range(500):
    #                 FPLoes[i,j]=float(row[j])
    #             i+=1
    #
    #
    #
    # FPLoe=NumpyVectorArray(FPLoes)


    #discretization.visualize(V, title='{},{} {}'.format(args['--problem'], args['--basis_type'],args['--m']))


    #d=date.now()
    #with open('{} {} {}.csv'.format(d.strftime("%y-%m-%d %H:%M:%S"),args['--basis_type'] ,args['--m'] ),'w') as csvfile:
    #    writer=csv.writer(csvfile)
    #    for j in range(np.shape(V.data)[0]):
    #        writer.writerow(V.data[j,:])

    #with open('zeiten.csv','w') as csvfile:
    #    writer=csv.writer(csvfile)
    #    writer.writerow(tvec)
    #pickle.dump((V,tvec) ,open( '{} {} Mio {}.p'.format(d.strftime("%y-%m-%d %H:%M:%S"),args['--basis_type'] ,args['--m'] ), "wb" ))
    return V, discretization



if __name__ == '__main__':
    # parse arguments
    args = docopt(__doc__)
    # run demo
    fp_demo(args)



