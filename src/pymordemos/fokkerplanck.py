# -*- coding: utf-8 -*-

# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

'''FP demo.

Usage:
  fokkerplanck.py [-hp] [--grid=NI] [--grid-type=TYPE] [--problem=TYPE] [--CFLtype=TYPE]
          [--num-flux=FLUX] [--m=COUNT] [--basis_type=TYPE]


Options:
  --grid=NI              Use grid with NI elements [default: 50].

  --grid-type=TYPE       Type of grid to use (oned) [default: oned].

  --problem=TYPE         Select the problem (2Beams, 2Pulses,SourceBeam) [default: SourceBeam].

  --CFLtype=TYPE         Type of CFL to use (matlab,computed)  [default: computed].

  --num-flux=FLUX        Numerical flux to use [default: godunov_upwind].

  --m=COUNT              Dimension of the system [default: 7].

  --basis_type=TYPE      Type of basis to use (Leg,RB) [default: RB].

  -h, --help             Show this message.


'''


#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)


import sys
import csv
import time
from functools import partial
import numpy as np
from docopt import docopt
import pymor.core as core
import pickle

from pymor.discretizers.advection import discretize_nonlinear_instationary_advection_fv_ndim
from pymor.domaindiscretizers import discretize_domain_default
from pymor.analyticalproblems.fokkerplanck import FPProblem
from pymor.grids import OnedGrid


#core.getLogger('pymor.algorithms').setLevel('INFO')
#core.getLogger('pymor.discretizations').setLevel('INFO')
#core.getLogger('pymor.la').setLevel('INFO')



def fp_demo(args):
    args['--grid'] = int(args['--grid'])
    args['--grid-type'] = args['--grid-type'].lower()
    args['--basis_type'] = args['--basis_type'].lower()
    assert args['--grid-type'] in ('oned')
    #args['--problem']=args['--problem']
    #args['--CFL']=float(args['--CFL'])
    args['--m'] = int(args['--m'])
    args['--num-flux'] = args['--num-flux'].lower()
    assert args['--num-flux'] in ('godunov_upwind')
    assert args['--CFLtype'] in ('matlab','computed')
    assert args['--basis_type'] in ('leg','rb')

    print('Setup Problem ...')
    grid_type_map = {'oned': OnedGrid}
    domain_discretizer = partial(discretize_domain_default, grid_type=grid_type_map[args['--grid-type']])
    problem = FPProblem(sysdim=args['--m'], problem=args['--problem'],CFLtype=args['--CFLtype'],basis_type=args['--basis_type'])


    print('Discretize ...')
    discretizer = discretize_nonlinear_instationary_advection_fv_ndim
    print(problem.domain.domain)
    xwidth=problem.domain.domain[1]-problem.domain.domain[0]
    discretization, data = discretizer(problem, args['--m'], diameter=float(xwidth) / args['--grid'],
                                       num_flux=args['--num-flux'],
                                       CFL=problem.CFL, domain_discretizer=domain_discretizer)
    print(discretization.operator.grid)



    mu=problem.basis_dict
    mu.update({'m':args['--m'] })
    #mu=(0,args['--m'],args['--basis_type'])


    sys.stdout.flush()
    # pr = cProfile.Profile()
    # pr.enable()
    tic = time.time()

    U,tvec = discretization.solve(mu)

    #u(x,t)=sum_{i=0}^{sysdim - 1} U[i]* int_{-1}^{1} legpol_i(mu)dmu
    #integral -1 bis 1 legpol_0 = sqrt(2), andere Integrale 0

    #U=U[0]*np.sqrt(2)





    V=U[0]*0
    for j in range(args['--m']):
        V.axpy(mu['basis_werte'][0,j],U[j])
    # pr.disable()
    print('Solving took {}s'.format(time.time() - tic))
    # pr.dump_stats('bla')
    discretization.visualize(V)
    #print(U)
    #Ud=U.data
    with open('sourcebeamfintest.csv','w') as csvfile:
        writer=csv.writer(csvfile)
        for j in range(np.shape(V.data)[0]):
            writer.writerow(V.data[j,:])
    #with open('zeiten.csv','w') as csvfile:
    #    writer=csv.writer(csvfile)
    #    writer.writerow(tvec)
    pickle.dump(V,open( "saveUxtLegwirklich.p", "wb" ))



if __name__ == '__main__':
    # parse arguments
    args = docopt(__doc__)
    # run demo
    fp_demo(args)



