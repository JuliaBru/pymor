# -*- coding: utf-8 -*-

# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

'''FP demo.

Usage:
  fokkerplanck.py [-hp] [--grid=NI] [--grid-type=TYPE] [--problem=TYPE] [--nt=COUNT]
          [--num-flux=FLUX] [--m=COUNT]


Options:
  --grid=NI              Use grid with (2*NI)*NI elements [default: 100].

  --grid-type=TYPE       Type of grid to use (oned) [default: oned].

  --problem=TYPE         Select the problem (2Beams, 2Pulses) [default: 2Beams].

  --nt=COUNT             Number of time steps [default: 400].

  --num-flux=FLUX        Numerical flux to use [default: godunov_upwind].

  --m=COUNT              Dimension of the system [default: 7].

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
core.logger.MAX_HIERACHY_LEVEL = 2
from pymor.discretizers.advection import discretize_nonlinear_instationary_advection_fv_ndim
from pymor.domaindiscretizers import discretize_domain_default
from pymor.analyticalproblems.fokkerplanck import FPProblem
from pymor.grids import OnedGrid


core.getLogger('pymor.algorithms').setLevel('INFO')
core.getLogger('pymor.discretizations').setLevel('INFO')



def fp_demo(args):
    args['--grid'] = int(args['--grid'])
    args['--grid-type'] = args['--grid-type'].lower()
    assert args['--grid-type'] in ('oned')
    args['--problem']=args['--problem']
    args['--nt'] = int(args['--nt'])
    args['--m'] = int(args['--m'])
    args['--num-flux'] = args['--num-flux'].lower()
    assert args['--num-flux'] in ('godunov_upwind')

    print('Setup Problem ...')
    grid_type_map = {'oned': OnedGrid}
    domain_discretizer = partial(discretize_domain_default, grid_type=grid_type_map[args['--grid-type']])
    problem = FPProblem(sysdim=args['--m'], problem=args['--problem'])


    print('Discretize ...')
    discretizer = discretize_nonlinear_instationary_advection_fv_ndim
    discretization, data = discretizer(problem, args['--m'], diameter=2. / args['--grid'],
                                       num_flux=args['--num-flux'],
                                       nt=args['--nt'], domain_discretizer=domain_discretizer)
    print(discretization.operator.grid)




    mu=args['--m']


    sys.stdout.flush()
    # pr = cProfile.Profile()
    # pr.enable()
    tic = time.time()

    U = discretization.solve(mu)
    U=U[0]
    # pr.disable()
    print('Solving took {}s'.format(time.time() - tic))
    # pr.dump_stats('bla')
    discretization.visualize(U)
    print(U)
    Ud=U.data
    with open('loesung.csv','w') as csvfile:
        writer=csv.writer(csvfile)
        for j in range(np.shape(Ud)[0]):
            writer.writerow(Ud[j,:])


if __name__ == '__main__':
    # parse arguments
    args = docopt(__doc__)
    # run demo
    fp_demo(args)




