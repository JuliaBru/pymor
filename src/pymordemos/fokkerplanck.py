# -*- coding: utf-8 -*-

# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

'''FP demo.

Usage:
  fokkerplanck.py [-hp] [--grid=NI] [--grid-type=TYPE] [--initial-data=TYPE] [--lxf-lambda=VALUE] [--nt=COUNT]
          [--num-flux=FLUX]





Options:
  --grid=NI              Use grid with (2*NI)*NI elements [default: 60].

  --grid-type=TYPE       Type of grid to use (oned) [default: oned].

  --problem=TYPE         Select the problem (2Beams) [default: 2Beams].

  --lxf-lambda=VALUE     Parameter lambda in Lax-Friedrichs flux [default: 1].

  --nt=COUNT             Number of time steps [default: 100].

  --sysdim=COUNT         Dimension of the system [default: 1]


  --num-flux=FLUX        Numerical flux to use (lax_friedrichs, engquist_osher)
                         [default: lax_friedrichs].

  -h, --help             Show this message.


'''




#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)





import sys
import math as m
import time
from functools import partial


from docopt import docopt

import pymor.core as core
core.logger.MAX_HIERACHY_LEVEL = 2





from pymor.discretizers.advection import discretize_nonlinear_instationary_advection_fv
from pymor.domaindiscretizers import discretize_domain_default
from pymor.analyticalproblems.fokkerplanck import FPProblem
from pymor.grids import OnedGrid

core.getLogger('pymor.algorithms').setLevel('INFO')
core.getLogger('pymor.discretizations').setLevel('INFO')



def fp_demo(args):
    args['--grid'] = int(args['--grid'])
    args['--grid-type'] = args['--grid-type'].lower()
    assert args['--grid-type'] in ('oned')
    args['--lxf-lambda'] = float(args['--lxf-lambda'])
    args['--nt'] = int(args['--nt'])
    args['--num-flux'] = args['--num-flux'].lower()
    assert args['--num-flux'] in ('lax_friedrichs', 'engquist_osher')

    print('Setup Problem ...')
    grid_type_map = {'oned': OnedGrid}
    domain_discretizer = partial(discretize_domain_default, grid_type=grid_type_map[args['--grid-type']])
    problem = FPProblem(problem='2Beams',sysdim=1)


    print('Discretize ...')
    discretizer = discretize_nonlinear_instationary_advection_fv
    discretization, data = discretizer(problem,
                                       num_flux=args['--num-flux'], lxf_lambda=args['--lxf-lambda'],
                                       nt=args['--nt'], domain_discretizer=domain_discretizer)
    print(discretization.operator.grid)



    # U = discretization.solve(0)

    sys.stdout.flush()
    # pr = cProfile.Profile()
    # pr.enable()
    tic = time.time()
    U = discretization.solve()
    # pr.disable()
    print('Solving took {}s'.format(time.time() - tic))
    # pr.dump_stats('bla')
    discretization.visualize(U)

if __name__ == '__main__':
    # parse arguments
    args = docopt(__doc__)
    # run demo
    fp_demo(args)




