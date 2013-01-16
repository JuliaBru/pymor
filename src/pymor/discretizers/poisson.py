from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from scipy.sparse.linalg import bicg
import matplotlib.pyplot as pl

import pymor.core as core
from pymor.analyticalproblems import PoissonProblem
from pymor.domaindescriptions import BoundaryType
from pymor.domaindiscretizers import DefaultDomainDiscretizer
from pymor.discreteoperators.cg import DiffusionOperatorP1D2, L2ProductFunctionalP1D2
from pymor.discreteoperators.affine import LinearAffinelyDecomposedOperator
from pymor.discretizations import EllipticDiscretization
from pymor.grids import TriaGrid


class PoissonCGDiscretizer(object):

    def __init__(self, analytical_problem):
        assert isinstance(analytical_problem, PoissonProblem)
        self.analytical_problem = analytical_problem

    def discretize_domain(self, domain_discretizer=None, diameter=None):
        domain_discretizer = domain_discretizer or DefaultDomainDiscretizer(self.analytical_problem.domain)
        if diameter is None:
            return domain_discretizer.discretize()
        else:
            return domain_discretizer.discretize(diameter=diameter)

    def discretize(self, domain_discretizer=None, diameter=None, grid=None, boundary_info=None):
        assert grid is None or boundary_info is None
        assert boundary_info is None or grid is None
        assert grid is None or domain_discretizer is None
        if grid is None:
            grid, boundary_info = self.discretize_domain(domain_discretizer, diameter)

        assert isinstance(grid, TriaGrid)

        p = self.analytical_problem

        if p.parameter_dependent:
            L0 = DiffusionOperatorP1D2(grid, boundary_info, diffusion_constant=0, name='diffusion_boundary_part')

            Li = tuple(DiffusionOperatorP1D2(grid, boundary_info, diffusion_function=df, dirichlet_clear_diag=True,
                                             name='diffusion_{}'.format(i))
                       for i, df in enumerate(p.diffusion_functions))

            L = LinearAffinelyDecomposedOperator(Li, L0, name='diffusion')
        else:
            L = DiffusionOperatorP1D2(grid, boundary_info, diffusion_function=p.diffusion_functions[0],
                                      name='diffusion')

        F = L2ProductFunctionalP1D2(grid, boundary_info, p.rhs, dirichlet_data=p.dirichlet_data)

        def visualize(U):
            pl.tripcolor(grid.centers(2)[:, 0], grid.centers(2)[:, 1], grid.subentities(0, 2), U)
            pl.colorbar()
            pl.show()

        discr = EllipticDiscretization(L, F, visualizer=visualize)

        return discr