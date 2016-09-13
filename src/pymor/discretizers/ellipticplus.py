# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
#
# Author: Julia Brunken (Extension of discretizers.elliptic)

from __future__ import absolute_import, division, print_function


from pymor.analyticalproblems.ellipticplus import EllipticPlusProblem
from pymor.discretizations.basic import StationaryDiscretization
from pymor.domaindiscretizers.default import discretize_domain_default
from pymor.grids.tria import TriaGrid
from pymor.grids.oned import OnedGrid
from pymor.gui.qt import PatchVisualizer, Matplotlib1DVisualizer
from pymor.operators.cg import DiffusionOperatorP1, L2ProductFunctionalP1, L2ProductP1WithoutBoundary, L2ProductP1Absorb, DiffusionOperatorP1WithoutBoundary
from pymor.operators.constructions import LincombOperator


def discretize_elliptic_cg_plus(analytical_problem, diameter=None, domain_discretizer=None,
                           grid=None, boundary_info=None):
    '''Discretizes an |EllipticPlusProblem| using finite elements.

    Parameters
    ----------
    analytical_problem
        The |EllipticPlusProblem| to discretize.
    diameter
        If not None, is passed to the domain_discretizer.
    domain_discretizer
        Discretizer to be used for discretizing the analytical domain. This has
        to be a function `domain_discretizer(domain_description, diameter, ...)`.
        If further arguments should be passed to the discretizer, use
        :func:`functools.partial`. If `None`, |discretize_domain_default| is used.
    grid
        Instead of using a domain discretizer, the |Grid| can also be passed directly
        using this parameter.
    boundary_info
        A |BoundaryInfo| specifying the boundary types of the grid boundary entities.
        Must be provided if `grid` is provided.

    Returns
    -------
    discretization
        The discretization that has been generated.
    data
        Dictionary with the following entries:

            :grid:           The generated |Grid|.
            :boundary_info:  The generated |BoundaryInfo|.

    Author: Julia Brunken (extension of discretizers.elliptic.discretize_elliptic_cg)
    '''

    assert isinstance(analytical_problem, EllipticPlusProblem)
    assert grid is None or boundary_info is not None
    assert boundary_info is None or grid is not None
    assert grid is None or domain_discretizer is None

    if grid is None:
        domain_discretizer = domain_discretizer or discretize_domain_default
        if diameter is None:
            grid, boundary_info = domain_discretizer(analytical_problem.domain)
        else:
            grid, boundary_info = domain_discretizer(analytical_problem.domain, diameter=diameter)

    assert isinstance(grid, (OnedGrid, TriaGrid))

    Operator = DiffusionOperatorP1
    Functional = L2ProductFunctionalP1
    AbsorbOperator = L2ProductP1Absorb
    p = analytical_problem

    if p.diffusion_functionals is not None or len(p.diffusion_functions) > 1:
        L0 = Operator(grid, boundary_info, diffusion_constant=0, name='diffusion_boundary_part')

        Li = [Operator(grid, boundary_info, diffusion_function=df, dirichlet_clear_diag=True,
                       name='diffusion_{}'.format(i))
              for i, df in enumerate(p.diffusion_functions)]

        L = LincombOperator(operators=[L0] + Li, coefficients=[1.] + list(p.diffusion_functionals),
                            name='diffusion')
    else:
        L = Operator(grid, boundary_info, diffusion_function=p.diffusion_functions[0],
                     name='diffusion')
    if p.absorb_functionals is not None or len(p.absorb_functions) > 1:
        Ai = [AbsorbOperator(grid, boundary_info, absorb_function=af, dirichlet_clear_diag=True,
                       name='absorb_{}'.format(i))
              for i, af in enumerate(p.absorb_functions)]

        A = LincombOperator(operators=Ai, coefficients=list(p.absorb_functionals),
                            name='absorb')
    else:
        A = AbsorbOperator(grid, boundary_info, absorb_function=p.absorb_functions[0],
                     name='absorb')

    L = LincombOperator(operators=[L] + [A], coefficients=[1, 1])

    F = Functional(grid, p.rhs, boundary_info, dirichlet_data=p.dirichlet_data)

    if isinstance(grid, TriaGrid):
        visualizer = PatchVisualizer(grid=grid, bounding_box=grid.domain, codim=2)
    else:
        visualizer = Matplotlib1DVisualizer(grid=grid, codim=1)

    products = {'h1': DiffusionOperatorP1WithoutBoundary(grid, boundary_info, diffusion_function=p.diffusion_functions[0]),
                'l2': L2ProductP1WithoutBoundary(grid, boundary_info),
                'absorb': AbsorbOperator(grid, boundary_info, absorb_function=p.absorb_functions[0])}

    parameter_space = p.parameter_space if hasattr(p, 'parameter_space') else None

    discretization = StationaryDiscretization(L, F, products=products, visualizer=visualizer,
                                              parameter_space=parameter_space, name='{}_CG'.format(p.name))

    return discretization, {'grid': grid, 'boundary_info': boundary_info}