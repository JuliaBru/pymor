# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

""" This module provides generic time-stepping algorithms for the solution of
instationary problems.

The algorithms are generic in the sense that each algorithms operates exclusively
on |Operators| and |VectorArrays|. In particular, the algorithms
can also be used to turn an arbitrary stationary |Discretization| provided
by an external library into an instationary |Discretization|.

Currently, implementations of :func:`explicit_euler` and :func:`implicit_euler`
time-stepping are provided. The :class:`TimeStepperInterface` defines a
common interface that has to be fulfilled by the time-steppers that are used
by |InstationaryDiscretization|. The classes :class:`ExplicitEulerTimeStepper`
and :class:`ImplicitEulerTimeStepper` encapsulate :func:`explicit_euler` and
:func:`implicit_euler` to provide this interface.
"""

from __future__ import absolute_import, division, print_function

from pymor.core.interfaces import ImmutableInterface, abstractmethod
from pymor.operators.interfaces import OperatorInterface
from pymor.vectorarrays.interfaces import VectorArrayInterface
from pymor.core.logger import getLogger
import numpy as np
from pymor.vectorarrays.numpy import NumpyVectorArray

class TimeStepperInterface(ImmutableInterface):
    """Interface for time-stepping algorithms.

    Algorithms implementing this interface solve time-dependent problems
    of the form ::

        M * d_t u + A(u, mu, t) = F(mu, t).

    Time-steppers used by |InstationaryDiscretization| have to fulfill
    this interface.
    """

    @abstractmethod
    def solve(self, initial_time, end_time, initial_data, operator, rhs=None, mass=None, mu=None, num_values=None):
        """Apply time-stepper to the equation ::

            M * d_t u + A(u, mu, t) = F(mu, t).

        Parameters
        ----------
        initial_time
            The time at which to begin time-stepping.
        end_time
            The time until which to perform time-stepping.
        initial_data
            The solution vector at `initial_time`.
        operator
            The |Operator| A.
        rhs
            The right hand side F (either |VectorArray| of length 1 or |Operator| with
            `range.dim == 1`). If `None`, zero right hand side is assumed.
        mass
            The |Operator| M. If `None`, the identity operator is assumed.
        mu
            |Parameter| for which `operator` and `rhs` are evaluated. The current time is added
            to `mu` with key `_t`.
        num_values
            The number of returned vectors of the solution trajectory. If `None`, each
            intermediate vector that is calculated is returned.

        Returns
        -------
        |VectorArray| containing the solution trajectory.
        """
        pass


class ImplicitEulerTimeStepper(TimeStepperInterface):
    """Implict-Euler time-stepper.

    Solves equations of the form ::

        M * d_t u + A(u, mu, t) = F(mu, t).

    Parameters
    ----------
    nt
        The number of time-steps the time-stepper will perform.
    solver_options
        The |solver_options| used to invert `M + dt*A`.
        The special values `'mass'` and '`operator'` are
        recognized, in which case the solver_options of
        M (resp. A) are used.
    """

    def __init__(self, nt, solver_options='operator'):
        self.nt = nt
        self.solver_options = solver_options

    def solve(self, initial_time, end_time, initial_data, operator, rhs=None, mass=None, mu=None, num_values=None):
        return implicit_euler(operator, rhs, mass, initial_data, initial_time, end_time, self.nt, mu, num_values,
                              solver_options=self.solver_options)


class ExplicitEulerTimeStepper(TimeStepperInterface):
    """Implict-Euler time-stepper.

    Solves equations of the form ::

        M * d_t u + A(u, mu, t) = F(mu, t).

    Parameters
    ----------
    nt
        The number of time-steps the time-stepper will perform.
    """

    def __init__(self, nt):
        self.nt = nt

    def solve(self, initial_time, end_time, initial_data, operator, rhs=None, mass=None, mu=None, num_values=None):
        if mass is not None:
            raise NotImplementedError
        return explicit_euler(operator, rhs, initial_data, initial_time, end_time, self.nt, mu, num_values)


class ExplicitEulerTimeStepperNDim(TimeStepperInterface):
    '''Explict-Euler time-stepper.

    Solves equations of the form ::

        M * d_t u + A(u, t) = F(t).

    Parameters
    ----------
    nt
        The number of time-steps the time-stepper will perform.

    Author: Julia Brunken (Extension of ExplicitEulerTimeStepperNDim)
    '''

    def __init__(self, nt):
        self.nt = nt

    def solve(self, sysdim, initial_time, end_time, initial_data, operator, rhs=None, mass=None, mu=None, num_values=None):
        if mass is not None:
            raise NotImplementedError
        return explicit_euler_ndim(sysdim, operator, rhs, initial_data, initial_time, end_time, self.nt, mu, num_values)


def implicit_euler(A, F, M, U0, t0, t1, nt, mu=None, num_values=None, solver_options='operator'):
    assert isinstance(A, OperatorInterface)
    assert isinstance(F, (type(None), OperatorInterface, VectorArrayInterface))
    assert isinstance(M, (type(None), OperatorInterface))
    assert A.source == A.range
    num_values = num_values or nt + 1
    dt = (t1 - t0) / nt
    DT = (t1 - t0) / (num_values - 1)


    if F is None:
        F_time_dep = False
    elif isinstance(F, OperatorInterface):
        assert F.range.dim == 1
        assert F.source == A.range
        F_time_dep = F.parametric and '_t' in F.parameter_type
        if not F_time_dep:
            dt_F = F.as_vector(mu) * dt
    else:
        assert len(F) == 1
        assert F in A.range
        F_time_dep = False
        dt_F = F * dt

    if M is None:
        from pymor.operators.constructions import IdentityOperator
        M = IdentityOperator(A.source)

    assert A.source == M.source == M.range
    assert not M.parametric
    assert U0 in A.source
    assert len(U0) == 1

    A_time_dep = A.parametric and '_t' in A.parameter_type

    R = A.source.empty(reserve=nt+1)
    R.append(U0)

    options = A.solver_options if solver_options == 'operator' else \
              M.solver_options if solver_options == 'mass' else \
              solver_options
    M_dt_A = (M + A * dt).with_(solver_options=options)
    if not A_time_dep:
        M_dt_A = M_dt_A.assemble(mu)

    t = t0
    U = U0.copy()

    for n in xrange(nt):
        t += dt
        mu['_t'] = t
        rhs = M.apply(U)
        if F_time_dep:
            dt_F = F.as_vector(mu) * dt
        if F:
            rhs += dt_F
        U = M_dt_A.apply_inverse(rhs, mu=mu)
        while t - t0 + (min(dt, DT) * 0.5) >= len(R) * DT:
            R.append(U)

    return R


def explicit_euler(A, F, U0, t0, t1, nt, mu=None, num_values=None):
    assert isinstance(A, OperatorInterface)
    assert F is None or isinstance(F, (OperatorInterface, VectorArrayInterface))
    assert A.source == A.range
    num_values = num_values or nt + 1

    if isinstance(F, OperatorInterface):
        assert F.range.dim == 1
        assert F.source == A.source
        F_time_dep = F.parametric and '_t' in F.parameter_type
        if not F_time_dep:
            F_ass = F.as_vector(mu)
    elif isinstance(F, VectorArrayInterface):
        assert len(F) == 1
        assert F in A.source
        F_time_dep = False
        F_ass = F

    assert len(U0) == 1
    assert U0 in A.source

    A_time_dep = A.parametric and '_t' in A.parameter_type
    if not A_time_dep:
        A = A.assemble(mu)

    dt = (t1 - t0) / nt
    DT = (t1 - t0) / (num_values - 1)
    R = A.source.empty(reserve=num_values)
    R.append(U0)

    t = t0
    U = U0.copy()

    if F is None:
        for n in xrange(nt):
            t += dt
            mu['_t'] = t
            U.axpy(-dt, A.apply(U, mu=mu))
            while t - t0 + (min(dt, DT) * 0.5) >= len(R) * DT:
                R.append(U)
    else:
        for n in xrange(nt):
            t += dt
            mu['_t'] = t
            if F_time_dep:
                F_ass = F.as_vector(mu)
            U.axpy(dt, F_ass - A.apply(U, mu=mu))
            while t - t0 + (min(dt, DT) * 0.5) >= len(R) * DT:
                R.append(U)

    return R


def explicit_euler_ndim(sysdim, A, F, U0, t0, t1, nt, mu=None, num_values=None):
    logger = getLogger('pymor.algorithms.timestepping.explicit_euler_ndim')
    assert isinstance(A, OperatorInterface)

    assert A.source.dim*sysdim == A.range.dim
    num_values = num_values or nt + 1
    proz = 0

    F_ass = dict.fromkeys(range(sysdim))

    assert F is None or isinstance(F, OperatorInterface)
    if isinstance(F, OperatorInterface):
        assert F.range.dim == 1
        assert F.source == A.source
        F_time_dep = F.parametric and '_t' in F.parameter_type
        if not F_time_dep:
            for j in range(sysdim):
                mu['komp'] = j
                F_ass[j] = F.as_vector(mu)

    for j in range(sysdim):
        assert isinstance(U0[j], VectorArrayInterface)
        assert len(U0[j]) == 1
        assert U0[j] in A.source

    A_time_dep = A.parametric and '_t' in A.parameter_type
    if hasattr(A, 'assemble') and not A_time_dep:
        A = A.assemble(mu)

    dt = (t1 - t0) / nt
    DT = (t1 - t0) / (num_values - 1)

    R = dict.fromkeys(range(sysdim))
    for j in range(sysdim):
        R[j] = A.source.empty(reserve=num_values)
        R[j].append(U0[j])

    t = t0
    tvec = np.array(t0)
    U = dict.fromkeys(range(sysdim))
    for j in range(sysdim):
        U[j] = NumpyVectorArray(U0[j].copy())

    while t - t0 + (min(dt, DT) * 0.5) >= len(R[0]) * DT: 
        tvec = np.append(tvec, t)
        for j in range(sysdim):
            R[j].append(U[j])

    if F is None:
        for n in xrange(nt):
            t += dt
            if n/nt > proz:
                logger.info('{} %'.format(proz*100))
                proz += 0.1
            mu['_t'] = t
            Ua = A.apply(U.copy(), mu=mu)
            if n * (num_values / nt) >= len(R[0]):
                tvec = np.append(tvec, t)

            for j in range(sysdim):
                if np.max(U[j].data) > 100:
                    raise ValueError
                U[j].axpy(dt, -Ua[j])

            while t - t0 + (min(dt, DT) * 0.5) >= len(R[0]) * DT:
                tvec = np.append(tvec, t)
                for j in range(sysdim):
                    R[j].append(U[j])

    else:
        for n in xrange(nt):
            t += dt
            if n/nt > proz:
                logger.info('{} %'.format(proz*100))
                proz += 0.1
            mu['_t'] = t
            if F_time_dep:
                for j in range(sysdim):
                    mu['komp'] = j
                    F_ass[j] = F.as_vector(mu)

            Ua = A.apply(U.copy(), mu=mu)

            for j in range(sysdim):
                U[j].axpy(dt, F_ass[j] - Ua[j])
                if np.max(U[j].data) > 100:
                    raise ValueError

            while t - t0 + (min(dt, DT) * 0.5) >= len(R[0]) * DT:
                tvec = np.append(tvec, t)
                for j in range(sysdim):
                    R[j].append(U[j])

    return R, tvec
