# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import inspect

def method_arguments(func):
    args = inspect.getargspec(func)[0]
    try:
        args.remove('self')
    except ValueError:
        pass
    return args