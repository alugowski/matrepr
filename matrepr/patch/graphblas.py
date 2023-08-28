# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause
"""
Import this module to automatically print python-graphblas `Matrix` and `Vector` using MatRepr.

A `__repr__` method that calls MatRepr is inserted to those classes.

Also includes all public methods from the matrepr module for convenient one-line imports.
"""

from .. import *

import graphblas


def _str_(mat):
    from matrepr.adapters.graphblas_driver import GraphBLASDriver
    # Enable terminal width detection
    return to_str(GraphBLASDriver.adapt(mat), width_str=0, max_cols=9999)


graphblas.Matrix.__repr__ = _str_
graphblas.Vector.__repr__ = _str_
