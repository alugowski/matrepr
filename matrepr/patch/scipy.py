# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause
"""
Import this module to automatically print SciPy sparse matrices using MatRepr.

A `__repr__` method that calls MatRepr is inserted to all SciPy sparse matrix classes.

Also includes all public methods from the matrepr module for convenient one-line imports.
"""

from .. import *

import scipy.sparse


def _str_(mat):
    from matrepr.adapters.scipy_driver import SciPyDriver
    # Enable terminal width detection
    return to_str(SciPyDriver.adapt(mat), width_str=0, max_cols=9999)


try:
    scipy.sparse.bsr_matrix.__repr__ = _str_
    scipy.sparse.coo_matrix.__repr__ = _str_
    scipy.sparse.csc_matrix.__repr__ = _str_
    scipy.sparse.csr_matrix.__repr__ = _str_
    scipy.sparse.dia_matrix.__repr__ = _str_
    scipy.sparse.dok_matrix.__repr__ = _str_
    scipy.sparse.lil_matrix.__repr__ = _str_
except AttributeError:
    pass

try:
    scipy.sparse.bsr_array.__repr__ = _str_
    scipy.sparse.coo_array.__repr__ = _str_
    scipy.sparse.csc_array.__repr__ = _str_
    scipy.sparse.csr_array.__repr__ = _str_
    scipy.sparse.dia_array.__repr__ = _str_
    scipy.sparse.dok_array.__repr__ = _str_
    scipy.sparse.lil_array.__repr__ = _str_
except AttributeError:
    pass
