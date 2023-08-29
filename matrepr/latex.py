# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause

"""
Jupyter extension that registers a LaTeX formatter for supported matrix types.

Usage:
%load_ext matrepr.latex
"""
from . import to_latex, _register_jupyter_formatter


def _repr_latex_(mat):
    return to_latex(mat)


def load_ipython_extension(ipython):
    _register_jupyter_formatter(ipython, "text/latex", _repr_latex_)


def unload_ipython_extension(ipython):
    _register_jupyter_formatter(ipython, "text/latex", None)
