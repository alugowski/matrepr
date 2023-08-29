# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause

"""
Jupyter extension that registers an HTML formatter for supported matrix types.

Usage:
%load_ext matrepr.html
"""


def _repr_html_(mat):
    from . import to_html
    return to_html(mat, notebook=True)


def load_ipython_extension(ipython):
    from . import _register_jupyter_formatter
    _register_jupyter_formatter(ipython, "text/html", _repr_html_)


def unload_ipython_extension(ipython):
    from . import _register_jupyter_formatter
    _register_jupyter_formatter(ipython, "text/html", None)
