# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause

import dataclasses
from dataclasses import dataclass, asdict
from typing import Type, Callable, Dict, List, Union

from .adapters import Driver, MatrixAdapter
from .html_formatter import HTMLTableFormatter, NotebookHTMLFormatter
from .latex_formatter import LatexFormatter, python_scientific_to_latex_times10


@dataclass
class MatReprParams:
    max_rows: int = 11
    """Maximum number of rows in HTML and Latex output."""

    max_cols: int = 15
    """Maximum number of columns in HTML and Latex output."""

    num_after_dots: Union[int, float] = 0.5
    """
    If a matrix has more rows or columns than allowed then an ellipsis (three dots) is emitted to cover the excess.
    This parameter controls how many rows/columns are drawn at the end of the matrix.
    
    For example, a value of 1 means the final row and final column are emitted in addition to the top-left corner.
    A value of 2 means the final two rows and columns are emitted, with a correspondingly smaller top-left corner.
    
    Note: May be ignored for very large matrices without fast row/column indexing.
    """

    cell_align: str = "center"
    """Horizontal text alignment for non-zero values."""

    indices: bool = True
    """Whether to show row/column indices."""

    title: Union[bool, str] = True
    """Title of table. If `True` then generate matrix description such as dimensions, nnz, datatype."""

    title_latex: bool = False
    """If not None then overrides `title` for Latex output only."""

    latex_matrix_env: str = "bmatrix"
    """Latex environment to use for matrices. For Jupyter this should be one supported by MathJax."""

    precision: int = 4
    """
    Floating-point precision. May be overriden by setting `float_formatter`.
    """

    float_formatter: Callable[[float], str] = None
    """
    A callable for converting floating point numbers to string.
    For convenience may also be a format string `fmt_str` and this will be done for you:
    `float_formatter = lambda f: format(f, fmt_str)`
    
    If None then formats using `precision`.
    """

    float_formatter_latex: Callable[[float], str] = None
    """
    Overwrites `float_formatter` for LaTeX output. If None then uses `float_formatter` but converts scientific
    notation from `1e22` to `1 \\times 10^{22}`.
    """

    def set_precision(self, precision, g=True):
        """
        Precision to use for floating-point to string conversion.
        """
        fmt_str = f".{precision}{'g' if g else ''}"
        self.float_formatter = lambda f: format(f, fmt_str)

    def _assert_one_of(self, var, choices):
        if getattr(self, var) not in choices:
            raise ValueError(f"{var} must be one of: " + ", ".join(choices))

    def get(self, **kwargs):
        ret = dataclasses.replace(self)

        # Allow some explicit overwrites for convenience
        if "title" in kwargs and "title_latex" not in kwargs:
            kwargs["title_latex"] = kwargs["title"]

        # Update all parameters with the ones in kwargs
        for key, value in kwargs.items():
            if hasattr(ret, key):
                setattr(ret, key, value)

        # Handy type conversions
        if ret.float_formatter is None:
            ret.set_precision(ret.precision)
        elif isinstance(ret.float_formatter, str):
            fmt_str = ret.float_formatter
            ret.float_formatter = lambda f: format(f, fmt_str)

        # validate
        self._assert_one_of("cell_align", ['center', 'left', 'right'])

        # Apply some default rules
        if ret.title_latex is None:
            ret.title_latex = ret.title

        if ret.float_formatter_latex is None:
            ret.float_formatter_latex = lambda f: python_scientific_to_latex_times10(ret.float_formatter(f))

        return ret

    def to_kwargs(self):
        return asdict(self)


params = MatReprParams()
_drivers: List[Type[Driver]] = []
_driver_map: Dict[str, Type[Driver]] = {}
_driver_registration_notify: List[Callable[[Type[Driver]], None]] = []


def register_driver(driver: Type[Driver]):
    _drivers.append(driver)

    for type_module, type_name, _ in driver.get_supported_types():
        _driver_map[".".join((type_module, type_name))] = driver

    for func in _driver_registration_notify:
        func(driver)


def _register_bundled():
    """
    Register the built-in drivers.
    """
    from .adapters.scipy_driver import SciPyDriver
    register_driver(SciPyDriver)

    from .adapters.list_like import ListDriver
    register_driver(ListDriver)

    from .adapters.graphblas_driver import GraphBLASDriver
    register_driver(GraphBLASDriver)


_register_bundled()


def _get_driver(mat):
    if isinstance(mat, list):
        type_str = "builtins.list"
    elif isinstance(mat, tuple):
        type_str = "builtins.tuple"
    else:
        type_str = ".".join((mat.__module__, mat.__class__.__name__))
    driver = _driver_map.get(type_str, None)
    if not driver:
        print("Supported types: \n" + "\n".join(sorted(list(_driver_map.keys()))))
        raise AttributeError("Unsupported type: " + type_str)
    return driver


def _get_adapter(mat) -> MatrixAdapter:
    adapter = _get_driver(mat).adapt(mat)
    if not adapter:
        raise AttributeError("Unsupported matrix")

    return adapter


def to_html(mat, notebook=False, **kwargs) -> str:
    options = params.get(**kwargs)
    adapter = _get_adapter(mat)

    if notebook:
        formatter = NotebookHTMLFormatter(**options.to_kwargs())
    else:
        formatter = HTMLTableFormatter(**options.to_kwargs())

    return str(formatter.format(adapter))


def to_latex(mat, **kwargs):
    options = params.get(**kwargs)
    adapter = _get_adapter(mat)

    formatter = LatexFormatter(**options.to_kwargs())

    return str(formatter.format(adapter))


def mdisplay(mat, method="html", **kwargs):
    from IPython.display import display, HTML, Latex

    if method == "html":
        display(HTML(to_html(mat, notebook=True, **kwargs)))
    elif method == "latex":
        display(Latex('$' + to_latex(mat, **kwargs) + '$'))
    else:
        raise ValueError("Unknown method: " + method)


def _register_jupyter_formatter(mime_type: str, repr_method: Callable):
    """
    See https://ipython.readthedocs.io/en/stable/config/integrating.html
    """
    # This import is unnecessary but makes static type checking work.
    # noinspection PyProtectedMember
    try:
        import IPython
    except ImportError:
        # no Jupyter
        return

    try:
        formatter = IPython.get_ipython().display_formatter.formatters[mime_type]
    except AttributeError:
        # not running in a notebook
        return

    for driver in _drivers:
        for type_module, type_name, register_with_jupyter in driver.get_supported_types():
            if register_with_jupyter:
                formatter.for_type_by_name(type_module, type_name, repr_method)


__all__ = ["to_html", "to_latex", "mdisplay"]
