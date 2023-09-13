# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause

import dataclasses
from dataclasses import dataclass, asdict
from shutil import get_terminal_size
from typing import Any, Type, Callable, Dict, List, Union, Mapping, Optional

from .adapters import Driver, MatrixAdapter, FallbackToNative
from .html_formatter import HTMLTableFormatter, NotebookHTMLFormatter
from .latex_formatter import LatexFormatter, python_scientific_to_latex_times10, tex_escape
from matrepr.html import load_ipython_extension, unload_ipython_extension


AUTO = "auto"


@dataclass
class MatReprParams:
    max_rows: int = 11
    """
    Maximum number of rows in output.
    Set to None for unlimited.
    """

    max_cols: int = 15
    """
    Maximum number of columns in output.
    Set to None for unlimited.
    """

    width_str: int = None
    """
    Maximum width of text output.
     - If 0 then autodetect terminal size.
     - If None then use max_cols.

    Some methods that print to the terminal may automatically set this value.
    """

    num_after_dots: Union[int, float] = 0.5
    """
    If a matrix has more rows or columns than allowed then an ellipsis (three dots) is emitted to cover the excess.
    This parameter controls how many rows/columns are drawn at the end of the matrix.
    
    For example, a value of 1 means the final row and final column are emitted in addition to the top-left corner.
    A value of 2 means the final two rows and columns are emitted, with a correspondingly smaller top-left corner.
    
    A fractional value between 0 and 1 means a fraction of `max_rows` and `max_cols`.
    
    Note: May be set to 0 for very large matrices without fast row/column indexing.
    """

    fill_value: Any = None
    """
    Represents the value to print when a sparse matrix does not specify a value (i.e. implicit zero). Setting a
    `fill_value` will emit a dense output.
    """

    cell_align: str = "center"
    """Horizontal text alignment for non-zero values."""

    indices: bool = True
    """Whether to show row/column indices."""

    title: Union[bool, str] = True
    """Title of table. If `True` then generate matrix description such as dimensions, nnz, datatype."""

    title_latex: bool = False
    """If not None then overrides `title` for Latex output only."""

    latex_matrix_env: str = "array"
    """
    Latex environment to use for matrices. For Jupyter this should be one supported by MathJax.
    Cell alignment (left/right/center) only supported by 'array' environment.
    """

    latex_tensor_env: str = "array"
    """
    Latex environment to use for tensors rendered as a 2D matrix.
    For Jupyter this should be one supported by MathJax.
    Cell alignment (left/right/center) only supported by 'array' environment.
    """

    latex_dupe_matrix_env: str = "Bmatrix"
    """Latex environment to use when a cell has multiple elements."""

    precision: int = 4
    """
    Floating-point precision. May be overriden by `floatfmt`.
    """

    floatfmt: Union[str, Callable[[float], str]] = None
    """
    A format string to format numbers as string, usable by :code:`format()`.

    May also be a `Callable` that converts numbers to strings.
    
    If None then `precision` is used.
    """

    floatfmt_latex: Callable[[float], str] = None
    """
    Overwrites `floatfmt` for LaTeX output. If None then uses `floatfmt` but converts scientific
    notation: :raw:`1e22` becomes :raw:`1 \\times 10^{22}`
    """

    row_labels: Union[Mapping[int, Any], bool] = AUTO
    """Labels for matrix rows. Default uses row index, False disables row labels."""

    col_labels: Union[Mapping[int, Any], bool] = AUTO
    """Labels for matrix columns. Default uses column index, False disables column labels."""

    def set_precision(self, precision, g=True):
        """
        Precision to use for floating-point to string conversion.

        :param precision: floating-point precision
        :param g: Whether to use :code:`g` or :code:`f` formatting.
        """
        fmt_str = f".{precision}{'g' if g else ''}"
        self.floatfmt = lambda f: format(f, fmt_str)

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
        if ret.floatfmt is None:
            ret.set_precision(ret.precision)
        elif isinstance(ret.floatfmt, str):
            fmt_str = ret.floatfmt
            ret.floatfmt_str = fmt_str
            ret.floatfmt = lambda f: format(f, fmt_str)

        # Apply some default rules
        if ret.max_rows is None:
            ret.max_rows = 999_999_999
        if ret.max_cols is None:
            ret.max_cols = 999_999_999
        if ret.title_latex is None:
            ret.title_latex = ret.title

        if ret.floatfmt_latex is None:
            ret.floatfmt_latex = lambda f: python_scientific_to_latex_times10(ret.floatfmt(f))

        # validate
        ret._assert_one_of("cell_align", ['center', 'left', 'right'])
        ret.max_rows = max(1, ret.max_rows)
        ret.max_cols = max(1, ret.max_cols)

        # compute automatic values
        if kwargs.get("auto_width_str", False):
            if "max_cols" not in kwargs and ret.width_str is None:
                # user did not explicitly specify a width
                ret.width_str = 0
                ret.max_cols = 999

        if ret.width_str == 0:
            width_str, _ = get_terminal_size()
            ret.width_str = width_str
        return ret

    def to_kwargs(self):
        return asdict(self)


params = MatReprParams()
"""
Changeable default parameters that apply unless overridden in a method call.
"""
_drivers: List[Type[Driver]] = []
_driver_map: Dict[str, Type[Driver]] = {}


def register_driver(driver: Type[Driver]):
    _drivers.append(driver)

    for type_str, _ in driver.get_supported_types():
        _driver_map[type_str] = driver


def _register_bundled():
    """
    Register the built-in drivers.
    """
    from .adapters.scipy_driver import SciPyDriver
    register_driver(SciPyDriver)

    from .adapters.graphblas_driver import GraphBLASDriver
    register_driver(GraphBLASDriver)

    from .adapters.sparse_driver import PyDataSparseDriver
    register_driver(PyDataSparseDriver)

    from .adapters.numpy_driver import NumpyDriver
    register_driver(NumpyDriver)

    from .adapters.torch_driver import PyTorchDriver
    register_driver(PyTorchDriver)

    from .adapters.tensorflow_driver import TensorFlowDriver
    register_driver(TensorFlowDriver)

    from .adapters.list_like import ListDriver
    register_driver(ListDriver)


_register_bundled()


def _get_driver(mat, unsupported_raise=True):
    type_str = ".".join((type(mat).__module__, type(mat).__name__))
    driver = _driver_map.get(type_str, None)
    if not driver and unsupported_raise:
        raise AttributeError("Unsupported type: " + type_str)
    return driver


def _get_adapter(mat: Any, options: Optional[MatReprParams], unsupported_raise=True) -> MatrixAdapter:
    if isinstance(mat, MatrixAdapter):
        adapter = mat
    else:
        driver = _get_driver(mat, unsupported_raise=unsupported_raise)
        adapter = driver.adapt(mat) if driver else None
        if not adapter and unsupported_raise:
            raise AttributeError("Unsupported matrix")

    if options:
        if options.row_labels is not AUTO and options.row_labels is not True:
            adapter.row_labels = options.row_labels
        if options.col_labels is not AUTO and options.col_labels is not True:
            adapter.col_labels = options.col_labels
    return adapter


def to_html(mat: Any, notebook=False, fallback_to_none=False, **kwargs):
    """
    Render a matrix to HTML.

    :param mat: A supported matrix.
    :param notebook: If true then adds extra styling appropriate for display in a Jupyter notebook.
    :param fallback_to_none: Whether to return None or a fallback LaTeX string
         if the formatter asks to fall back to native formatting.
    :param kwargs: Any argument in :class:`MatReprParams`.
    :rtype: str
    :return: A string containing an HTML representation of `mat`.
    """
    options = params.get(**kwargs)

    try:
        adapter = _get_adapter(mat, options)

        if notebook:
            formatter = NotebookHTMLFormatter(**options.to_kwargs())
        else:
            formatter = HTMLTableFormatter(**options.to_kwargs())

        return str(formatter.format(adapter))
    except FallbackToNative:
        if hasattr(mat, "_repr_html_"):
            return getattr(mat, "_repr_html_")()
        elif fallback_to_none:
            return None
        else:
            return f"<pre>{str(mat)}</pre>"


def to_latex(mat: Any, fallback_to_none=False, **kwargs):
    """
    Render a matrix to LaTeX.

    :param mat: A supported matrix.
    :param fallback_to_none: Whether to return None or a fallback LaTeX string
         if the formatter asks to fall back to native formatting.
    :param kwargs: Any argument in :class:`MatReprParams`.
    :rtype: str
    :return: A string containing a LaTeX representation of `mat`.
    """
    options = params.get(**kwargs)

    try:
        adapter = _get_adapter(mat, options)
        formatter = LatexFormatter(**options.to_kwargs())
        return str(formatter.format(adapter))
    except FallbackToNative:
        if hasattr(mat, "_repr_latex_"):
            return getattr(mat, "_repr_latex_")()
        elif fallback_to_none:
            return None
        else:
            return r"\texttt{" + tex_escape(str(mat)) + r"}"


def to_str(mat: Any, **kwargs) -> str:
    """
    Render a matrix to a string.

    Internally uses `tabulate` and will pass through any arguments not explicitly set by this method.

    :param mat: A supported matrix.
    :param kwargs: Any argument in :class:`MatReprParams`.
    :return: a string representation of `mat`.
    """
    options = params.get(**kwargs)

    try:
        ret = []

        if options.title:
            if options.title is True:
                adapter = _get_adapter(mat, None)
                title = f"<{adapter.describe()}>"
            else:
                title = options.title
            ret.append(title)

        from .string_formatter import to_tabulate
        ret.append(to_tabulate(mat, **kwargs))
        return "\n".join(ret)
    except FallbackToNative:
        return str(mat)


def mprint(mat: Any, **kwargs):
    """
    Prints the output of :func:`to_str`.

    :param mat: A supported matrix.
    :param kwargs: Any argument in :class:`MatReprParams`.
    """
    print(to_str(mat, auto_width_str=True, **kwargs))


def mdisplay(mat: Any, method="html", **kwargs):
    """
    Display a matrix in Jupyter.

    :param mat: A supported matrix.
    :param method: Style to use. One of :code:`"html"`, :code:`"latex"`, :code:`"str"`.
    :param kwargs: Any argument in :class:`MatReprParams`.
    """
    from IPython.display import display, HTML, Latex, Pretty

    if method == "html":
        display(HTML(to_html(mat, notebook=True, **kwargs)))
    elif method == "latex":
        display(Latex('$' + to_latex(mat, **kwargs) + '$'))
    elif method == "str":
        display(Pretty(to_str(mat, auto_width_str=True, **kwargs)))
    else:
        raise ValueError("Unknown method: " + method)


def _register_jupyter_formatter(ipython, mime_type: str, repr_method: Optional[Callable]):
    """
    See https://ipython.readthedocs.io/en/stable/config/integrating.html
    """
    if not ipython:
        return

    formatter = ipython.get_ipython().display_formatter.formatters[mime_type]

    for driver in _drivers:
        for type_str, register_with_jupyter in driver.get_supported_types():
            if register_with_jupyter:
                if repr_method:
                    formatter.for_type(type_str, repr_method)
                else:
                    formatter.pop(type_str)


__all__ = ["to_html", "to_latex", "to_str", "mprint", "mdisplay"]
