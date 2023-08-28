# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause

from typing import Any, Dict, Tuple, Optional
from tabulate import tabulate, TableFormat, Line, DataRow

from . import params, _get_adapter
from .adapters import to_trunc
from .list_converter import ListConverter

row_vector_table_format = TableFormat(
    lineabove=None,
    linebelowheader=None,
    linebetweenrows=None,
    linebelow=None,
    headerrow=DataRow(" ", "  ", " "),
    datarow=DataRow("[", "  ", "]"),
    padding=0,
    with_header_hide=None,
)
"""
Renders a table that looks like a horizontal list, supporting a column index header.
"""

row_vector_comma_table_format = TableFormat(
    lineabove=None,
    linebelowheader=None,
    linebetweenrows=None,
    linebelow=None,
    headerrow=DataRow(" ", "  ", " "),
    datarow=DataRow("[", ", ", "]"),
    padding=0,
    with_header_hide=None,
)
"""
Renders a table that looks like a horizontal list, using comma separators instead of a header.
"""


matrix_table_format = TableFormat(
    lineabove=Line("┌", " ", " ", "┐"),
    linebelowheader=Line("┌", " ", " ", "┐"),
    linebetweenrows=None,
    linebelow=Line("└", " ", " ", "┘"),
    headerrow=DataRow(" ", " ", " "),
    datarow=DataRow("│", " ", "│"),
    padding=1,
    with_header_hide=["lineabove"],
)
"""
Renders table that looks like a LaTeX bmatrix.

Renders column indices correctly, but not row indices.
"""

matrix_format_patched_indices = TableFormat(
    lineabove=None,
    linebelowheader=Line("", " ", "┌", "┐"),
    linebetweenrows=None,
    linebelow=Line("", " ", "└", "┘"),
    headerrow=DataRow("", " ", ""),
    datarow=DataRow("", "│", "│"),
    padding=1,
    with_header_hide=None,
)
"""
A corrupted version of :ref:`matrix_table_format` that, when processed using :func:`_tabulate_sep_to_border`,
yields the same beautiful bmatrix but with indices nicely outside the brackets.
"""

tensor_table_format = TableFormat(
    lineabove=None,
    linebelowheader=None,
    linebetweenrows=None,
    linebelow=None,
    headerrow=DataRow(" ", "  ", " "),
    datarow=DataRow("(", ", ", ")"),
    padding=0,
    with_header_hide=None,
)
"""
Renders a table that looks like a horizontal tuple, using comma separators.
"""


tensor_format_patched_indices = TableFormat(
    lineabove=None,
    linebelowheader=None,
    linebetweenrows=None,
    linebelow=None,
    headerrow=DataRow("", " ", ""),
    datarow=DataRow("", "(", ")"),
    padding=1,
    with_header_hide=None,
)
"""
Renders a table that looks like a horizontal tuple, using comma separators.
"""


def _tabulate_sep_to_border(s: str, tf: TableFormat) -> str:
    lines = s.split("\n")

    # Must correct below-header row, data rows, bottom row.
    for row in range(1, len(lines)):
        line = lines[row]
        save_inds = []

        if tf.linebelowheader is not None and tf.linebelow is not None:
            if row == 1:
                # below-header row
                needle = tf.linebelowheader.sep
            elif row < len(lines) - 1:
                # data row uses | separator which appears at the very end too
                needle = tf.datarow.sep
                save_inds.append(-1)
            else:
                # bottom row
                needle = tf.linebelow.sep
        else:
            needle = tf.datarow.sep

        # The leftmost separator is actually the matrix border, so save that one
        save_inds.append(line.find(needle))

        # convert all but the one or two saved copies of the separator to spaces
        line_chars = list(line.replace(needle, " "))
        for i in save_inds:
            line_chars[i] = needle

        lines[row] = "".join(line_chars)
    return "\n".join(lines)


def _labels_to_str(labels):
    for label in labels:
        yield "" if label is None else str(label)


def max_line_width(s: str):
    return max(len(line) for line in s.split("\n"))


def _to_tabulate_args(mat: Any, is_tensor=False, **kwargs) -> Tuple[Dict, Optional[TableFormat]]:
    """
    Creates arguments for :func:`~tabulate.tabulate` to format the matrix.

    Note that tabulate is not expressive enough to render a matrix with nice row indices. Tabulate does not distinguish
    between the index column and data columns. Therefore, index output renders a corrupted output that is then patched
    using :func:`_tabulate_sep_to_border`.

    :param mat: A supported matrix.
    :param kwargs: Any argument in :class:`MatReprParams`, or any argument supported by :func:`~tabulate.tabulate`.
    :return: the argument dictionary and an object. If the object is not None then :func:`_tabulate_sep_to_border`
             must be called with the tabulate output and this object.
    """
    options = params.get(**kwargs)
    adapter = _get_adapter(mat, options)

    # collect the data
    conv = ListConverter(**options.to_kwargs())
    data, row_labels, col_labels = conv.to_lists_and_labels(adapter, is_1d_ok=False)

    # determine the table format
    patch_required = None
    if "tablefmt" in kwargs:
        matrix_format = kwargs["tablefmt"]
    elif not row_labels:
        # horizontal vector
        matrix_format = row_vector_table_format if options.indices else row_vector_comma_table_format
    elif options.indices and row_labels:
        # matrix with row and column indices
        if is_tensor:
            matrix_format = tensor_format_patched_indices
            patch_required = tensor_format_patched_indices
        else:
            matrix_format = matrix_format_patched_indices
            patch_required = matrix_format_patched_indices
    else:
        # matrix without row indices (column indices ok)
        if is_tensor:
            matrix_format = tensor_table_format
        else:
            matrix_format = matrix_table_format

    # collect floatfmt argument
    if "floatfmt" in kwargs and isinstance(kwargs["floatfmt"], str):
        floatfmt = kwargs["floatfmt"]
    elif isinstance(options.floatfmt, str):
        floatfmt = options.floatfmt
    else:
        floatfmt = f".{options.precision}g"

    # base arguments
    tab_args = {
        "tabular_data": data,
        "headers": _labels_to_str(col_labels) if options.indices else (),
        "showindex": _labels_to_str(row_labels) if options.indices and row_labels else False,
        "tablefmt": matrix_format,
        "floatfmt": floatfmt,
    }

    # colalign
    if "colalign" not in kwargs and options.indices:
        tab_args["colalign"] = (["right"] if tab_args["showindex"] else []) + ["center"] * len(col_labels)

    # forward any remaining arguments
    for arg in ["intfmt", "numalign", "stralign", "missingval", "disable_numparse",
                "colalign", "maxcolwidths", "rowalign", "maxheadercolwidths"]:
        if arg in kwargs:
            tab_args[arg] = kwargs[arg]

    return tab_args, patch_required


def to_tabulate(mat: Any, **kwargs) -> str:
    """
    Format a matrix to string using :func:`~tabulate.tabulate`.

    Supports argument passthrough.

    :param mat: A supported matrix.
    :param kwargs: Any argument in :class:`MatReprParams`, or any argument supported by :func:`~tabulate.tabulate`.
    :return: output of :func:`~tabulate.tabulate`
    """
    options = params.get(**kwargs)
    adapter = _get_adapter(mat, options)

    if options.width_str:
        txt_max_cols = max(1, int(options.width_str / 3))  # minimum 3 chars per column (if empty)
        cols = min(options.max_cols, txt_max_cols)
        trunc = to_trunc(adapter, options.max_rows, cols, options.num_after_dots)

        args, patch_tf = _to_tabulate_args(trunc, is_tensor=adapter.is_tensor(), **kwargs)
        ret = tabulate(**args)
        ret_width = max_line_width(ret)

        while ret_width > options.width_str and cols > 1:
            # too wide
            cols = trunc.drop_column()
            args, patch_tf = _to_tabulate_args(trunc, is_tensor=adapter.is_tensor(), **kwargs)
            ret = tabulate(**args)
            ret_width = max_line_width(ret)

    else:
        args, patch_tf = _to_tabulate_args(adapter, is_tensor=adapter.is_tensor(), **kwargs)
        ret = tabulate(**args)

    if patch_tf:
        ret = _tabulate_sep_to_border(ret, patch_tf)
    return ret
