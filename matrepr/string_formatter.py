# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause

from typing import Any, Dict, Iterable, Tuple, Optional
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
    lineabove=Line("┌", " ", "", "┐"),
    linebelowheader=Line("┌", " ", "", "┐"),
    linebetweenrows=None,
    linebelow=Line("└", " ", "", "┘"),
    headerrow=DataRow(" ", "", " "),
    datarow=DataRow("│", "", "│"),
    padding=1,
    with_header_hide=["lineabove"],
)
"""
Renders table that looks like a LaTeX bmatrix.

Supports column indices, but not row indices.
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


def _labels_to_str(labels):
    for label in labels:
        yield "" if label is None else str(label)


def max_line_width(s: str):
    return max(len(line) for line in s.split("\n"))


def _has_line(tf: TableFormat, line, has_header):
    if getattr(tf, line) is None:
        return False

    if not has_header:
        return line not in ["headerrow", "linebelowheader"]
    else:
        return tf.with_header_hide is None or line not in tf.with_header_hide


def _format_row_labels(row_labels: Optional[Iterable], has_header: bool, tf: TableFormat):
    if isinstance(tf, str):
        # find this format in tabulate
        import tabulate as tab
        try:
            # noinspection PyProtectedMember,PyUnresolvedReferences
            tf = tab._table_formats.get(tf, None)

            if tf.lineabove is not None and not isinstance(tf.lineabove, Line) or \
               tf.linebelowheader is not None and not isinstance(tf.linebelowheader, Line) or \
               tf.linebetweenrows is not None and not isinstance(tf.linebetweenrows, Line) or \
               tf.linebelow is not None and not isinstance(tf.linebelow, Line) or \
               tf.datarow is not None and hasattr(tf.datarow, "__call__"):
                # only support regular tables
                tf = None
        except AttributeError:
            tf = None
        if tf is None:
            return None

    lines = []

    # add a space between label and data
    sep = " "

    if _has_line(tf, "lineabove", has_header):
        lines.append("")

    if has_header:
        lines.append("")

    if _has_line(tf, "linebelowheader", has_header):
        lines.append("")

    for i, label in enumerate(row_labels):
        if i != 0 and _has_line(tf, "linebetweenrows", has_header):
            lines.append("")
        lines.append(f"{str(label)}{sep}")

    if _has_line(tf, "linebelow", has_header):
        lines.append("")

    # Ensure all lines have the same width
    max_len = max(len(line) for line in lines)

    fmt = "{0: >" + str(max_len) + "}"

    for i in range(len(lines)):
        lines[i] = fmt.format(lines[i])

    return "\n".join(lines)


def multiline_concat(row_labels: str, matrix: str) -> str:
    if row_labels is None:
        return matrix

    row_labels = row_labels.split("\n")
    matrix = matrix.split("\n")
    assert len(row_labels) == len(matrix)

    ret = [row_labels[i] + matrix[i] for i in range(len(row_labels))]
    return "\n".join(ret)


def extract_forwardable_tabulate_args(**kwargs):
    ret = {
        k: v for k, v in kwargs.items() if k in [
            "tablefmt", "intfmt", "numalign", "stralign", "missingval", "disable_numparse",
            "colalign", "maxcolwidths", "rowalign", "maxheadercolwidths"
        ]
    }
    return ret


def _to_tabulate_args(mat: Any, is_tensor, options, tab_extra_args) -> Tuple[Dict, Optional[str]]:
    """
    Creates arguments for :func:`~tabulate.tabulate` to format the matrix.

    Note that tabulate is not expressive enough to render a matrix with nice row indices. Tabulate does not distinguish
    between the index column and data columns. Therefore, if applicable, row indices are generated separately and
    must be prepended to the tabulate result using  :func:`multiline_concat`.

    :param mat: A supported matrix.
    :param is_tensor: whether to use tensor or matrix style
    :param options: pre-parsed :class:`MatReprParams`
    :param tab_extra_args: any additional arguments to pass to :func:`~tabulate.tabulate`.
    :return: the argument dictionary and an optional string. If the string is not None use :func:`multiline_concat`
             to prepend it to tabulate output.
    """
    adapter = _get_adapter(mat, options)

    # collect the data
    conv = ListConverter(**options.to_kwargs())
    data, row_labels, col_labels = conv.to_lists_and_labels(adapter, is_1d_ok=False)

    # determine the table format
    if "tablefmt" in tab_extra_args:
        matrix_format = tab_extra_args["tablefmt"]
    elif adapter.get_shape()[0] == 1 and not adapter.has_row_labels():
        # horizontal vector
        matrix_format = row_vector_table_format \
            if options.indices and adapter.has_col_labels() else row_vector_comma_table_format
    else:
        # matrix
        if is_tensor:
            matrix_format = tensor_table_format
        else:
            matrix_format = matrix_table_format

    # render the row indices
    row_labels_txt = None
    has_header = options.indices and col_labels
    if options.indices and adapter.has_row_labels():
        row_labels_txt = _format_row_labels(_labels_to_str(row_labels), has_header=has_header, tf=matrix_format)

    # collect floatfmt argument
    disable_numparse = tab_extra_args.pop("disable_numparse", False)
    if hasattr(options, "floatfmt_str"):
        floatfmt = options.floatfmt_str
    elif isinstance(options.floatfmt, str):
        floatfmt = options.floatfmt
    else:
        disable_numparse = True  # the numbers are already formatted; leave them as is
        floatfmt = "g"

    # column alignment
    index_cols = set(mat.columns_as_index())
    index_align = "right"
    if len(data) == 0:
        ncols = 0
    elif isinstance(data[0], list):
        ncols = len(data[0])
    else:
        ncols = len(data)
    colalign = [index_align if idx in index_cols else options.cell_align for idx in range(ncols)]

    # base arguments
    tab_args = {
        "tabular_data": data,
        "headers": _labels_to_str(col_labels) if has_header else (),
        "showindex": False,
        "tablefmt": matrix_format,
        "floatfmt": floatfmt,
        "colalign": colalign,
        "disable_numparse": disable_numparse,
        **tab_extra_args
    }

    return tab_args, row_labels_txt


def to_tabulate(mat: Any, **kwargs) -> str:
    """
    Format a matrix to string using :func:`~tabulate.tabulate`.

    Supports argument passthrough.

    :param mat: A supported matrix.
    :param kwargs: Any argument in :class:`MatReprParams`, or any argument supported by :func:`~tabulate.tabulate`.
    :return: output of :func:`~tabulate.tabulate`
    """
    options = params.get(**kwargs)
    tab_extra_args = extract_forwardable_tabulate_args(**kwargs)
    adapter = _get_adapter(mat, options)

    if options.width_str:
        txt_max_cols = max(1, options.width_str // 3)  # minimum 3 chars per column (if empty)
        cols = min(options.max_cols, txt_max_cols)
        trunc = to_trunc(adapter, options.max_rows, cols, options.num_after_dots)

        args, row_labels_txt = _to_tabulate_args(trunc, adapter.is_tensor(), options, tab_extra_args)
        ret = tabulate(**args)
        ret = multiline_concat(row_labels_txt, ret)
        ret_width = max_line_width(ret)

        while ret_width > options.width_str and cols > 1:
            # too wide
            cols = trunc.drop_column()
            args, row_labels_txt = _to_tabulate_args(trunc, adapter.is_tensor(), options, tab_extra_args)
            ret = tabulate(**args)
            ret = multiline_concat(row_labels_txt, ret)
            ret_width = max_line_width(ret)

    else:
        args, row_labels_txt = _to_tabulate_args(adapter, adapter.is_tensor(), options, tab_extra_args)
        ret = tabulate(**args)
        ret = multiline_concat(row_labels_txt, ret)

    return ret
