# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause

from typing import Any, Dict, Tuple
from tabulate import tabulate, TableFormat, Line, DataRow

from . import params, _get_adapter
from .list_converter import ListConverter

matrix_table_format = TableFormat(
    lineabove=Line("┌", " ", " ", "┐"),
    linebelowheader=None,
    linebetweenrows=None,
    linebelow=Line("└", " ", " ", "┘"),
    headerrow=DataRow(" ", " ", ""),
    datarow=DataRow("│", " ", "│"),
    padding=1,
    with_header_hide=None,
)
"""
Renders table that looks like a LaTeX bmatrix.

Does not work with indices.
"""

matrix_format_patched_indices = TableFormat(
    lineabove=None,
    linebelowheader=Line("", " ", "┌", "┐"),
    linebetweenrows=None,
    linebelow=Line("", " ", "└", "┘"),
    headerrow=DataRow("", "│", ""),
    datarow=DataRow("", "│", "│"),
    padding=1,
    with_header_hide=None,
)
"""
A corrupted version of :ref:`matrix_table_format` that, when patched using :func:`_patch_index_tabulate_output`
yields the same beautiful bmatrix but with indices nicely outside the brackets.
"""


def _patch_index_tabulate_output(s: str) -> str:
    lines = s.split("\n")

    # header only has separators
    lines[0] = lines[0].replace("│", " ")

    # below-header row, data rows, bottom row
    for row in range(1, len(lines)):
        line = lines[row]

        for needle in ["┌", "│", "└", None]:
            if line.find(needle) >= 0:
                break

        if needle is None:
            continue

        save_inds = [line.find(needle)]
        if line.endswith(needle):
            save_inds.append(-1)

        line_chars = list(line.replace(needle, " "))
        for i in save_inds:
            line_chars[i] = needle

        lines[row] = "".join(line_chars)
    return "\n".join(lines)


def _labels_to_str(labels):
    for label in labels:
        yield "" if label is None else str(label)


def _to_tabulate_args(mat: Any, **kwargs) -> Tuple[Dict, bool]:
    """
    Creates arguments for :func:`~tabulate.tabulate` to format the matrix.

    Note that tabulate is not expressive enough to render a matrix with nice row indices. Tabulate does not distinguish
    between the index column and data columns. Therefore, index output renders a corrupted output that is then patched
    using :func:`_patch_index_tabulate_output`.

    :param mat: A supported matrix.
    :param kwargs: Any argument in :class:`MatReprParams`, or any argument supported by :func:`~tabulate.tabulate`.
    :return: the argument dictionary and a flag. If the flag is True then :func:`_patch_index_tabulate_output` must be
             called on the output.
    """
    options = params.get(**kwargs)
    adapter = _get_adapter(mat)

    # determine the table format
    if "tablefmt" in kwargs:
        matrix_format = kwargs["tablefmt"]
        patch_required = False
    elif options.indices:
        matrix_format = matrix_format_patched_indices
        patch_required = True
    else:
        matrix_format = matrix_table_format
        patch_required = False

    # collect the data
    conv = ListConverter(**options.to_kwargs())
    data, row_labels, col_labels = conv.to_lists_and_labels(adapter)

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
        "showindex": _labels_to_str(row_labels) if options.indices else False,
        "tablefmt": matrix_format,
        "floatfmt": floatfmt,
    }

    # colalign
    if "colalign" not in kwargs and options.indices:
        tab_args["colalign"] = (["right"] if len(row_labels) > 0 else []) + ["center"] * len(col_labels)

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
    args, patch_required = _to_tabulate_args(mat, **kwargs)
    ret = tabulate(**args)

    if patch_required:
        ret = _patch_index_tabulate_output(ret)
    return ret
