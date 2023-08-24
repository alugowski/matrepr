# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause

from .adapters import MatrixAdapter, MatrixAdapterRow, Truncated2DMatrix, to_trunc, DupeList
from .base_formatter import unicode_dots


fixed_width_dots = {
    "h": "...",  # '⋯'
    "v": ":",  # '⋮'
    "d": "..."  # '⋱'
}
"""
Many fixed-width fonts don't have unicode ellipsis characters. Output viewed in those fonts may have misalignment
issues due to this.

This mapping uses plain characters present in all fonts that should be the same width everywhere.
"""

unicode_to_fixed_width = {
    u: fixed_width_dots[k] for k, u in unicode_dots.items()
}


class ListConverter:
    def __init__(self, max_rows, max_cols, num_after_dots, floatfmt, **_):
        super().__init__()
        self.max_rows = max_rows
        self.max_cols = max_cols
        self.num_after_dots = num_after_dots
        self.floatfmt = floatfmt
        self.replace_newline_char = " "
        self.use_fixed_width_dots = True

    def pprint(self, obj):
        if obj is None:
            return ""

        if type(obj).__module__ == "numpy":
            obj = obj.item()

        if isinstance(obj, (int, float)):
            return self.floatfmt(obj)

        if isinstance(obj, complex):
            sign = "-" if obj.imag < 0 else "+"
            r = self.pprint(obj.real).replace("e+", "e")
            c = self.pprint(obj.imag).replace("e+", "e")
            if "e" in r:
                # add space to visually separate the exponent from the complex plus
                sign = f" {sign} "
            return f"{r}{sign}{c}i"

        if isinstance(obj, str):
            if self.use_fixed_width_dots and obj in unicode_to_fixed_width:
                return unicode_to_fixed_width[obj]
            if obj in unicode_dots.values():
                return obj
            if "\n" in obj:
                obj = obj.replace("\n", self.replace_newline_char)

        if isinstance(obj, DupeList):
            return [self.pprint(sub) for sub in obj]

        from . import _get_adapter
        adapter = _get_adapter(obj, unsupported_raise=False)
        if adapter:
            fmt = ListConverter(max_rows=max(self.max_rows / 2, 2),
                                max_cols=max(self.max_cols / 2, 2),
                                num_after_dots=0,
                                floatfmt=self.floatfmt)
            ret, _, _ = fmt.to_lists_and_labels(adapter)
            return ret

        return obj

    def _write_matrix(self, mat: MatrixAdapterRow, empty_cell=""):
        if isinstance(mat, Truncated2DMatrix):
            mat.apply_dots(unicode_dots)

        nrows, ncols = mat.get_shape()

        ret = []
        for row in range(nrows):
            ret.append([empty_cell] * ncols)

        # values
        for row_idx in range(nrows):
            for col_idx, cell in mat.get_row(row_idx, col_range=(0, ncols)):
                if cell is not None:
                    ret[row_idx][col_idx] = self.pprint(cell)

        return ret

    def to_lists_and_labels(self, mat: MatrixAdapter):
        if not isinstance(mat, MatrixAdapterRow) or \
                len(mat.get_shape()) != 2 or \
                mat.get_shape()[0] > self.max_rows or \
                mat.get_shape()[1] > self.max_cols:
            mat = to_trunc(mat, self.max_rows, self.max_cols, self.num_after_dots)

        return self._write_matrix(mat), mat.get_row_labels(), mat.get_col_labels()
