# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause

import html

from .adapters import MatrixAdapter, MatrixAdapterRow, Truncated2DMatrix, to_trunc, DupeList
from .base_formatter import BaseFormatter, unicode_dots


html_dots = {
    "h": "&ctdot;",  # '⋯'
    "v": "&vellip;",  # '⋮'
    "d": "&dtdot;"  # '⋱'
}

unicode_to_html = {
    u: html_dots[k] for k, u in unicode_dots.items()
}


class HTMLTableFormatter(BaseFormatter):
    def __init__(self, max_rows, max_cols, num_after_dots, title, indices=False, cell_align="center",
                 floatfmt=None, **_):
        super().__init__()
        self.max_rows = max_rows
        self.max_cols = max_cols
        self.num_after_dots = num_after_dots
        self.title = title
        self.indices = indices
        self.cell_align = cell_align
        self.floatfmt = floatfmt if floatfmt else lambda f: format(f)
        self.indent_width = 4
        self.left_td_class = None
        self.right_td_class = None
        self.thead_class = None
        self.center_header = False
        self.table_attributes = None

    # noinspection PyMethodMayBeStatic
    def _attributes_to_string(self, attributes: dict) -> str:
        if attributes is None:
            return ""

        ret = ' '.join([f'{k}="{v}"' for k, v in attributes.items()])
        if ret:
            ret = " " + ret
        return ret

    def pprint(self, obj, current_indent=0):
        if obj is None:
            return ""

        if isinstance(obj, (int, float)):
            return self.floatfmt(obj)

        if isinstance(obj, complex):
            sign = "-" if obj.imag < 0 else "+"
            r = self.pprint(obj.real).replace("e+", "e")
            c = self.pprint(obj.imag).replace("e+", "e")
            if "e" in r:
                # add space to visually separate the exponent from the complex plus
                sign = f" {sign} "
            return f"{r}{sign}{c}<i>i</i>"

        if isinstance(obj, str):
            if obj in unicode_to_html:
                return unicode_to_html[obj]
            else:
                return html.escape(obj).replace("\n", "<br>")

        if isinstance(obj, DupeList):
            return "<br>".join(self.pprint(sub) for sub in obj)

        from . import _get_adapter
        adapter = _get_adapter(obj, unsupported_raise=False)
        if adapter:
            from copy import copy
            fmt = copy(self)
            fmt.lines = []
            fmt.max_rows = max(self.max_rows / 2, 2)
            fmt.max_cols = max(self.max_cols / 2, 2)
            fmt.num_after_dots = 0
            fmt.title = None
            fmt.indices = False
            fmt.table_attributes = {"style": "margin: 0 auto;"}
            return "\n" + str(fmt.format(adapter, indent=current_indent + self.indent_width))

        return html.escape(str(obj))

    def _write_matrix(self, mat: MatrixAdapterRow,
                      indent: int = 0):
        if isinstance(mat, Truncated2DMatrix):
            mat.apply_dots(unicode_dots)

        nrows, ncols = mat.get_shape()
        self.write(f"<table{self._attributes_to_string(self.table_attributes)}>", indent=indent)

        body_indent = indent + self.indent_width
        cell_indent = body_indent + self.indent_width

        # Header
        if self.indices:
            thead_class = f' class="{self.thead_class}"' if self.thead_class else ""
            self.write(f'<thead{thead_class}>', indent=body_indent)
            self.write("<tr>", indent=body_indent)
            self.write(f"<th></th>", indent=cell_indent)
            for col_label in mat.get_col_labels():
                attr = ' style="text-align: center;"' if self.center_header else ""
                self.write(f"<th{attr}>{self.pprint(col_label)}</th>", indent=cell_indent)
            self.write("</tr>", indent=body_indent)
            self.write("</thead>", indent=body_indent)

        # values
        self.write("<tbody>", indent=body_indent)
        row_labels = iter(mat.get_row_labels()) if self.indices else None
        for row_idx in range(nrows):
            self.write("<tr>", body_indent)
            if self.indices:
                self.write(f"<th>{self.pprint(next(row_labels))}</th>", cell_indent)

            col_range = (0, ncols)
            for col_idx, cell in enumerate(mat.get_dense_row(row_idx, col_range=col_range)):
                td_classes = []
                if col_idx == col_range[0] and self.left_td_class:
                    td_classes.append(self.left_td_class)
                if col_idx == col_range[1] - 1 and self.right_td_class:
                    td_classes.append(self.right_td_class)
                td_class = f' class="{" ".join(td_classes)}"' if td_classes else ""
                self.write(f"<td{td_class}>{self.pprint(cell, cell_indent)}</td>", cell_indent)

            self.write("</tr>", body_indent)

        self.write("</tbody>", indent=body_indent)

        self.write("</table>", indent=indent)

    def format(self, mat: MatrixAdapter, indent: int = 0):
        if self.title:
            title = mat.describe() if self.title is True else self.title
            self.write(f"<p>{title}</p>")

        if not isinstance(mat, MatrixAdapterRow) or \
                len(mat.get_shape()) != 2 or \
                mat.get_shape()[0] > self.max_rows or \
                mat.get_shape()[1] > self.max_cols:
            mat = to_trunc(mat, self.max_rows, self.max_cols, self.num_after_dots)

        self._write_matrix(mat, indent=indent)

        return self


class NotebookHTMLFormatter(HTMLTableFormatter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _write_style(self):
        index_attributes = {"font-size": "smaller", "vertical-align": "middle"}
        self.write("<style scoped>")
        for tags, attributes in [
            ("thead.head_no_border", {"border": "0px"}),
            ("tbody tr th", {**index_attributes, "text-align": "right"}),  # row indices
            ("thead tr th", {**index_attributes, "text-align": "center"}),  # column indices
            ("tbody tr td", {"vertical-align": "middle", "text-align": self.cell_align}),
            ("tbody tr td.left_cell", {"border-left": "solid 2px"}),  # left border
            ("tbody tr td.right_cell", {"border-right": "solid 2px"}),  # right border
            ("tbody tr td:empty::after", {"content": "'&nbsp;'", "visibility": "hidden"}),  # fill empty cells
        ]:
            self.write(f"{tags} " + '{', indent=self.indent_width)
            for k, v in attributes.items():
                self.write(f"{k}: {v};", indent=2*self.indent_width)
            self.write("}", indent=self.indent_width)
        self.write("</style>")

    def format(self, mat: MatrixAdapter, indent: int = 0):
        self.write("<div>")
        self._write_style()
        self.left_td_class = "left_cell"
        self.right_td_class = "right_cell"
        self.thead_class = "head_no_border"
        self.center_header = True
        super().format(mat, indent=indent)
        self.write("</div>")
        return self
