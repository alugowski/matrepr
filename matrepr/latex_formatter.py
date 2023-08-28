# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause

import re

from .adapters import MatrixAdapter, MatrixAdapterRow, Truncated2DMatrix, to_trunc, DupeList
from .base_formatter import BaseFormatter, unicode_dots


def python_scientific_to_latex_times10(s):
    return re.sub(r'e\+?(-?)0*([0-9]+)', ' \\\\times 10^{\\1\\2}', s)


def tex_escape(text):
    """
    :param text: a plain text message
    :return: the message escaped to appear correctly in LaTeX
    """
    conv = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\^{}',
        # '\\': r'\textbackslash{}',
        '\\': r'\\',
        # '<': r'\textless{}',
        # '>': r'\textgreater{}',
        " '": r' `',
    }
    regex = re.compile('|'.join(re.escape(str(key)) for key in sorted(conv.keys(), key=lambda item: - len(item))))
    return regex.sub(lambda match: conv[match.group()], text)


latex_dots = {
    "h": "\\dots",  # '⋯'
    "v": "\\vdots",  # '⋮'
    "d": "\\ddots"  # '⋱'
}

unicode_to_latex = {
    u: latex_dots[k] for k, u in unicode_dots.items()
}


class LatexFormatter(BaseFormatter):
    def __init__(self, max_rows, max_cols, num_after_dots, title_latex, latex_matrix_env, latex_tensor_env,
                 floatfmt_latex=None, latex_dupe_matrix_env="Bmatrix", **_):
        super().__init__()
        self.max_rows = max_rows
        self.max_cols = max_cols
        self.num_after_dots = num_after_dots
        self.title = title_latex
        self.latex_matrix_env = latex_matrix_env
        self.latex_tensor_env = latex_tensor_env
        self.dupe_env = latex_dupe_matrix_env
        self.floatfmt = floatfmt_latex
        if not self.floatfmt:
            self.floatfmt = lambda f: python_scientific_to_latex_times10(format(f))
        self.indent_width = 4

    def pprint(self, obj):
        if obj is None:
            return ""

        if isinstance(obj, (int, float)):
            return self.floatfmt(obj)

        if isinstance(obj, complex):
            r = obj.real
            c = obj.imag
            sign = "-" if c < 0 else "+"
            return f"{self.pprint(r)}{sign}{self.pprint(c)}i"

        dupe_list_env = self.dupe_env
        if isinstance(obj, str):
            if obj in unicode_to_latex:
                return unicode_to_latex[obj]
            elif "\n" in obj:
                obj = DupeList(obj.split("\n"))
                dupe_list_env = "matrix"
            else:
                return "\\textrm{" + tex_escape(obj) + "}"

        if isinstance(obj, DupeList):
            fmt = LatexFormatter(max_rows=len(obj), max_cols=1, num_after_dots=0, title_latex=None,
                                 latex_matrix_env=dupe_list_env,
                                 latex_dupe_matrix_env=self.dupe_env,
                                 latex_tensor_env=self.latex_tensor_env,
                                 floatfmt_latex=self.floatfmt)
            from .adapters.list_like import List1DColumnAdapter
            return str(fmt.format(List1DColumnAdapter(obj)))

        from . import _get_adapter
        adapter = _get_adapter(obj, None, unsupported_raise=False)
        if adapter:
            fmt = LatexFormatter(max_rows=max(self.max_rows / 2, 2),
                                 max_cols=max(self.max_cols / 2, 2),
                                 num_after_dots=0, title_latex=None,
                                 latex_matrix_env=self.latex_matrix_env,
                                 latex_tensor_env=self.latex_tensor_env,
                                 latex_dupe_matrix_env=self.dupe_env,
                                 floatfmt_latex=self.floatfmt)
            return str(fmt.format(adapter))

        return "\\textrm{" + tex_escape(str(obj)) + "}"

    def _write_matrix(self, mat: MatrixAdapterRow, matrix_env, indent: int = 0):
        if isinstance(mat, Truncated2DMatrix):
            mat.apply_dots(unicode_dots)

        nrows, ncols = mat.get_shape()
        self.write("\\begin{" + matrix_env + "}", indent=indent)

        body_indent = indent + self.indent_width

        # values
        for row_idx in range(nrows):
            row_contents = []

            col_range = (0, ncols)
            for col_idx, cell in enumerate(mat.get_dense_row(row_idx, col_range=col_range)):
                if cell is not None:
                    row_contents.append(self.pprint(cell))

                if col_idx == col_range[1] - 1:
                    if row_idx != nrows - 1:
                        row_contents.append("\\\\")
                else:
                    row_contents.append("&")

            self.write(" ".join(row_contents), body_indent)

        self.write("\\end{" + matrix_env + "}", indent=indent)

    def format(self, mat: MatrixAdapter):
        if self.title:
            title = mat.describe() if self.title is True else self.title
            self.write("\\stackrel{\\textrm{" + tex_escape(title) + "}}{")

        matrix_env = self.latex_tensor_env if mat.is_tensor() else self.latex_matrix_env

        if not isinstance(mat, MatrixAdapterRow) or \
                len(mat.get_shape()) != 2 or \
                mat.get_shape()[0] > self.max_rows or \
                mat.get_shape()[1] > self.max_cols:
            mat = to_trunc(mat, self.max_rows, self.max_cols, self.num_after_dots)

        self._write_matrix(mat, matrix_env)

        if self.title:
            self.write("}")
        return self
