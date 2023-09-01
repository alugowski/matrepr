# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause

from typing import Any, Iterable, Tuple

import graphblas as gb

from . import describe, MatrixAdapterCoo, MatrixAdapterRow


class GraphBLASAdapter:
    def __init__(self, mat, type_name):
        self.mat = mat
        self.type_name = type_name

    def get_shape(self) -> tuple:
        return self.mat.shape

    def get_format(self, is_transposed=False):
        x = self.mat
        try:
            # SS, SuiteSparse-specific: format (ends with "r" or "c"), and is_iso
            fmt = x.ss.format
            if is_transposed:
                fmt = fmt[:-1] + ("c" if fmt[-1] == "r" else "r")
            if x.ss.is_iso:
                return f"{fmt} (iso)"
            return fmt
        except AttributeError:
            return None

    def describe(self) -> str:
        parts = [f"gb.{self.type_name}"]

        return describe(shape=self.mat.shape,
                        nnz=self.mat.nvals, nz_type=self.mat.dtype,
                        layout=self.get_format(),
                        notes=", ".join(parts))


class GraphBLASMatrixAdapter(GraphBLASAdapter, MatrixAdapterCoo):
    def __init__(self, mat: gb.Matrix, type_name):
        MatrixAdapterCoo.__init__(self)
        GraphBLASAdapter.__init__(self, mat, type_name)

    def get_coo(self, row_range: Tuple[int, int], col_range: Tuple[int, int]) -> Iterable[Tuple[int, int, Any]]:
        ret = self.mat[slice(*row_range), slice(*col_range)]

        rows, cols, vals = ret.to_coo()
        rows += row_range[0]
        cols += col_range[0]
        return zip(rows, cols, vals)


class GraphBLASVectorAdapter(GraphBLASAdapter, MatrixAdapterRow):
    def __init__(self, vec: gb.Vector, type_name):
        MatrixAdapterRow.__init__(self)
        GraphBLASAdapter.__init__(self, vec, type_name)
        self.row_labels = False

    def get_row(self, row_idx: int, col_range: Tuple[int, int]) -> Iterable[Any]:
        assert row_idx == 0

        idx, vals = self.mat[slice(*col_range)].to_coo()
        idx += col_range[0]
        return zip(idx, vals)
