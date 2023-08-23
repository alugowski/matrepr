# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause

from typing import Any, Iterable, Tuple

import scipy.sparse

from . import describe, MatrixAdapterRow, MatrixAdapterCol, MatrixAdapterCoo


class SciPyAdapter:
    def __init__(self, mat):
        self.mat = mat

    def get_shape(self) -> tuple:
        return self.mat.shape

    def describe(self) -> str:
        format_name = self.mat.getformat()

        return describe(shape=self.mat.shape, nnz=self.mat.nnz, nz_type=self.mat.dtype,
                        notes=f"{format_name}")


class SciPyCSRAdapter(SciPyAdapter, MatrixAdapterRow):
    def __init__(self, mat: scipy.sparse.csr_matrix):
        super().__init__(mat)

    def get_row(self, row_idx: int, col_range: Tuple[int, int]) -> Iterable[Tuple[int, Any]]:
        start = self.mat.indptr[row_idx]
        end = self.mat.indptr[row_idx + 1]

        indices = self.mat.indices[start:end]
        data = self.mat.data[start:end]

        mask = (col_range[0] <= indices) & (indices < col_range[1])
        return zip(indices[mask], data[mask])


class SciPyCSCAdapter(SciPyAdapter, MatrixAdapterCol):
    def __init__(self, mat: scipy.sparse.csc_matrix):
        super().__init__(mat)

    def get_col(self, col_idx: int, row_range: Tuple[int, int]) -> Iterable[Tuple[int, Any]]:
        start = self.mat.indptr[col_idx]
        end = self.mat.indptr[col_idx + 1]

        indices = self.mat.indices[start:end]
        data = self.mat.data[start:end]

        mask = (row_range[0] <= indices) & (indices < row_range[1])
        return zip(indices[mask], data[mask])


class SciPyCOOAdapter(SciPyAdapter, MatrixAdapterCoo):
    def __init__(self, mat: scipy.sparse.coo_matrix):
        super().__init__(mat)

    def get_coo(self, row_range: Tuple[int, int], col_range: Tuple[int, int]) -> Iterable[Tuple[int, int, Any]]:
        mask = (self.mat.row >= row_range[0]) & (self.mat.row < row_range[1]) & \
               (self.mat.col >= col_range[0]) & (self.mat.col < col_range[1])

        return zip(self.mat.row[mask], self.mat.col[mask], self.mat.data[mask])
