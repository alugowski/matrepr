# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause

import bisect
from typing import Any, Iterable, Tuple

import numpy as np
import scipy.sparse

from . import describe, MatrixAdapterRow, MatrixAdapterCol, MatrixAdapterCoo


class CompressedSortedIterator:
    """A dense iterator over a CSR row or CSC column with sorted indices."""
    def __init__(self, indices_slice, data_slice, desired_index_range):
        self.indices_slice = indices_slice
        self.data_slice = data_slice
        self.index_iterator = iter(range(*desired_index_range))
        self.pos = bisect.bisect_left(indices_slice, desired_index_range[0])

    def __iter__(self):
        return self

    def __next__(self):
        # index of value being returned
        idx = next(self.index_iterator)

        if self.pos >= len(self.indices_slice):
            # reached past the end of this row/column
            return None

        if idx == self.indices_slice[self.pos]:
            # pointing at the next value in the row/column, so return it
            cur_pos = self.pos
            self.pos += 1
            return self.data_slice[cur_pos]

        # empty space
        return None


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

    def get_row(self, row_idx: int, col_range: Tuple[int, int]) -> Iterable[Any]:
        start = self.mat.indptr[row_idx]
        end = self.mat.indptr[row_idx + 1]

        index_slice = self.mat.indices[start:end]
        data_slice = self.mat.data[start:end]

        if not self.mat.has_sorted_indices:
            perm = np.argsort(index_slice)
            index_slice = index_slice[perm]
            data_slice = data_slice[perm]

        return CompressedSortedIterator(index_slice, data_slice, col_range)


class SciPyCSCAdapter(SciPyAdapter, MatrixAdapterCol):
    def __init__(self, mat: scipy.sparse.csc_matrix):
        super().__init__(mat)

    def get_col(self, col_idx: int, row_range: Tuple[int, int]) -> Iterable[Any]:
        start = self.mat.indptr[col_idx]
        end = self.mat.indptr[col_idx + 1]

        index_slice = self.mat.indices[start:end]
        data_slice = self.mat.data[start:end]

        if not self.mat.has_sorted_indices:
            perm = np.argsort(index_slice)
            index_slice = index_slice[perm]
            data_slice = data_slice[perm]

        return CompressedSortedIterator(index_slice, data_slice, row_range)


class SciPyCOOAdapter(SciPyAdapter, MatrixAdapterCoo):
    def __init__(self, mat: scipy.sparse.coo_matrix):
        super().__init__(mat)

    def get_coo(self, row_range: Tuple[int, int], col_range: Tuple[int, int]) -> Iterable[Tuple[int, int, Any]]:
        mask = (self.mat.row >= row_range[0]) & (self.mat.row < row_range[1]) & \
               (self.mat.col >= col_range[0]) & (self.mat.col < col_range[1])

        return zip(self.mat.row[mask], self.mat.col[mask], self.mat.data[mask])
