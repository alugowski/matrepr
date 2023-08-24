# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause

from typing import Any, Iterable, Tuple
from . import describe, Driver, MatrixAdapterRow


def is_single(x):
    return not hasattr(x, '__len__') or isinstance(x, str)


class ListAdapter(MatrixAdapterRow):
    def __init__(self, mat: list):
        self.mat = mat
        self.row_lengths = [(1 if is_single(row) else len(row)) for row in mat]
        self.shape = (len(mat), max(self.row_lengths if self.row_lengths else [0]))
        self.nnz = sum((1 if is_single(row) else len(row) - row.count(None)) for row in mat)

    def get_shape(self) -> Tuple[int, int]:
        return self.shape

    def describe(self) -> str:
        return describe(shape=self.shape, nnz=self.nnz, notes=None)

    def get_row(self, row_idx: int, col_range: Tuple[int, int]) -> Iterable[Tuple[int, Any]]:
        row = self.mat[row_idx]

        if is_single(row):
            # this row is a single element
            if col_range[0] <= 0 < col_range[1]:
                return ((0, row),)
            else:
                return ()
        else:
            return enumerate(row[col_range[0]:col_range[1]], start=col_range[0])


class ListDriver(Driver):
    @staticmethod
    def get_supported_types() -> Iterable[Tuple[str, str, bool]]:
        return [
            ("builtins", "list", False),
            ("builtins", "tuple", False),
        ]

    @staticmethod
    def adapt(mat: Any):
        if isinstance(mat, (list, tuple)):
            return ListAdapter(mat)
        raise ValueError
