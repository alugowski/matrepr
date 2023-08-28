# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause

from typing import Any, Iterable, Tuple
from . import describe, Driver, MatrixAdapterRow, MatrixAdapterCol


def is_single(x):
    return (not hasattr(x, '__len__') or isinstance(x, str))\
        or type(x).__module__.startswith("scipy")\
        or type(x).__module__.startswith("sparse")\
        or type(x).__name__ == "Matrix"


def count_none(iterable):
    return sum((1 if x is None else 0) for x in iterable)


class ListAdapter(MatrixAdapterRow):
    def __init__(self, mat: list):
        self.mat = mat
        self.row_lengths = [(1 if is_single(row) else len(row)) for row in mat]
        if len(mat) == 0 or all(is_single(row) for row in mat):
            # 1D
            self.shape = (len(mat),)
        else:
            # 2D
            self.shape = (len(mat), max(self.row_lengths if self.row_lengths else [0]))
        self.nnz = sum((1 if is_single(row) else len(row) - count_none(row)) for row in mat)

    def get_shape(self) -> Tuple[int, int]:
        return self.shape

    def describe(self) -> str:
        return describe(shape=self.shape, nnz=self.nnz, notes=None)

    def get_row(self, row_idx: int, col_range: Tuple[int, int]) -> Iterable[Tuple[int, Any]]:
        row = self.mat if len(self.shape) == 1 else self.mat[row_idx]

        if is_single(row):
            # this row is a single element
            if col_range[0] <= 0 < col_range[1]:
                return ((0, row),)
            else:
                return ()
        else:
            return enumerate(row[col_range[0]:col_range[1]], start=col_range[0])


class List1DColumnAdapter(MatrixAdapterCol):
    def __init__(self, mat: list):
        self.mat = mat
        self.shape = (len(mat), 1)
        self.nnz = len(mat)

    def get_shape(self) -> tuple:
        return self.shape

    def describe(self) -> str:
        return describe(shape=self.shape, nnz=self.nnz, notes=None)

    def get_col(self, col_idx: int, row_range: Tuple[int, int]) -> Iterable[Tuple[int, Any]]:
        assert col_idx == 0
        return enumerate(self.mat[row_range[0]:row_range[1]], start=row_range[0])


class ListDriver(Driver):
    @staticmethod
    def get_supported_types() -> Iterable[Tuple[str, bool]]:
        return [
            ("builtins.list", False),
            ("builtins.tuple", False),
        ]

    @staticmethod
    def adapt(mat: Any):
        if isinstance(mat, (list, tuple)):
            return ListAdapter(mat)
        raise ValueError
