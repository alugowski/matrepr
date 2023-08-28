# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause

from typing import Any, Iterable, Tuple

import sparse
from sparse import COO

from . import describe, MatrixAdapterRow, MatrixAdapterCoo, TensorAdapterCooRow


def sizeof_fmt(num, suffix="B"):
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}".rstrip('0').rstrip('.')+f"{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


class PyDataSparseBase:
    def __init__(self, mat):
        self.mat = mat

    def get_shape(self) -> tuple:
        return self.mat.shape

    def describe(self) -> str:
        parts = [
            self.mat.format,
            f"fill_value={self.mat.fill_value}",
        ]

        if not hasattr(self.mat, "__setitem__"):
            parts.append("read-only")

        if hasattr(self.mat, "nbytes"):
            parts.append(sizeof_fmt(self.mat.nbytes))

        return describe(shape=self.mat.shape,
                        nnz=self.mat.nnz, nz_type=self.mat.dtype,
                        notes=", ".join(parts))


class PyDataSparse1DAdapter(PyDataSparseBase, MatrixAdapterRow):
    def __init__(self, mat):
        assert mat.ndim <= 1
        super().__init__(mat)

    def get_row(self, row_idx: int, col_range: Tuple[int, int]) -> Iterable[Any]:
        assert row_idx == 0

        ret = COO(self.mat[slice(*col_range)])
        ret.coords += col_range[0]
        return zip(ret.coords.flat, ret.data.flat)


class PyDataSparse2DAdapter(PyDataSparseBase, MatrixAdapterCoo):
    def __init__(self, mat):
        assert mat.ndim == 2
        super().__init__(mat)

    def get_coo(self, row_range: Tuple[int, int], col_range: Tuple[int, int]) -> Iterable[Tuple[int, int, Any]]:
        ret = COO(self.mat[slice(*row_range), slice(*col_range)])
        ret.coords[0] += row_range[0]
        ret.coords[1] += col_range[0]
        return zip(ret.coords[0], ret.coords[1], ret.data)


class PyDataSparseNDAdapter(PyDataSparseBase, TensorAdapterCooRow):
    def __init__(self, mat):
        super().__init__(mat)
        self.coo = sparse.as_coo(mat)

    def get_shape(self) -> tuple:
        return self.mat.nnz, self.mat.ndim+1

    def get_dense_row(self, row_idx: int, col_range: Tuple[int, int]) -> Iterable[Any]:
        ret = self.coo.coords[:, row_idx]
        ret = list(ret) + [self.coo.data[row_idx]]
        return ret[slice(*col_range)]


def _warmup_indexing():
    _ = COO(coords=[[0, 1], [2, 3]], data=[111, 222], shape=(5, 5))[:1]


_warmup_indexing()
