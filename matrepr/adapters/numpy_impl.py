# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause

from typing import Any, Iterable, Tuple

from . import describe, MatrixAdapterRow


class NumpyArrayAdapter(MatrixAdapterRow):
    def __init__(self, mat):
        self.mat = mat
        self.is_vec = len(mat.shape) < 2

    def get_shape(self) -> tuple:
        # present at most 2D, higher dimensions will be handled as nested arrays
        if self.is_vec:
            return (self.mat.shape[0],)
        else:
            return self.mat.shape[0:2]

    def describe(self) -> str:
        return describe(shape=self.mat.shape,
                        nnz=self.mat.size, nz_type=self.mat.dtype,
                        notes="array")

    def get_row(self, row_idx: int, col_range: Tuple[int, int]) -> Iterable[Any]:
        row = self.mat if self.is_vec else self.mat[row_idx]
        return enumerate(row[col_range[0]:col_range[1]], start=col_range[0])
