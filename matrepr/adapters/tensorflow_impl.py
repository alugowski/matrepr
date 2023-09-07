# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause

from typing import Any, Iterable, Union, Tuple

import numpy as np
import tensorflow as tf

from . import describe, FallbackToNative, MatrixAdapterRow, MatrixAdapterCoo, TensorAdapterCooRow


class TensorFlowBase:
    def __init__(self, tensor: Union[tf.Tensor, tf.SparseTensor]):
        self.tensor: Union[tf.Tensor, tf.SparseTensor] = tensor
        self.is_sparse = isinstance(tensor, tf.SparseTensor)

    def get_shape(self) -> tuple:
        return tuple(self.tensor.shape)

    def describe(self) -> str:
        parts = []

        nnz = len(self.tensor.values) if self.is_sparse else None

        return describe(shape=tuple(self.tensor.shape),
                        nnz=nnz, nz_type=self.tensor.dtype,
                        layout="tf.SparseTensor" if self.is_sparse else "tf.Tensor",
                        notes=", ".join(parts))


class TensorFlowFallbackAdapter(TensorFlowBase, MatrixAdapterRow):
    def __init__(self, tensor):
        MatrixAdapterRow.__init__(self)
        TensorFlowBase.__init__(self, tensor)

    def get_shape(self) -> tuple:
        return (1, )

    def get_row(self, row_idx: int, col_range: Tuple[int, int]) -> Iterable[Any]:
        raise FallbackToNative


class TensorFlow1DTensorAdapter(TensorFlowBase, MatrixAdapterRow):
    def __init__(self, tensor: Union[tf.Tensor, tf.SparseTensor]):
        assert len(tensor.shape) <= 1
        MatrixAdapterRow.__init__(self)
        TensorFlowBase.__init__(self, tensor)
        self.row_labels = False

    def get_row(self, row_idx: int, col_range: Tuple[int, int]) -> Iterable[Any]:
        assert row_idx == 0

        if not self.is_sparse:
            # dense
            return enumerate([v.numpy().item() for v in self.tensor[slice(*col_range)]], start=col_range[0])

        indices = self.tensor.indices.numpy()
        values = self.tensor.values.numpy()
        mask = (col_range[0] <= indices) & (indices < col_range[1])
        return [(i.item(), v.item()) for i, v in zip(indices[mask], values[mask.reshape(len(values),)])]


class TensorFlow2DTensorDenseAdapter(TensorFlowBase, MatrixAdapterRow):
    def __init__(self, tensor: tf.Tensor):
        assert len(tensor.shape) == 2
        MatrixAdapterRow.__init__(self)
        TensorFlowBase.__init__(self, tensor)

    def get_row(self, row_idx: int, col_range: Tuple[int, int]) -> Iterable[Any]:
        return enumerate([v.numpy().item() for v in self.tensor[row_idx, slice(*col_range)]], start=col_range[0])


class TensorFlow2DTensorCooAdapter(TensorFlowBase, MatrixAdapterCoo):
    def __init__(self, tensor: tf.SparseTensor):
        assert len(tensor.shape) == 2
        MatrixAdapterCoo.__init__(self)
        TensorFlowBase.__init__(self, tensor)

    def get_coo(self, row_range: Tuple[int, int], col_range: Tuple[int, int]) -> Iterable[Tuple[int, int, Any]]:
        # rc_pairs = self.tensor.indices.numpy()
        rows, cols = np.split(self.tensor.indices.numpy(), 2, axis=1)
        rows = rows.flat
        cols = cols.flat
        values = self.tensor.values.numpy()
        row_mask = (row_range[0] <= rows) & (rows < row_range[1])
        col_mask = (col_range[0] <= cols) & (cols < col_range[1])
        mask = row_mask & col_mask
        return [(r.item(), c.item(), v.item())
                for r, c, v in zip(rows[mask], cols[mask], values[mask])]


class TensorFlowNDTensorCooAdapter(TensorFlowBase, TensorAdapterCooRow):
    def __init__(self, tensor: tf.SparseTensor):
        self.ndim = len(tensor.shape)
        self.nnz = len(tensor.values)

        TensorFlowBase.__init__(self, tensor)
        TensorAdapterCooRow.__init__(self)

    def get_shape(self) -> tuple:
        return self.nnz, self.ndim+1

    def get_dense_row(self, row_idx: int, col_range: Tuple[int, int]) -> Iterable[Any]:
        indices = self.tensor.indices.numpy()
        end = min(col_range[1], self.ndim)
        ret = [indices[row_idx][i].item() for i in range(col_range[0], end)]

        if col_range[1] == self.ndim + 1:
            value = self.tensor.values[row_idx]
            value = value.numpy().item()
            ret.append(value)

        return ret
