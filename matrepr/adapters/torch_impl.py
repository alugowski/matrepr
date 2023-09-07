# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause

from typing import Any, Iterable, Tuple

import torch

from . import describe, FallbackToNative, MatrixAdapterRow, MatrixAdapterCol, MatrixAdapterCoo, TensorAdapterCooRow


class PyTorchBase:
    def __init__(self, tensor: torch.Tensor):
        self.tensor: torch.Tensor = tensor

    def get_shape(self) -> tuple:
        return tuple(self.tensor.shape)

    def describe(self) -> str:
        parts = []

        try:
            get_nnz = getattr(self.tensor, "_nnz", None)
            nnz = get_nnz() if get_nnz else None
        except NotImplementedError:
            nnz = None

        return describe(shape=tuple(self.tensor.shape),
                        nnz=nnz, nz_type=self.tensor.dtype,
                        layout=self.tensor.layout,
                        notes=", ".join(parts))


class PyTorchFallbackAdapter(PyTorchBase, MatrixAdapterRow):
    def __init__(self, tensor):
        MatrixAdapterRow.__init__(self)
        PyTorchBase.__init__(self, tensor)

    def get_shape(self) -> tuple:
        return (1, )

    def get_row(self, row_idx: int, col_range: Tuple[int, int]) -> Iterable[Any]:
        raise FallbackToNative


class PyTorch1DTensorAdapter(PyTorchBase, MatrixAdapterRow):
    def __init__(self, tensor: torch.Tensor):
        assert len(tensor.shape) <= 1
        MatrixAdapterRow.__init__(self)
        PyTorchBase.__init__(self, tensor)
        self.row_labels = False
        self.coo: torch.Tensor = tensor.to_sparse_coo() if tensor.is_sparse else None

    def get_row(self, row_idx: int, col_range: Tuple[int, int]) -> Iterable[Any]:
        assert row_idx == 0

        if not self.tensor.is_sparse:
            # dense
            return enumerate([v.item() for v in self.tensor[slice(*col_range)]], start=col_range[0])

        indices = self.coo.indices()
        values = self.coo.values()
        mask = (col_range[0] <= indices) & (indices < col_range[1])
        return [(i.item(), v.item()) for i, v in zip(indices[mask], values[mask.reshape(len(values),)])]


class PyTorch2DTensorDenseAdapter(PyTorchBase, MatrixAdapterRow):
    def __init__(self, tensor: torch.Tensor):
        assert len(tensor.shape) == 2 and not tensor.is_sparse
        MatrixAdapterRow.__init__(self)
        PyTorchBase.__init__(self, tensor)

    def get_row(self, row_idx: int, col_range: Tuple[int, int]) -> Iterable[Any]:
        return enumerate([v.item() for v in self.tensor[row_idx, slice(*col_range)]], start=col_range[0])


class PyTorch2DTensorCSRAdapter(PyTorchBase, MatrixAdapterRow):
    def __init__(self, tensor: torch.Tensor):
        assert tensor.layout == torch.sparse_csr
        batchsize = tensor.shape[:-tensor.sparse_dim() - tensor.dense_dim()]
        if len(batchsize) > 0:
            # batches not supported yet
            # see also https://github.com/pytorch/pytorch/issues/107478
            raise FallbackToNative
        assert len(tensor.shape) == 2

        MatrixAdapterRow.__init__(self)
        PyTorchBase.__init__(self, tensor)
        self.indptr = tensor.crow_indices()
        self.indices = tensor.col_indices()
        self.values = tensor.values()

    def get_row(self, row_idx: int, col_range: Tuple[int, int]) -> Iterable[Any]:
        return [
            (self.indices[i].item(), self.values[i].item())
            for i in range(self.indptr[row_idx], self.indptr[row_idx + 1])
            if col_range[0] <= self.indices[i] < col_range[1]
        ]


class PyTorch2DTensorCSCAdapter(PyTorchBase, MatrixAdapterCol):
    def __init__(self, tensor: torch.Tensor):
        assert tensor.layout == torch.sparse_csc
        batchsize = tensor.shape[:-tensor.sparse_dim() - tensor.dense_dim()]
        if len(batchsize) > 0:
            # batches not supported yet
            raise FallbackToNative
        assert len(tensor.shape) == 2

        MatrixAdapterCol.__init__(self)
        PyTorchBase.__init__(self, tensor)
        self.indptr = tensor.ccol_indices()
        self.indices = tensor.row_indices()
        self.values = tensor.values()

    def get_col(self, col_idx: int, row_range: Tuple[int, int]) -> Iterable[Tuple[int, Any]]:
        return [
            (self.indices[i].item(), self.values[i].item())
            for i in range(self.indptr[col_idx], self.indptr[col_idx + 1])
            if row_range[0] <= self.indices[i] < row_range[1]
        ]


class PyTorch2DTensorCooAdapter(PyTorchBase, MatrixAdapterCoo):
    def __init__(self, tensor: torch.Tensor):
        assert len(tensor.shape) == 2
        MatrixAdapterCoo.__init__(self)
        PyTorchBase.__init__(self, tensor)
        self.coo: torch.Tensor = tensor.to_sparse_coo()
        self.coo = self.coo.coalesce()

    def get_coo(self, row_range: Tuple[int, int], col_range: Tuple[int, int]) -> Iterable[Tuple[int, int, Any]]:
        rows, cols = self.coo.indices()
        values = self.coo.values()
        row_mask = (row_range[0] <= rows) & (rows < row_range[1])
        col_mask = (col_range[0] <= cols) & (cols < col_range[1])
        mask = row_mask & col_mask
        return [(r.item(), c.item(), v.item())
                for r, c, v in zip(rows[mask], cols[mask], values[mask])]


class PyTorchNDTensorDenseAdapter(PyTorchBase, TensorAdapterCooRow):
    def __init__(self, tensor):
        PyTorchBase.__init__(self, tensor)
        TensorAdapterCooRow.__init__(self)
        self.nnz = sum(tuple(tensor.shape))
        self.ndim = len(tensor.shape)

    def get_shape(self) -> tuple:
        return self.nnz, self.ndim+1

    def get_dense_row(self, row_idx: int, col_range: Tuple[int, int]) -> Iterable[Any]:
        ret = self.coo.coords[:, row_idx]
        ret = list(ret) + [self.coo.data[row_idx]]
        return ret[slice(*col_range)]


class PyTorchNDTensorCooAdapter(PyTorchBase, TensorAdapterCooRow):
    def __init__(self, tensor):
        self.ndim = len(tensor.shape)
        self.nsparsedim = tensor.sparse_dim()
        try:
            # noinspection PyProtectedMember
            self.nnz = tensor._nnz()
        except (NotImplementedError, AttributeError):
            raise FallbackToNative

        PyTorchBase.__init__(self, tensor)
        TensorAdapterCooRow.__init__(self)
        self.coo = tensor.to_sparse_coo()

    def get_shape(self) -> tuple:
        return self.nnz, self.nsparsedim+1

    def get_dense_row(self, row_idx: int, col_range: Tuple[int, int]) -> Iterable[Any]:
        indices = self.coo.indices()
        end = min(col_range[1], self.nsparsedim)
        ret = [indices[i][row_idx].item() for i in range(col_range[0], end)]

        if col_range[1] == self.nsparsedim + 1:
            value = self.coo.values()[row_idx]
            if self.ndim == self.nsparsedim:
                value = value.item()
            ret.append(value)

        return ret
