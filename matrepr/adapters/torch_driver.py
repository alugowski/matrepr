# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause

from typing import Any, Iterable, Tuple

from . import Driver


class PyTorchDriver(Driver):
    @staticmethod
    def get_supported_types() -> Iterable[Tuple[str, bool]]:
        return [
            ("torch.Tensor", True),
        ]

    @staticmethod
    def adapt(tensor: Any):
        import torch
        from . import FallbackToNative
        from .torch_impl import PyTorch1DTensorAdapter, PyTorchFallbackAdapter
        from .torch_impl import PyTorch2DTensorDenseAdapter, PyTorch2DTensorCooAdapter
        from .torch_impl import PyTorch2DTensorCSRAdapter, PyTorch2DTensorCSCAdapter
        from .torch_impl import PyTorchNDTensorCooAdapter

        sparse_layouts = [
            torch.sparse_coo,
            torch.sparse_csr,
            torch.sparse_csc,
            torch.sparse_bsr,
            torch.sparse_bsc,
        ]

        # test layouts because it's possible to have is_sparse == False and is_sparse_csr == True
        is_sparse = tensor.is_sparse or tensor.layout in sparse_layouts
        ndims = len(tensor.shape)

        batchsize = tensor.shape[:-tensor.sparse_dim() - tensor.dense_dim()]
        if len(batchsize) != 0:
            # batches not supported yet
            return PyTorchFallbackAdapter(tensor)

        try:
            if ndims == 0:
                # single value
                return PyTorchFallbackAdapter(tensor)
            elif ndims == 1:
                return PyTorch1DTensorAdapter(tensor)
            elif tensor.layout == torch.sparse_csr:
                return PyTorch2DTensorCSRAdapter(tensor)
            elif tensor.layout == torch.sparse_csc:
                return PyTorch2DTensorCSCAdapter(tensor)
            elif ndims == 2:
                if is_sparse:
                    return PyTorch2DTensorCooAdapter(tensor)
                else:
                    return PyTorch2DTensorDenseAdapter(tensor)
            else:  # ndims > 2
                if is_sparse:
                    return PyTorchNDTensorCooAdapter(tensor)
        except FallbackToNative:
            pass

        return PyTorchFallbackAdapter(tensor)
