# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause

from typing import Any, Iterable, Tuple

from . import Driver


class TensorFlowDriver(Driver):
    @staticmethod
    def get_supported_types() -> Iterable[Tuple[str, bool]]:
        return [
            ("tf.Tensor", True),
            ("tensorflow.python.framework.ops.Tensor", True),
            ("tensorflow.python.framework.ops.EagerTensor", True),
            ("tf.SparseTensor", True),
            ("tensorflow.python.framework.sparse_tensor.SparseTensor", True),
        ]

    @staticmethod
    def adapt(tensor: Any):
        import tensorflow as tf
        from .tensorflow_impl import TensorFlow1DTensorAdapter, TensorFlowFallbackAdapter
        from .tensorflow_impl import TensorFlow2DTensorDenseAdapter, TensorFlow2DTensorCooAdapter
        from .tensorflow_impl import TensorFlowNDTensorCooAdapter

        is_sparse = isinstance(tensor, tf.SparseTensor)
        ndims = len(tensor.shape)

        if ndims == 0:
            # single value
            return TensorFlowFallbackAdapter(tensor)
        elif ndims == 1:
            return TensorFlow1DTensorAdapter(tensor)
        elif ndims == 2:
            if is_sparse:
                return TensorFlow2DTensorCooAdapter(tensor)
            else:
                return TensorFlow2DTensorDenseAdapter(tensor)
        else:  # ndims > 2
            if is_sparse:
                return TensorFlowNDTensorCooAdapter(tensor)

        return TensorFlowFallbackAdapter(tensor)
