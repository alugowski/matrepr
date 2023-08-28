# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause

from typing import Any, Iterable, Tuple

from . import Driver


class GraphBLASDriver(Driver):
    @staticmethod
    def get_supported_types() -> Iterable[Tuple[str, bool]]:
        return [
            ("graphblas.Matrix", True),
            ("graphblas.core.matrix.Matrix", True),
            ("graphblas.Vector", True),
            ("graphblas.core.vector.Vector", True),
        ]

    @staticmethod
    def adapt(mat: Any):
        from .graphblas_impl import GraphBLASMatrixAdapter, GraphBLASVectorAdapter
        if "Matrix" in str(type(mat)):
            return GraphBLASMatrixAdapter(mat)
        elif "Vector" in str(type(mat)):
            return GraphBLASVectorAdapter(mat)
        else:
            raise ValueError("Unknown type: " + str(type(mat)))
