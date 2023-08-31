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
            # Expressions:
            ("graphblas.core.expr.InfixExprBase", True),
            ("graphblas.core.infix.VectorInfixExpr", True),
            ("graphblas.core.infix.MatrixInfixExpr", True),
            ("graphblas.core.infix.VectorMatMulExpr", True),
            ("graphblas.core.infix.MatrixMatMulExpr", True),
            ("graphblas.core.vector.VectorExpression", True),
            ("graphblas.core.matrix.MatrixExpression", True),
            ("graphblas.core.vector.VectorIndexExpr", True),
            ("graphblas.core.matrix.MatrixIndexExpr", True),
        ]

    @staticmethod
    def adapt(mat: Any):
        from .graphblas_impl import GraphBLASMatrixAdapter, GraphBLASVectorAdapter
        type_name = type(mat).__name__
        if hasattr(mat, "_get_value"):
            # This is an expression. Compute the value and format that.
            # noinspection PyProtectedMember
            mat = mat._get_value()

        if "Matrix" == type(mat).__name__:
            return GraphBLASMatrixAdapter(mat, type_name)
        elif "Vector" == type(mat).__name__:
            return GraphBLASVectorAdapter(mat, type_name)
        else:
            raise ValueError("Unknown type: " + str(type(mat)))
