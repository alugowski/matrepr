# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause

from typing import Any, Iterable, Tuple

from . import Driver


class PyDataSparseDriver(Driver):
    @staticmethod
    def get_supported_types() -> Iterable[Tuple[str, bool]]:
        return [
            ("sparse.COO", True),
            ("sparse._coo.core.COO", True),
            ("sparse.DOK", True),
            ("sparse._dok.DOK", True),
            ("sparse.GCXS", True),
            ("sparse._compressed.compressed.GCXS", True),
            ("sparse.CSR", True),
            ("sparse._compressed.compressed.CSR", True),
            ("sparse.CSC", True),
            ("sparse._compressed.compressed.CSC", True),
        ]

    @staticmethod
    def adapt(mat: Any):
        from .sparse_impl import PyDataSparse1DAdapter, PyDataSparse2DAdapter, PyDataSparseNDAdapter
        dims = len(mat.shape)
        if dims < 2:
            return PyDataSparse1DAdapter(mat)
        elif dims == 2:
            return PyDataSparse2DAdapter(mat)
        else:
            return PyDataSparseNDAdapter(mat)
