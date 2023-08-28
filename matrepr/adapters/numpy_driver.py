# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause

from typing import Any, Iterable, Tuple

from . import Driver


class NumpyDriver(Driver):
    @staticmethod
    def get_supported_types() -> Iterable[Tuple[str, bool]]:
        return [
            ("numpy.ndarray", True),
        ]

    @staticmethod
    def adapt(mat: Any):
        from .numpy_impl import NumpyArrayAdapter
        return NumpyArrayAdapter(mat)
