# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause

import unittest

from matrepr import to_html, to_latex, to_str
import matrepr


class ListLikeTests(unittest.TestCase):
    def setUp(self):
        self.mats = [
            [],
            [1, 2, 3, 4],
            (1, 2, 3, 4),
            [[1, 2], [1003, 1004]],
            [[1, 2], [1003, 1004, 1005]],
        ]

    def test_no_crash(self):
        for mat in self.mats:
            res = to_html(mat, title=True)
            self.assertGreater(len(res), 10)

            res = to_latex(mat, title=True)
            self.assertGreater(len(res), 10)

    def test_no_crash_edge_cases(self):
        try:
            # noinspection PyUnresolvedReferences
            from scipy.sparse import coo_matrix
            # noinspection PyUnresolvedReferences
            import numpy as np

            sps_mat = coo_matrix(([1, 2, 3, 4], ([0, 0, 1, 1], [0, 1, 0, 1])), shape=(2, 2))

            dtype_a = np.dtype([("x", np.bool_), ("y", np.int64)], align=True)
            np_a = np.array([(False, 2)], dtype=dtype_a)[0]
            dtype_b = np.dtype("(3,)uint16")
            np_b = np.array([(1, 2, 3)], dtype=dtype_b)[0]
        except ImportError:
            sps_mat = None
            np_a = None
            np_b = None

        list_mat = [
            (0, 12e34, 1e-6, None, 123456789),
            1,
            [complex(1, 2), complex(123456, 0.123456)],
            [[1], sps_mat, [2.1, 2.2], [[1.1, 2.2], [3.3, 4.4]]],
            ["multiline\nstring", "<escape!>", "\\begin{escape!}", {"a Python set"}],
            [np_a, np_b]
        ]

        res = to_html(list_mat, notebook=True, title=True)
        self.assertGreater(len(res), 10)

        res = to_latex(list_mat, title=True)
        self.assertGreater(len(res), 10)

        # tabulate SEPARATING_LINE detection will compare an element to a string. If that element happens to be a
        # numpy array, numpy issues this warning. It's a ValueError in future versions.
        # Ensure it's not thrown.
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings(action='error', category=FutureWarning)

            res = to_str(list_mat, title=True)
        self.assertGreater(len(res), 10)

    def test_shape(self):
        mat = (1, 2, 3, 4)
        adapter = matrepr._get_adapter(mat)
        self.assertEqual((4,), adapter.get_shape())

        mat = [[1, 2], [1003, 1004, 1005]]
        adapter = matrepr._get_adapter(mat)
        self.assertEqual((2, 3), adapter.get_shape())


if __name__ == '__main__':
    unittest.main()
