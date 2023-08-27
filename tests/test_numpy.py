# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause

import unittest

import numpy as np

from matrepr import to_html, to_latex, to_str

np.random.seed(123)


class NumpyTests(unittest.TestCase):
    def setUp(self):
        self.mats = [
            np.array([]),
            np.array([1, 2, 3, 4]),
            np.array([[1, 2], [1003, 1004]]),
            np.array([
                [[1, 2], [3, 4]],
                [[100, 200], [300, 400]],
            ]),
            np.random.randint(low=0, high=10_000, size=(2, 2, 2, 2))  # 4D
        ]

    def test_no_crash(self):
        for mat in self.mats:
            res = to_str(mat, title=True)
            self.assertGreater(len(res), 10)

            res = to_html(mat, title=True)
            self.assertGreater(len(res), 10)

            res = to_latex(mat, title=True)
            self.assertGreater(len(res), 10)

    def test_contents_html(self):
        for mat in self.mats:
            if len(mat) == 0:
                continue

            res = to_html(mat, notebook=True, max_rows=200, max_cols=200, title=False, indices=False)
            self.assertGreater(len(res), 10)

            for value in np.nditer(mat):
                self.assertIn(f"<td>{value}</td>", res)


if __name__ == '__main__':
    unittest.main()
