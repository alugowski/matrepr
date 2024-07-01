# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause

import unittest

try:
    import numpy as np

    np.random.seed(123)
except ImportError:
    np = None

from matrepr import to_html, to_latex, to_str


@unittest.skipIf(np is None, "numpy not installed")
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
            np.random.randint(low=0, high=10_000, size=(3, 5, 10)),  # 3D
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
                self.assertIn(f">{value}</td>", res)

    def test_large_size(self):
        mats = [
            np.arange(1000),
            np.arange(1000).reshape(25, 40),
        ]

        for mat in mats:
            res = to_str(mat, max_rows=None, max_cols=None)
            self.assertNotIn("...", res)

            res = to_html(mat, max_rows=None, max_cols=None)
            self.assertNotIn("dot;", res)

            res = to_latex(mat, max_rows=None, max_cols=None)
            self.assertNotIn("dots", res)

    def test_gh_35(self):
        mat = np.arange(5000).reshape(2, 2500)
        res = to_str(mat, width_str=50)
        self.assertNotIn("6 ", res)

    def test_auto_width_str_vals(self):
        mat = np.arange(5000)
        # Try a range of string widths.
        for width_str in range(1, 200):
            res = to_str(mat, width_str=width_str)
            # The header has the expected matrix value, so compare the two.
            lines = res.split("\n")
            headers = lines[1].split()
            vals = lines[2].replace("...", " ").replace("[", "").replace("]", "").split()
            self.assertEqual(headers, vals)


if __name__ == '__main__':
    unittest.main()
