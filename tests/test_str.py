# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause

import unittest

import scipy.sparse

from matrepr import to_str
from matrepr.string_formatter import max_line_width
from .test_html import generate_fixed_value


class ToHTMLTests(unittest.TestCase):
    def setUp(self):
        self.mats = [
            scipy.sparse.random(10, 10, density=0.4),
            scipy.sparse.random(5, 10, density=0.4),
            scipy.sparse.random(5, 1, density=0.4),
            scipy.sparse.coo_matrix(([], ([], [])), shape=(10, 10)),
            generate_fixed_value(10, 10)
        ]

    def test_contents(self):
        mat = generate_fixed_value(9, 9)
        res = to_str(mat, max_rows=20, max_cols=20, title=True, indices=True)
        for value in mat.data:
            self.assertIn(f"{value}", res)
        for idx in range(10):
            self.assertIn(f"{idx}", res)

    def test_truncate(self):
        for fmt in ["csr", "csc", "coo"]:
            mat = generate_fixed_value(9, 9).asformat(fmt)

            for after_dots, expected_count in [
                (0, 25),  # 5*5
                (1, 25),  # 4*4 + 4 + 4 + 1
                (2, 25),  # 3*3 + 2*3 + 3*2 + 2*2
            ]:
                res = to_str(mat, max_rows=6, max_cols=6, num_after_dots=after_dots)
                count = 0
                for value in mat.data:
                    if f"{value}" in res:
                        count += 1
                self.assertEqual(expected_count, count)

    def test_width(self):
        mat = generate_fixed_value(9, 9)

        res = to_str(mat, max_cols=20, width_str=100, title=False, indices=False)
        self.assertLessEqual(max_line_width(res), 100)

        res = to_str(mat, max_cols=20, width_str=30, title=False, indices=False)
        self.assertLessEqual(max_line_width(res), 30)

        res = to_str(mat, max_cols=20, width_str=15, title=False, indices=True)
        self.assertLessEqual(max_line_width(res), 15)

        # test limit by max_cols
        res = to_str(mat, max_cols=2, width_str=200, title=False, indices=False)
        self.assertLessEqual(max_line_width(res), 20)

        # no crash test
        to_str(mat, max_cols=20, width_str=1, title=False, indices=True)
        to_str(mat, max_cols=20, width_str=0, title=False, indices=True)
        to_str(mat, max_cols=20, width_str=-1, title=False, indices=True)

    def test_title(self):
        mat = generate_fixed_value(4, 4)
        off = to_str(mat, title=False)
        self.assertNotIn("elements", off)

        on = to_str(mat, title=True)
        self.assertIn(f"{mat.nnz} 'int64' elements", on)

        title = "test title"
        custom = to_str(mat, title=title)
        self.assertIn(title, custom)


if __name__ == '__main__':
    unittest.main()
