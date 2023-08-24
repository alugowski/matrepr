# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause

import unittest

import numpy.random
import scipy.sparse

from matrepr import to_html, to_latex, to_str
from .test_html import generate_fixed_value

numpy.random.seed(123)


class SciPyTests(unittest.TestCase):
    def setUp(self):
        self.mats = [
            scipy.sparse.coo_matrix(([], ([], [])), shape=(0, 0)),
            scipy.sparse.random(10, 10, density=0.4).tocoo(),
            scipy.sparse.random(5, 10, density=0.4).tocsr(),
            scipy.sparse.random(5, 1, density=0.4).tocsc(),
            scipy.sparse.coo_matrix(([], ([], [])), shape=(10, 10)).tocoo(),
            scipy.sparse.coo_matrix(([], ([], [])), shape=(10, 10)).tocsr(),
            scipy.sparse.coo_matrix(([], ([], [])), shape=(10, 10)).tocsc(),
            generate_fixed_value(10, 10),
            scipy.sparse.coo_matrix(([111, 222], ([0, 0], [0, 0])), shape=(13, 13))
        ]

    def test_no_crash(self):
        for mat in self.mats:
            res = to_html(mat)
            self.assertGreater(len(res), 10)

            res = to_latex(mat)
            self.assertGreater(len(res), 10)

            res = to_str(mat)
            self.assertGreater(len(res), 10)

    def test_formats(self):
        to_html_args = dict(notebook=False, max_rows=20, max_cols=20, title=False)

        expected = [to_html(m, **to_html_args) for m in self.mats]

        for fmt in ["coo", "csr", "csc", "dok", "lil"]:
            for i, source_mat in enumerate(self.mats):
                if 13 in source_mat.shape:
                    # this matrix has dupes, which act differently in different formats
                    continue
                mat = source_mat.asformat(fmt)
                res = to_html(mat, **to_html_args)
                self.assertEqual(expected[i], res)


class PatchSciPyTests(unittest.TestCase):
    def test_patch_scipy(self):
        source_mat = scipy.sparse.coo_matrix(([111, 222], ([0, 1], [0, 1])), shape=(10, 10))

        # noinspection PyUnresolvedReferences
        import matrepr.patch.scipy

        for fmt in ["coo", "csr", "csc", "dok", "lil"]:
            mat = source_mat.asformat(fmt)
            res = repr(mat)
            self.assertIn("222", res)
            self.assertIn("┌", res)  # a character used by MatRepr


if __name__ == '__main__':
    unittest.main()
