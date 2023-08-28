# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause

import unittest

try:
    import sparse
except ImportError:
    sparse = None

from matrepr import to_html, to_latex, to_str
from .test_html import generate_fixed_value

import numpy.random
numpy.random.seed(123)


@unittest.skipIf(sparse is None, "pydata/sparse not installed")
class PyDataSparseTests(unittest.TestCase):
    def setUp(self):
        self.mats = [
            sparse.COO([], shape=(0,)),
            sparse.COO(coords=[1, 4], data=[11, 44], shape=(10,)),
            sparse.COO([], shape=(0, 0)),
            sparse.COO([], shape=(10, 10)),
            sparse.random((10, 10), density=0.4),
            sparse.COO.from_scipy_sparse(generate_fixed_value(10, 10)),
            sparse.COO(coords=[[0, 0], [0, 0]], data=[111, 222], shape=(13, 13)),
            sparse.COO(coords=[[0, 1], [3, 2], [1, 3]], data=[111, 222], shape=(5, 5, 5)),
        ]

        self.types = [
            sparse.COO,
            sparse.DOK,
            sparse.GCXS,
        ]

    def test_type_registration(self):
        from matrepr.adapters.sparse_driver import PyDataSparseDriver
        reg_types = [
            typ for typ, _ in PyDataSparseDriver.get_supported_types()
        ]

        for tp in self.types:
            self.assertIn(f"{tp.__module__}.{tp.__name__}", reg_types)

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

        for tp in self.types:
            for i, source_mat in enumerate(self.mats):
                if 13 in source_mat.shape:
                    # this matrix has dupes, which act differently in different formats
                    continue
                mat = tp(source_mat)
                res = to_html(mat, **to_html_args)
                self.assertEqual(expected[i], res)

    def test_contents_1d(self):
        values = [1000, 1001, 1002, 1003, 1004]
        vec = sparse.COO([0, 1, 2, 3, 4], data=values, shape=(10,))
        res = to_html(vec, notebook=False, max_rows=20, max_cols=20, title=True, indices=True)
        for value in values:
            self.assertIn(f"<td>{value}</td>", res)

    def test_truncate_1d(self):
        values = [1000, 1001, 1002, 1003, 1009]
        vec = sparse.COO([0, 1, 2, 3, 9], data=values, shape=(10,))
        res = to_html(vec, notebook=False, max_rows=3, max_cols=3, num_after_dots=1, title=True, indices=True)
        for value in [1000, 1009]:
            self.assertIn(f"<td>{value}</td>", res)

    def test_contents_2d(self):
        mat = generate_fixed_value(10, 10)
        sparse_mat = sparse.COO.from_scipy_sparse(mat)
        res = to_html(sparse_mat, notebook=False, max_rows=20, max_cols=20, title=True, indices=True)
        for value in mat.data:
            self.assertIn(f"<td>{value}</td>", res)

    def test_truncate_2d(self):
        mat = generate_fixed_value(20, 20)
        sparse_mat = sparse.COO.from_scipy_sparse(mat)

        for after_dots, expected_count in [
            (0, 25),  # 5*5
            (1, 25),  # 4*4 + 4 + 4 + 1
            (2, 25),  # 3*3 + 2*3 + 3*2 + 2*2
        ]:
            res = to_html(sparse_mat, notebook=False, max_rows=6, max_cols=6, num_after_dots=after_dots)
            count = 0
            for value in mat.data:
                if f"<td>{value}</td>" in res:
                    count += 1
            self.assertEqual(expected_count, count)

    def test_contents_3d(self):
        values = [111, 222]
        mat = sparse.COO(coords=[[0, 1], [3, 2], [1, 3]], data=values, shape=(5, 5, 5))
        res = to_html(mat, notebook=False, max_rows=20, max_cols=20, title=True, indices=True)
        res_str = to_str(mat)
        for value in values:
            self.assertIn(f"<td>{value}</td>", res)
            self.assertIn(f"{value}", res_str)
        self.assertIn(f"<th>val</th>", res)
        self.assertIn(f"val", res_str)

    def test_truncate_3d(self):
        values = [111, 222]
        mat = sparse.COO(coords=[[0, 1], [3, 2], [1, 3]], data=values, shape=(5, 5, 5))

        res = to_html(mat, notebook=False, max_rows=30, max_cols=3, num_after_dots=1)
        res_str = to_str(mat, max_rows=30, max_cols=3, num_after_dots=1)
        count = count_str = 0
        for value in values:
            if f"<td>{value}</td>" in res:
                count += 1
            if f"{value}" in res_str:
                count_str += 1
        self.assertEqual(len(values), count)
        self.assertEqual(len(values), count_str)

        self.assertIn(f"<th>val</th>", res)
        self.assertIn(f"val", res_str)

    def test_patch_sparse(self):
        source_mat = sparse.COO(coords=[1, 4, 6], data=[11, 44, 222], shape=(10,))

        # noinspection PyUnresolvedReferences
        import matrepr.patch.sparse

        for tp in self.types:
            mat = tp(source_mat)
            res = repr(mat)
            self.assertIn("222", res)
            self.assertIn("[", res)  # a character used by MatRepr


if __name__ == '__main__':
    unittest.main()
