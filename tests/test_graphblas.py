# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause

import unittest

from matrepr import to_html, to_latex, to_str
import matrepr


try:
    import graphblas as gb

    # Context initialization must happen before any other imports
    gb.init("suitesparse", blocking=False)

    have_gb = True
except ImportError:
    have_gb = False
    gb = None


def generate_fixed_value(m, n):
    row_factor = 10**(1+len(str(n)))
    nnz = m*n
    rows, cols, data = [1] * nnz, [1] * nnz, [1] * nnz
    for i in range(nnz):
        r = int(i / n)
        c = i % n
        rows[i] = r
        cols[i] = c
        data[i] = (r+1)*row_factor + c

    return gb.Matrix.from_coo(rows, cols, data, nrows=m, ncols=n, dtype='int64'), data


@unittest.skipIf(not have_gb, "python-graphblas not installed")
class GraphBLASMatrixTests(unittest.TestCase):
    def setUp(self):
        mat = gb.Matrix.from_coo([0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], nrows=5, ncols=5)
        self.mats = [
            mat,
            mat @ mat,  # an expression class
        ]

    def test_no_crash(self):
        for mat in self.mats:
            res = to_html(mat, title=True)
            self.assertGreater(len(res), 10)

            res = to_latex(mat, title=True)
            self.assertGreater(len(res), 10)

            res = to_str(mat, title=True)
            self.assertGreater(len(res), 10)

    def test_shape(self):
        mat = gb.Matrix.from_coo([0, 1, 2, 3, 4], [0, 0, 0, 0, 0], [0, 1, 2, 3, 4], nrows=5, ncols=1)
        adapter = matrepr._get_adapter(mat, None)
        self.assertEqual((5, 1), adapter.get_shape())

    def test_contents(self):
        gb_mat, data = generate_fixed_value(10, 10)
        res = to_html(gb_mat, notebook=False, max_rows=20, max_cols=20, title=True, indices=True)
        for value in data:
            self.assertIn(f"<td>{value}</td>", res)

    def test_truncate(self):
        gb_mat, data = generate_fixed_value(20, 20)

        for after_dots, expected_count in [
            (0, 25),  # 5*5
            (1, 25),  # 4*4 + 4 + 4 + 1
            (2, 25),  # 3*3 + 2*3 + 3*2 + 2*2
        ]:
            res = to_html(gb_mat, notebook=False, max_rows=6, max_cols=6, num_after_dots=after_dots)
            count = 0
            for value in data:
                if f"<td>{value}</td>" in res:
                    count += 1
            self.assertEqual(expected_count, count)


@unittest.skipIf(not have_gb, "python-graphblas not installed")
class GraphBLASVectorTests(unittest.TestCase):
    def setUp(self):
        vec = gb.Vector.from_coo([0, 3, 4, 6], [12.1, -5.4, 2.9, 2.2], size=8)
        self.vecs = [
            vec,
            vec + vec,  # an expression
        ]

    def test_no_crash(self):
        for vec in self.vecs:
            res = to_html(vec, title=True)
            self.assertGreater(len(res), 10)

            res = to_latex(vec, title=True)
            self.assertGreater(len(res), 10)

            res = to_str(vec, title=True)
            self.assertGreater(len(res), 10)

    def test_shape(self):
        vec = gb.Vector.from_coo([0, 3, 4, 6], [12.1, -5.4, 2.9, 2.2], size=8)
        adapter = matrepr._get_adapter(vec, None)
        self.assertEqual((8,), adapter.get_shape())

    def test_contents(self):
        values = [1000, 1001, 1002, 1003, 1004]
        vec = gb.Vector.from_coo([0, 1, 2, 3, 4], values, size=10)
        res = to_html(vec, notebook=False, max_rows=20, max_cols=20, title=True, indices=True)
        for value in values:
            self.assertIn(f"<td>{value}</td>", res)

    def test_truncate(self):
        values = [1000, 1001, 1002, 1003, 1004]
        vec = gb.Vector.from_coo([0, 1, 2, 3, 4], values, size=5)
        res = to_html(vec, notebook=False, max_rows=3, max_cols=3, num_after_dots=1, title=True, indices=True)
        for value in [1000, 1004]:
            self.assertIn(f"<td>{value}</td>", res)


@unittest.skipIf(not have_gb, "python-graphblas not installed")
class PatchGraphBLASTests(unittest.TestCase):
    def test_patch_graphblas(self):
        mat = gb.Matrix.from_coo([0, 1], [0, 1], [111, 222], nrows=5, ncols=5),
        vec = gb.Vector.from_coo([0, 1], [111, 222], size=8)

        # noinspection PyUnresolvedReferences
        import matrepr.patch.graphblas

        res = repr(mat)
        self.assertIn("222", res)
        self.assertIn("┌", res)  # a character used by MatRepr

        res = repr(vec)
        self.assertIn("222", res)
        self.assertIn("[", res)  # a character used by MatRepr

    def test_type_registration(self):
        from matrepr.adapters.graphblas_driver import GraphBLASDriver
        reg_types = [
            typ for typ, _ in GraphBLASDriver.get_supported_types()
        ]

        for tp in [gb.Matrix, gb.Vector]:
            self.assertIn(f"{tp.__module__}.{tp.__name__}", reg_types)


if __name__ == '__main__':
    unittest.main()
