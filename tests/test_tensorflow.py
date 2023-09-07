# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause

import unittest
import warnings

try:
    # Suppress warning from inside tensorflow
    warnings.filterwarnings("ignore", message="module 'sre_constants' is deprecated")
    import tensorflow as tf

    tf.random.set_seed(1234)
except ImportError:
    tf = None

from matrepr import to_html, to_latex, to_str


def generate_fixed_value(m, n):
    row_factor = 10**(1+len(str(n)))
    data = []
    for r in range(m):
        data.append([1] * n)
        for c in range(n):
            data[r][c] = (r+1)*row_factor + c

    return tf.constant(data, dtype=tf.int64), data


@unittest.skipIf(tf is None, "TensorFlow not installed")
class TensorFlowTests(unittest.TestCase):
    def setUp(self):
        rand1d = tf.random.uniform(shape=(50,)).numpy()
        rand1d[rand1d < 0.6] = 0
        self.rand1d = tf.convert_to_tensor(rand1d)

        rand2d = tf.random.uniform(shape=(50, 30)).numpy()
        rand2d[rand2d < 0.6] = 0
        self.rand2d = tf.convert_to_tensor(rand2d)

        rand3d = tf.random.uniform(shape=(50, 30, 10)).numpy()
        rand3d[rand3d < 0.6] = 0
        self.rand3d = tf.convert_to_tensor(rand3d)

        self.tensors = [
            (True, tf.constant(5)),
            (False, tf.constant([])),
            (False, tf.constant([1, 2, 3, 4])),
            (False, tf.constant([[1, 2], [1003, 1004]])),
            (False, tf.sparse.from_dense(tf.constant([[1, 2], [1003, 1004]]))),
            (False, self.rand1d),
            (False, tf.sparse.from_dense(self.rand1d)),
            (False, self.rand2d),
            (False, tf.sparse.from_dense(self.rand2d)),
            (True, self.rand3d),
            (False, tf.sparse.from_dense(self.rand3d)),
            (False, tf.sparse.SparseTensor(indices=[[0, 3], [2, 4]], values=[10, 20], dense_shape=[3, 10])),
        ]

    def test_no_crash(self):
        for fallback_ok, tensor in self.tensors:
            res = to_str(tensor, title=True)
            self.assertGreater(len(res), 5)

            res = to_html(tensor, title=True)
            self.assertGreater(len(res), 5)
            if not fallback_ok:
                self.assertNotIn("<pre>", res)

            res = to_latex(tensor, title=True)
            self.assertGreater(len(res), 5)

    def test_contents_2d(self):
        source_tensor, data = generate_fixed_value(8, 8)
        for to_sparse in (False, True):
            tensor = tf.sparse.from_dense(source_tensor) if to_sparse else source_tensor

            res = to_html(tensor, notebook=False, max_rows=20, max_cols=20, title=True, indices=True)
            for row in data:
                for value in row:
                    self.assertIn(f"<td>{value}</td>", res)

            trunc = to_html(tensor, notebook=False, max_rows=5, max_cols=5, title=True, indices=True)
            for value in (data[0][0], data[0][-1], data[-1][0], data[-1][-1]):
                self.assertIn(f"<td>{value}</td>", trunc)


if __name__ == '__main__':
    unittest.main()
