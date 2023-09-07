# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause

import unittest
import warnings

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from matrepr import to_html, to_latex, to_str

np.random.seed(123)


def generate_fixed_value(m, n):
    row_factor = 10**(1+len(str(n)))
    data = []
    for r in range(m):
        data.append([1] * n)
        for c in range(n):
            data[r][c] = (r+1)*row_factor + c

    return torch.tensor(data, dtype=torch.int64), data


@unittest.skipIf(torch is None, "PyTorch not installed")
class PyTorchTests(unittest.TestCase):
    def setUp(self):
        # filter beta state warning
        warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")

        rand2d = torch.rand(50, 30)
        self.rand2d = rand2d[rand2d < 0.6] = 0

        rand3d = torch.rand(50, 30, 10)
        self.rand3d = rand3d[rand3d < 0.6] = 0

        self.tensors = [
            (True, torch.tensor(5)),
            (False, torch.tensor([])),
            (False, torch.tensor([1, 2, 3, 4])),
            (False, torch.tensor([[1, 2], [1003, 1004]])),
            (False, torch.tensor([[1, 2], [1003, 1004]]).to_sparse_coo()),
            (False, torch.tensor([[1, 2], [1003, 1004]]).to_sparse_csr()),
            (False, rand2d),
            (False, rand2d.to_sparse_coo()),
            (True, rand3d),
            (False, rand3d.to_sparse_coo()),
            (True, torch.tensor([[[0., 0], [1., 2.]], [[0., 0], [3., 4.]]]).to_sparse(2)),
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

    def test_2d_formats(self):
        to_html_args = dict(notebook=False, max_rows=20, max_cols=20, title=False)

        expected = [to_html(t.to_sparse_coo(), **to_html_args) for _, t in self.tensors]

        for converter in ["to_sparse_coo", "to_sparse_csr", "to_sparse_csc", "to_sparse_bsr", "to_sparse_bsc"]:
            for i, (flag, source_tensor) in enumerate(self.tensors):
                if flag or len(source_tensor.shape) != 2:
                    continue

                if "_bs" in converter:
                    if source_tensor.shape[0] > 10:
                        # The block-compressed tensor formats may have explicit zeros that COO and CSC/CSR won't.
                        # The smaller sample problems don't run into this, so skip the bigger ones.
                        continue
                    tensor = getattr(source_tensor, converter)(blocksize=2)
                else:
                    tensor = getattr(source_tensor, converter)()
                res = to_html(tensor, **to_html_args)
                self.assertEqual(expected[i], res)

    def test_contents_2d(self):
        source_tensor, data = generate_fixed_value(8, 8)
        for converter in ["to_dense", "to_sparse_coo", "to_sparse_csr", "to_sparse_csc",
                          "to_sparse_bsr", "to_sparse_bsc"]:
            if "_bs" in converter:
                tensor = getattr(source_tensor, converter)(blocksize=2)
            else:
                tensor = getattr(source_tensor, converter)()

            res = to_html(tensor, notebook=False, max_rows=20, max_cols=20, title=True, indices=True)
            for row in data:
                for value in row:
                    self.assertIn(f"<td>{value}</td>", res)

            trunc = to_html(tensor, notebook=False, max_rows=5, max_cols=5, title=True, indices=True)
            for value in (data[0][0], data[0][-1], data[-1][0], data[-1][-1]):
                self.assertIn(f"<td>{value}</td>", trunc)


if __name__ == '__main__':
    unittest.main()
