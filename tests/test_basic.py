# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause

import unittest

import matrepr


class JupyterExtensionImportTests(unittest.TestCase):
    def test_import_jupyter(self):
        matrepr.load_ipython_extension(None)
        matrepr.unload_ipython_extension(None)
        self.assertEqual(True, True)  # did not raise

    def test_import_jupyter_html(self):
        import matrepr.html
        matrepr.html.load_ipython_extension(None)
        matrepr.html.unload_ipython_extension(None)
        self.assertEqual(True, True)  # did not raise

    def test_import_jupyter_latex(self):
        import matrepr.latex
        matrepr.latex.load_ipython_extension(None)
        matrepr.latex.unload_ipython_extension(None)
        self.assertEqual(True, True)  # did not raise


class BasicTests(unittest.TestCase):
    def test_no_crash_mdisplay(self):
        try:
            import IPython
        except ImportError:
            self.skipTest("no Jupyter")
            return

        mat = [[1, 2], [1003, 1004]]
        matrepr.mdisplay(mat, "html")
        matrepr.mdisplay(mat, "latex")
        matrepr.mdisplay(mat, "str")

        with self.assertRaises(ValueError):
            matrepr.mdisplay(mat, "foobar")

    def test_no_rash_mprint(self):
        mat = [[1000, 1001]]
        matrepr.mprint(mat)
        self.assertEqual(True, True)  # did not raise

    def test_adaptation_errors(self):
        with self.assertRaises(AttributeError):
            matrepr._get_adapter(set(), None)

    def test_arguments(self):
        mat = [[1, 2], [1003, 1004]]
        with self.assertRaises(ValueError):
            matrepr.to_html(mat, cell_align="foobar")

        res = matrepr.to_latex(mat, title="TITLE", title_latex=None)
        self.assertIn("TITLE", res)

        res = matrepr.to_latex(mat, title=None, title_latex="TITLE")
        self.assertIn("TITLE", res)


if __name__ == '__main__':
    unittest.main()
