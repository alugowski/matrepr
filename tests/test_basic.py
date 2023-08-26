# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause

import unittest

import matrepr


class JupyterImportTests(unittest.TestCase):
    def test_import_jupyter(self):
        # noinspection PyUnresolvedReferences
        import matrepr.jupyter
        self.assertEqual(True, True)  # did not raise

    def test_import_jupyter_html(self):
        # noinspection PyUnresolvedReferences
        import matrepr.jupyter_html
        self.assertEqual(True, True)  # did not raise

    def test_import_jupyter_latex(self):
        # noinspection PyUnresolvedReferences
        import matrepr.jupyter_latex
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
            matrepr._get_adapter(set())

    def test_arguments(self):
        mat = [[1, 2], [1003, 1004]]
        with self.assertRaises(ValueError):
            matrepr.to_html(mat, cell_align="foobar")

        res = matrepr.to_latex(mat, title="TITLE", title_latex=None)
        self.assertIn("TITLE", res)

        res = matrepr.to_latex(mat, title=None, title_latex="TITLE")
        self.assertIn("TITLE", res)

    def test_driver_registration_notify(self):
        from unittest.mock import MagicMock

        callback = MagicMock()

        matrepr._driver_registration_notify.append(callback)
        self.assertFalse(callback.called)

        class MockDriver(matrepr.Driver):
            @staticmethod
            def get_supported_types():
                return []

            @staticmethod
            def adapt(_):
                return None

        matrepr.register_driver(MockDriver)
        self.assertTrue(callback.called)


if __name__ == '__main__':
    unittest.main()
