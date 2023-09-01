# Copyright (C) 2023 Adam Lugowski.
# Use of this source code is governed by the BSD 2-clause license found in the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause

from abc import ABC, abstractmethod
from typing import Any, List, Iterable, Mapping, Optional, Tuple, Union


def describe(shape: tuple = None, nnz: int = None, nz_type=None, layout: str = None, notes: str = None) -> str:
    """
    Create a simple description string from potentially interesting pieces of metadata.
    """
    parts = []
    by = chr(215)  # ×
    if len(shape) == 1:
        parts.append(f"length={shape[0]}")
    elif len(shape) == 2:
        parts.append(f"{shape[0]}{by}{shape[1]}")
    else:
        parts.append(f"shape=({','.join(str(d) for d in shape)})")

    if nnz is not None:
        dtype_str = f" '{str(nz_type)}'" if nz_type else ""
        parts.append(f"{nnz}{dtype_str} elements")

    if layout is not None:
        parts.append(str(layout))

    if notes:
        parts.append(notes)

    return ", ".join(parts)


class DupeList(list):
    """
    A list but a different type to distinguish between a list in the original data
    and a list created to handle duplicate entries.
    """

    def __init__(self, iterable):
        super().__init__(iterable)


class MatrixAdapter(ABC):
    def __init__(self):
        self.row_labels: Union[Mapping[int, Any], bool] = True
        """
        A mapping from row index to row label. If True, then the label is the index.
        """
        self.col_labels: Union[Mapping[int, Any], bool] = True
        """
        A mapping from col index to col label. If True, then the label is the index.
        """

    @abstractmethod
    def describe(self) -> str:
        """
        Return a human-readable description of the matrix, usable as a title.
        """

    @abstractmethod
    def get_shape(self) -> tuple:
        """
        Return the shape of the matrix.
        """

    def has_row_labels(self) -> bool:
        return self.row_labels is not False and self.row_labels is not None and self.get_shape()[0] > 0

    def has_col_labels(self) -> bool:
        return self.col_labels is not False and self.col_labels is not None

    @staticmethod
    def _label_impl(idx, labels) -> Optional[Any]:
        if labels is None or labels is False:  # can't do `if not labels` due to gb.Vector
            return None
        if labels is True:
            return idx

        ret = labels[idx]
        if hasattr(ret, "value"):
            ret = ret.value
        return ret

    def get_row_label(self, idx: int) -> Optional[Any]:
        return MatrixAdapter._label_impl(idx, self.row_labels)

    def get_col_label(self, idx: int) -> Optional[Any]:
        return MatrixAdapter._label_impl(idx, self.col_labels)

    # noinspection PyMethodMayBeStatic
    def is_tensor(self) -> bool:
        return False


class MatrixAdapterRow(MatrixAdapter):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_row(self, row_idx: int, col_range: Tuple[int, int]) -> Iterable[Tuple[int, Any]]:
        """
        Extract a portion of single row from the matrix, as sparse tuples.

        :param row_idx: index of row to fetch
        :param col_range: half-open range of column indices to return
        :return: an iterable of `(index, value)` tuples
        """

    def get_dense_row(self, row_idx: int, col_range: Tuple[int, int]) -> Iterable[Any]:
        """
        Extract a portion of single row from the matrix, as a dense array.

        :param row_idx: index of row to fetch
        :param col_range: half-open range of columns to return
        :return: an iterable of length `col_range[1] - col_range[0]`
        """
        ret: List[Any] = [None] * (col_range[1] - col_range[0])
        for idx, value in self.get_row(row_idx, col_range):
            idx = idx - col_range[0]
            if ret[idx] is None:
                ret[idx] = value
            elif isinstance(ret[idx], DupeList):
                ret[idx].append(value)
            else:
                ret[idx] = DupeList([ret[idx], value])
        return ret


class MatrixAdapterCol(MatrixAdapter):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_col(self, col_idx: int, row_range: Tuple[int, int]) -> Iterable[Tuple[int, Any]]:
        """
        Extract a portion of single column from the matrix, as sparse tuples.

        :param col_idx: index of column to fetch
        :param row_range: half-open range of rows to return
        :return: an iterable of `(index, value)` tuples
        """


class MatrixAdapterCoo(MatrixAdapter):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_coo(self, row_range: Tuple[int, int], col_range: Tuple[int, int]) -> Iterable[Tuple[int, int, Any]]:
        """
        Extract a portion of the matrix, as sparse tuples.

        :param row_range: half-open range of rows to return
        :param col_range: half-open range of columns to return
        :return: an iterable of `(row_index, column_index, value)` tuples
        """


class OverrideColumnLabels:
    def __init__(self, overrides):
        self.overrides = overrides

    def __getitem__(self, item):
        if item in self.overrides:
            return self.overrides[item]
        return item


class TensorAdapterCooRow(MatrixAdapterRow):
    def __init__(self):
        super().__init__()
        self.col_labels = OverrideColumnLabels({self.get_shape()[1] - 1: "val"})

    """
    Present an n-dimensional tensor as a 2D table where each row is a coordinate tuple of length n+1:
    coordinates and value.
    """
    @abstractmethod
    def get_shape(self) -> Tuple[int, int]:
        """
        Return the shape of the tabular view of the tensor.

        :returns: a tuple: (nnz, number of dimensions+1)
        """

    @abstractmethod
    def get_dense_row(self, row_idx: int, col_range: Tuple[int, int]) -> Iterable[Any]:
        """
        Extract a single tuple from the tensor, as a dense array.

        :param row_idx: index of tensor tuple to fetch
        :param col_range: half-open range of dimensions to return
        :return: an iterable of length `col_range[1] - col_range[0]`
        """

    def get_row(self, row_idx: int, col_range: Tuple[int, int]) -> Iterable[Tuple[int, Any]]:
        return enumerate(self.get_dense_row(row_idx, col_range), start=col_range[0])

    # noinspection PyMethodMayBeStatic
    def is_tensor(self) -> bool:
        return True


class Driver(ABC):
    @staticmethod
    @abstractmethod
    def get_supported_types() -> Iterable[Tuple[str, bool]]:
        """
        Declares the types that this :class:`Driver` supports, and whether they should be registered with Jupyter.

        Does not import the modules that it supports.

        :rtype: (str, bool)
        :returns: An iterable of `(type_as_str, should_register_with_Jupyter_bool)` tuples.
        """

    @staticmethod
    @abstractmethod
    def adapt(mat: Any) -> Optional[MatrixAdapter]:
        """
        Return a :class:`MatrixAdapter` for a supported matrix.
        """


class Truncated2DMatrix(MatrixAdapterRow):
    """
    An intermediary class used to present a dense view of an adapted matrix.

    - Convert any adapter to a :class:`MatrixAdapterRow`.
    - If a matrix is too large then supports showing just the corners with ellipses designating the truncated portions.
    """
    def __init__(self, orig_shape: Tuple[int, int], display_shape: Tuple[int, int],
                 row_labels, col_labels, num_after_dots=2, description=None):
        super().__init__()
        self.show_row_labels = len(orig_shape) != 1
        if len(orig_shape) == 1:
            orig_shape = (1, orig_shape[0])
        elif len(orig_shape) > 2:
            raise ValueError
        self.orig_shape = orig_shape
        self.display_shape = [min(orig_shape[0], display_shape[0]), min(orig_shape[1], display_shape[1])]
        self.nrows, self.ncols = self.display_shape
        self.num_after_dots = num_after_dots
        self.elements: List[Any] = [[None] * self.ncols for _ in range(self.nrows)]
        self.row_labels = row_labels
        self.col_labels = col_labels
        self.description = description

        self.dot_col = None
        self.dot_row = None

        if self.display_shape[0] < self.orig_shape[0]:
            # need to truncate rows
            self.dot_row = self._calc_dots(self.display_shape[0], self.orig_shape[0])

        if self.display_shape[1] < self.orig_shape[1]:
            # need to truncate columns
            self.dot_col = self._calc_dots(self.display_shape[1], self.orig_shape[1])

    def _calc_dots(self, display, orig):
        if display >= orig:
            return orig

        if 0 < self.num_after_dots < 1:
            # fractional
            return int(display * self.num_after_dots)
        else:
            return max(0, display - 1 - int(self.num_after_dots))

    def describe(self) -> str:
        return self.description

    def get_shape(self):
        return self.display_shape

    def get_displayed_row_indices(self) -> List[int]:
        pre_dot_end, post_dot_start = self.get_dot_indices_row()
        return list(range(pre_dot_end)) + list(range(post_dot_start, self.orig_shape[0]))

    def get_displayed_col_indices(self) -> List[int]:
        pre_dot_end, post_dot_start = self.get_dot_indices_col()
        return list(range(pre_dot_end)) + list(range(post_dot_start, self.orig_shape[1]))

    def get_row_label(self, idx: int) -> Optional[Any]:
        if not self.show_row_labels or idx == self.dot_row:
            return None

        if self.dot_row is not None and idx > self.dot_row:
            _, post_dot_start = self.get_dot_indices_row()
            idx = post_dot_start + (idx - self.dot_row - 1)
        return super().get_row_label(idx)

    def get_col_label(self, idx: int) -> Optional[Any]:
        if idx == self.dot_col:
            return None

        if self.dot_col is not None and idx > self.dot_col:
            _, post_dot_start = self.get_dot_indices_col()
            idx = post_dot_start + (idx - self.dot_col - 1)
        return super().get_col_label(idx)

    def get_row(self, row_idx: int, col_range: Tuple[int, int]) -> Iterable[Tuple[int, Any]]:
        return enumerate(self.get_dense_row(row_idx, col_range), start=col_range[0])

    def get_dense_row(self, row_idx: int, col_range: Tuple[int, int]):
        return self.elements[row_idx][col_range[0]:col_range[1]]

    def set(self, row_idx: int, col_idx: int, value: Any):
        row_idx = int(row_idx)
        col_idx = int(col_idx)

        pre_dot_end, post_dot_start = self.get_dot_indices_row()
        if self.dot_row is not None and row_idx >= pre_dot_end:
            row_idx = self.dot_row + 1 + (row_idx - post_dot_start)
            if row_idx < self.dot_row:
                # within the dots
                return

        pre_dot_end, post_dot_start = self.get_dot_indices_col()
        if self.dot_col is not None and col_idx >= pre_dot_end:
            col_idx = self.dot_col + 1 + (col_idx - post_dot_start)
            if col_idx < self.dot_col:
                # within the dots
                return

        if self.elements[row_idx][col_idx] is None:
            self.elements[row_idx][col_idx] = value
        elif isinstance(self.elements[row_idx][col_idx], DupeList):
            self.elements[row_idx][col_idx].append(value)
        else:
            self.elements[row_idx][col_idx] = DupeList([self.elements[row_idx][col_idx], value])

    def apply_dots(self, dots):
        if self.dot_row is not None:
            for i in range(len(self.elements[self.dot_row])):
                self.elements[self.dot_row][i] = dots["v"]
        if self.dot_col is not None:
            for i in range(len(self.elements)):
                self.elements[i][self.dot_col] = dots["h"]
            if self.dot_row is not None:
                self.elements[self.dot_row][self.dot_col] = dots["d"]

    def get_dot_indices_row(self) -> Tuple[int, int]:
        if self.dot_row is None:
            return self.orig_shape[0], self.orig_shape[0]
        else:
            num_post_dot = self.display_shape[0] - 1 - self.dot_row
            return self.dot_row, self.orig_shape[0] - num_post_dot

    def get_dot_indices_col(self) -> Tuple[int, int]:
        if self.dot_col is None:
            return self.orig_shape[1], self.orig_shape[1]
        else:
            num_post_dot = self.display_shape[1] - 1 - self.dot_col
            return self.dot_col, self.orig_shape[1] - num_post_dot

    def drop_column(self):
        if self.display_shape[1] < 2:
            return self.display_shape[1]

        # determine which column to drop
        old_dot_col = self.dot_col
        self.dot_col = self._calc_dots(self.display_shape[1] - 1, self.orig_shape[1])

        if old_dot_col is None:
            if self.dot_col > 0:
                drop = self.dot_col - 1
            else:
                drop = self.dot_col
        else:
            assert self.dot_col <= old_dot_col
            if self.dot_col == 0:
                drop = self.dot_col
            elif self.dot_col < old_dot_col:
                drop = self.dot_col - 1
            else:
                drop = self.dot_col

        # adjust metadata
        self.display_shape[1] -= 1
        self.ncols -= 1

        # drop
        for row in self.elements:
            row.pop(drop)

        return self.display_shape[1]


def to_trunc(mat: MatrixAdapter, max_rows, max_cols, num_after_dots) -> Truncated2DMatrix:
    """
    Convert an adapted matrix to a :class:`Truncated2DMatrix`.
    """
    if isinstance(mat, Truncated2DMatrix):
        return mat

    shape = mat.get_shape()
    if len(shape) == 1:
        # vector
        nrows = 1
        ncols = mat.get_shape()[0]
    elif len(shape) == 2:
        nrows, ncols = mat.get_shape()
        max_rows = min(max_rows, nrows)
        max_cols = min(max_cols, ncols)
    else:
        raise ValueError("Only 1 or 2 dimensional matrices supported at this time.")

    if nrows == 0 or ncols == 0:
        return Truncated2DMatrix(orig_shape=mat.get_shape(),
                                 display_shape=(max_rows, max_cols),
                                 num_after_dots=0,
                                 row_labels=mat.row_labels,
                                 col_labels=mat.col_labels,
                                 description=mat.describe())

    if isinstance(mat, MatrixAdapterCoo):
        if max_rows >= nrows and max_cols >= ncols:
            num_after_dots = 0

        import time
        begin = time.time()
        top_left = list(mat.get_coo(row_range=(0, max_rows), col_range=(0, max_cols)))
        duration = time.time() - begin

        if duration > 0.1:
            # slow to get data, only show the top-left
            num_after_dots = 0

        if num_after_dots == 0:
            trunc = Truncated2DMatrix(orig_shape=mat.get_shape(),
                                      display_shape=(max_rows, max_cols),
                                      num_after_dots=0,
                                      row_labels=mat.row_labels,
                                      col_labels=mat.col_labels,
                                      description=mat.describe())
            for row, col, val in top_left:
                trunc.set(row, col, val)
        else:
            trunc = Truncated2DMatrix(orig_shape=mat.get_shape(),
                                      display_shape=(max_rows, max_cols),
                                      num_after_dots=num_after_dots,
                                      row_labels=mat.row_labels,
                                      col_labels=mat.col_labels,
                                      description=mat.describe())
            for row, col, val in top_left:
                trunc.set(row, col, val)

            # fetch the other three quadrants
            dot_row = trunc.dot_row if trunc.dot_row else max_rows
            dot_col = trunc.dot_col if trunc.dot_col else max_cols
            for row_range, col_range in [
                ((0, dot_row), (max(max_cols, dot_col + 1), ncols)),  # top-right
                ((max(max_rows, dot_row + 1), nrows), (0, dot_col)),  # bottom-left
                ((max(max_rows, dot_row + 1), nrows), (max(max_cols, dot_col + 1), ncols)),  # bottom-right
            ]:
                if row_range[1] - row_range[0] <= 0 or col_range[1] - col_range[0] <= 0:
                    continue

                for row, col, val in mat.get_coo(row_range=row_range, col_range=col_range):
                    trunc.set(row, col, val)
        return trunc

    if isinstance(mat, MatrixAdapterRow):
        trunc = Truncated2DMatrix(orig_shape=mat.get_shape(),
                                  display_shape=(max_rows, max_cols),
                                  num_after_dots=num_after_dots,
                                  row_labels=mat.row_labels,
                                  col_labels=mat.col_labels,
                                  description=mat.describe())

        pre_dot_end, post_dot_start = trunc.get_dot_indices_col()

        # fetch the pre-dot rows
        rows_to_fetch = [0] if len(shape) == 1 else trunc.get_displayed_row_indices()
        for row_idx in rows_to_fetch:
            if row_idx is None:
                # dots
                continue

            for col_range in [(0, pre_dot_end), (post_dot_start, ncols)]:
                for col_idx, value in mat.get_row(row_idx, col_range=col_range):
                    trunc.set(row_idx, col_idx, value)
        return trunc

    if isinstance(mat, MatrixAdapterCol):
        trunc = Truncated2DMatrix(orig_shape=mat.get_shape(),
                                  display_shape=(max_rows, max_cols),
                                  num_after_dots=num_after_dots,
                                  row_labels=mat.row_labels,
                                  col_labels=mat.col_labels,
                                  description=mat.describe())

        pre_dot_end, post_dot_start = trunc.get_dot_indices_row()

        # fetch the pre-dot rows
        for col_idx in trunc.get_displayed_col_indices():
            if col_idx is None:
                # dots
                continue

            for row_range in [(0, pre_dot_end), (post_dot_start, nrows)]:
                for row_idx, value in mat.get_col(col_idx, row_range=row_range):
                    trunc.set(row_idx, col_idx, value)
        return trunc

    raise NotImplementedError
