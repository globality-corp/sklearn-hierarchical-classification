"""Helpers for workings with sequences and (numpy) arrays."""
import numpy as np
from scipy.sparse import issparse


def flatten_list(lst):
    return [
        item
        for sublist in lst
        for item in sublist
    ]


def apply_along_rows(func, X):
    """
    Apply function row-wise to input matrix X.
    This will work for dense matrices (eg np.ndarray)
    as well as for CSR sparse matrices.

    """
    if issparse(X):
        return np.array([
            func(X.getrow(i))
            for i in range(X.shape[0])
        ])
    else:
        # XXX might break vis-a-vis this issue merging: https://github.com/numpy/numpy/pull/8511
        # See discussion over issue with truncated string when using np.apply_along_axis here:
        #   https://github.com/numpy/numpy/issues/8352
        return np.ma.apply_along_axis(
            lambda x: func(x.reshape(1, -1)),
            axis=1,
            arr=X,
        )


def nnz_rows_ix(X):
    """Return row indices which have at least one non-zero column value."""
    return np.unique(X.nonzero()[0])


def nnz_columns_count(X):
    """Return count of columns which have at least one non-zero value."""
    return len(np.count_nonzero(X, axis=0))
