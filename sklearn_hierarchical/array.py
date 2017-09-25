"""Helpers for workings with (numpy) arrays."""
import numpy as np
from scipy.sparse import issparse


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
        return np.apply_along_axis(
            lambda x: func(x.reshape(1, -1)),
            axis=1,
            arr=X,
        )


def nnz_rows_ix(X):
    """
    Return row indices which have at least one non-zero column value.

    """
    return np.unique(X.nonzero()[0])
