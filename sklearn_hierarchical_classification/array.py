"""Helpers for workings with sequences and (numpy) arrays."""
from itertools import chain

import numpy as np
from scipy.sparse import csr_matrix, issparse


def flatten_list(lst):
    """Flatten down a list-of-lists to a list with all elements of child lists expanded.

    This does *not* work recursively, only on 1-level deep list containment.

    Example:

    >>> flatten_list([[0], [1, 2], [3, 4]])
    [0, 1, 2, 3, 4]

    """
    return list(chain(*lst))


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


def apply_rollup_Xy(X, y):
    """
    Parameters
    ----------
    X : (sparse) array-like, shape = [n_samples, n_features]
        Data.

    y : list-of-lists - [n_samples]
        For each sample, y maintains list of labels this sample should be used for in training.

    Returns
    -------
    X_, y_
        Transformed by 'flattening' out y parameter and duplicating corresponding rows in X

    """
    # Compute number of rows we will have after transformation, which corresponds
    # to the total number of labels we have in y.
    n_rows = sum(len(labelset) for labelset in y)

    if n_rows == X.shape[0]:
        # This will happen when we have exactly one label in y per row,
        # coresponding to the non-multi-label scenario. No expansion needed,
        # simply flatten out y by transforming it to a list-of-labels instead of list-of-lists.
        return X, flatten_list(y)

    if not isinstance(X, csr_matrix):
        # Performance improvements require csr matrix
        X = csr_matrix(X)

    indptr = np.zeros((n_rows+1), dtype=np.int32)
    indices = []
    data = []

    indices_count = 0
    offset = 0

    # Our goal is to expand the equal labelsets into their own row within X
    # We do this by repeating each row exactly "labelset" times
    for i, labelset in enumerate(y):
        labelset_sz = len(labelset)
        for j in range(labelset_sz):
            indptr[offset+j] = indices_count

            indices.append(X.indices[X.indptr[i]:X.indptr[i+1]])
            data.append(X.data[X.indptr[i]:X.indptr[i+1]])

            indices_count += len(X.data[X.indptr[i]:X.indptr[i+1]])

        offset += labelset_sz

    indptr[-1] = indices_count

    indices = np.concatenate(indices)
    data = np.concatenate(data)

    X_ = csr_matrix(
        (data, indices, indptr),
        shape=(n_rows, X.shape[1]),
        dtype=X.dtype,
    )
    y_ = flatten_list(y)

    return X_, y_


def apply_rollup_Xy_raw(X, y):
    """
    Similar to `apply_rollup_Xy`, but for when X is the raw data matrix (E.g. 1D list of texts or raw image data).

    Parameters
    ----------
    X : List

    y : list-of-lists - [n_samples]
        For each sample, y maintains list of labels this sample should be used for in training.

    Returns
    -------
    X_, y_
        Transformed by 'flattening' out y parameter and duplicating corresponding rows in X

    """
    # Compute number of rows we will have after transformation
    n_rows = sum(len(labelset) for labelset in y)

    if n_rows == X.shape[0]:
        # No expansion needed
        return X, flatten_list(y)

    # Our goal is to expand the equal labelsets into their own row within X
    # We do this by repeating each row exactly "labelset" times
    X_rows = []
    for i, labelset in enumerate(y):
        labelset_sz = len(labelset)
        for j in range(labelset_sz):
            X_rows.append(X[j])

    y_ = flatten_list(y)
    return X_rows, y_


def extract_rows_csr(matrix, rows):
    """
    Parameters
    ----------
    matrix : (sparse) csr_matrix

    rows : list of row ids

    Returns
    -------
    matrix_: (sparse) csr_matrix
        Transformed by extracting the desired rows from `matrix`

    """
    if not isinstance(matrix, csr_matrix):
        matrix = csr_matrix(matrix)

    # Short circuit if we want a blank matrix
    if len(rows) == 0:
        return csr_matrix(matrix.shape)

    # Keep a record of the desired rows
    indptr = np.zeros(matrix.indptr.shape, dtype=np.int32)
    indices = []
    data = []

    # Keep track of the current index pointer
    indices_count = 0

    for i in range(matrix.shape[0]):
        indptr[i] = indices_count

        if i in rows:
            indices.append(matrix.indices[matrix.indptr[i]:matrix.indptr[i+1]])
            data.append(matrix.data[matrix.indptr[i]:matrix.indptr[i+1]])
            indices_count += len(matrix.data[matrix.indptr[i]:matrix.indptr[i+1]])

    indptr[-1] = indices_count

    indices = np.concatenate(indices)
    data = np.concatenate(data)

    return csr_matrix((data, indices, indptr), shape=matrix.shape)


def nnz_rows_ix(X):
    """Return row indices which have at least one non-zero column value."""
    return np.unique(X.nonzero()[0])


def nnz_columns_count(X):
    """Return count of columns which have at least one non-zero value."""
    return len(np.count_nonzero(X, axis=0))
