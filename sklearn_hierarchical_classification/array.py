"""Helpers for workings with sequences and (numpy) arrays."""
from itertools import chain

import numpy as np
from scipy.sparse import issparse, csr_matrix


def flatten_list(lst):
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
    # Compute number of rows we will have after transformation
    n_rows = sum(len(labelset) for labelset in y)

    if n_rows == X.shape[0]:
        # No expansion needed
        return X, flatten_list(y)

    # Performance improvements require csr matrix
    if not isinstance(X, csr_matrix):
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

    y_ = flatten_list(y)
    return csr_matrix((data, indices, indptr), shape=(n_rows, X.shape[1]), dtype=X.dtype), y_


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
