import numpy as np
from hamcrest import (
    assert_that,
    equal_to,
    is_,
)

from sklearn_hierarchical_classification.array import apply_rollup_Xy


def test_apply_rollup_xy():
    X = np.arange(9).reshape(3, 3)
    y_rolled_up = [
        [0, 1],
        [2],
        [3, 4, 5],
    ]

    X_, y_ = apply_rollup_Xy(X, y_rolled_up)

    assert_that((X_[0] != X_[1]).nnz, is_(equal_to(0)))
    assert_that((X_[3] != X_[4]).nnz, is_(equal_to(0)))
    assert_that((X_[4] != X_[5]).nnz, is_(equal_to(0)))

    for i in range(6):
        assert_that(y_[i], is_(equal_to(i)))
