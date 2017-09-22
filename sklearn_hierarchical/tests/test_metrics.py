"""Unit-tests for the evaluation metrics module."""
import numpy as np
from hamcrest import (
    assert_that,
    close_to,
    is_,
)

from sklearn_hierarchical.metrics import (
    h_fbeta_score,
    h_precision_score,
    h_recall_score,
)
from sklearn_hierarchical.tests.fixtures import METRICS_TEST_CASES


def test_h_scores():
    """
    Test the hR, hP, hF1 metrics on a few synthetic data test cases.

    """
    for (G, y_shape, ix_true, ix_pred, expected_hr_score, expected_hp_score, expected_hf1_score) in METRICS_TEST_CASES:
        y_true = np.zeros(y_shape)
        y_true[ix_true] = 1
        y_pred = np.zeros(y_shape)
        y_pred[ix_pred] = 1

        assert_that(
            h_recall_score(y_true=y_true, y_pred=y_pred, class_hierarchy=G),
            is_(close_to(expected_hr_score, delta=0.0001)),
        )
        assert_that(
            h_precision_score(y_true=y_true, y_pred=y_pred, class_hierarchy=G),
            is_(close_to(expected_hp_score, delta=0.0001)),
        )
        assert_that(
            h_fbeta_score(y_true=y_true, y_pred=y_pred, class_hierarchy=G),
            is_(close_to(expected_hf1_score, delta=0.0001)),
        )
