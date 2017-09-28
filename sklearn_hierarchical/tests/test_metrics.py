"""Unit-tests for the evaluation metrics module."""
import numpy as np
from hamcrest import (
    assert_that,
    close_to,
    is_,
)
from networkx import DiGraph, path_graph

from sklearn_hierarchical.metrics import (
    h_fbeta_score,
    h_precision_score,
    h_recall_score,
)


METRICS_TEST_CASES = [
    # G (directed): 0 -> 1 -> 2 -> 3
    # y_true:
    #
    #    [0, 1, 0, 0]
    #    [0, 1, 1, 0]
    #    [0, 0, 0, 1]
    #
    # y_pred:
    #
    #    [0, 1, 0, 0]
    #    [1, 1, 0, 0]
    #    [0, 1, 0, 0]
    #
    # Expected hR: 6 / 9 = 0.666
    # Expected hP: 6 / 6 = 1.0
    # Expected hF1: 6 / 6 = 1.0
    (path_graph(4, create_using=DiGraph()), (3, 4), ([0, 1, 1, 2], [1, 1, 2, 3]), ([0, 1, 1, 2], [1, 0, 1, 1]), 0.6666, 1.0, 0.8),  # noqa:E501
    # G (directed): 0 -> 1 -> 2 -> 3
    # y_true:
    #
    #    [0, 1, 0, 0]
    #    [0, 1, 1, 0]
    #    [1, 0, 0, 0]
    #
    # y_pred:
    #
    #    [0, 1, 0, 0]
    #    [1, 1, 0, 0]
    #    [0, 0, 0, 1]
    #
    # Expected hR: 5 / 6 = 0.8333
    # Expected hP: 5 / 8 = 0.625
    (path_graph(4, create_using=DiGraph()), (3, 4), ([0, 1, 1, 2], [1, 1, 2, 0]), ([0, 1, 1, 2], [1, 0, 1, 3]), 0.8333, 0.625, 0.7142),  # noqa:E501
]


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
