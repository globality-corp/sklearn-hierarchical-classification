"""Unit-tests for the evaluation metrics module."""
import inflect
from hamcrest import assert_that, close_to, is_
from networkx import DiGraph, relabel_nodes
from parameterized import parameterized

from sklearn_hierarchical_classification.constants import ROOT
from sklearn_hierarchical_classification.metrics import (
    h_fbeta_score,
    h_precision_score,
    h_recall_score,
    multi_labeled,
)


def graph_fixture(as_str=False):
    """Sets up our fixture class hierarchy graph for the metrics unit-tests.
    This class hierarchy looks like this (directed graph):

                   R
                 /  \
                0    1
                    / \
                   2   3
                  / \
                 4   5
                     |
                     6

    Parameters
    ----------
    as_str : bool, default False
        If set to true, will return graph with node ids converted to strings using
        inflect library (e.g. 1 will be converted to the string literal 'one').

    """
    G = DiGraph()
    G.add_edges_from([
        (ROOT, 0),
        (ROOT, 1),
        (1, 2),
        (1, 3),
        (2, 4),
        (2, 5),
        (5, 6),
    ])

    if as_str:
        inflect_engine = inflect.engine()
        relabel_nodes(
            G,
            {
                node_id: inflect_engine.number_to_words(node_id)
                for node_id in G.nodes()
                if node_id != ROOT
            },
            copy=False,
        )

    return G


# Nb. the test cases can be multi-label so each y is an iterable of labels for a single example
METRICS_TEST_CASES = [
    # Test metrics for a single instance (labeled datapoint)
    # y_true: [[4]]
    # y_pred: [[3]]
    #
    # Expected hR: 1 / 3 = 0.333
    # Expected hP: 1 / 2 = 0.5
    # Expected hF1: 0.4
    (
        graph_fixture(),
        [[4]],
        [[3]],
        0.333333,
        0.5,
        0.4,
    ),

    # Test metrics for multiple instances (labeled datapoints)
    # y_true: [[4], [3]]
    # y_pred: [[3], [4]]
    #
    # Expected hR: (1 + 1) / (3 + 2) = 0.4
    # Expected hP: (1 + 1) / (3 + 2) = 0.4
    # Expected hF1: 0.4
    (
        graph_fixture(),
        [[4], [3]],
        [[3], [4]],
        0.4,
        0.4,
        0.4,
    ),
    # Test metrics for multiple instances (labeled datapoints) with multi-label ground truth, perfect match
    # y_true: [[0, 4], [1, 3]]
    # y_pred: [[0, 4], [1, 3]]
    #
    # Expected hR: 1.0
    # Expected hP: 1.0
    # Expected hF1: 1.0
    (
        graph_fixture(),
        [[0, 4], [1, 3]],
        [[0, 4], [1, 3]],
        1.0,
        1.0,
        1.0,
    ),
    # Test metrics for multiple instances (labeled datapoints) with multi-label ground truth
    # y_true: [[0, 4], [1, 3]]
    # y_pred: [[4, 6], [1]]
    #
    # Expected hR: (3 + 1) / (5 + 1) = 0.666...
    # Expected hP: (3 + 1) / (4 + 2) = 0.666...
    # Expected hF1: 0.666...
    (
        graph_fixture(),
        [[0, 4], [1, 3]],
        [[4, 6], [1]],
        0.6666,
        0.6666,
        0.6666,
    ),
    # Test metrics for multiple instances (labeled datapoints) with multi-label ground truth, string labels
    # y_true: [["one", "two"], ["one"]]
    # y_pred: [["one", "two"], ["one"]]
    #
    # Expected hR: 1.0
    # Expected hP: 1.0
    # Expected hF1: 1.0
    (
        graph_fixture(as_str=True),
        [["one", "two"], ["one"]],
        [["one", "two"], ["one"]],
        1.0,
        1.0,
        1.0,
    ),
]


@parameterized(METRICS_TEST_CASES)
def test_h_scores(graph, y_true, y_pred, expected_hr_score, expected_hp_score, expected_hf1_score):
    """Test the hR, hP, hF1 metrics on a few synthetic data test cases."""
    with multi_labeled(y_true, y_pred, graph) as (y_true_, y_pred_, graph_):
        assert_that(
            h_recall_score(y_true=y_true_, y_pred=y_pred_, class_hierarchy=graph_),
            is_(close_to(expected_hr_score, delta=0.0001)),
        )
        assert_that(
            h_precision_score(y_true=y_true_, y_pred=y_pred_, class_hierarchy=graph_),
            is_(close_to(expected_hp_score, delta=0.0001)),
        )
        assert_that(
            h_fbeta_score(y_true=y_true_, y_pred=y_pred_, class_hierarchy=graph_),
            is_(close_to(expected_hf1_score, delta=0.0001)),
        )
