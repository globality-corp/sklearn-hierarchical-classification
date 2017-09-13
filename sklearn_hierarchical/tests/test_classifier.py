"""
Unit-tests for the classifier interface.

"""

from hamcrest import (
    assert_that,
    equal_to,
    is_,
)
from networkx import DiGraph, gn_graph, to_dict_of_lists
from sklearn.datasets import make_classification

from sklearn_hierarchical.classifier import HierarchicalClassifier
from sklearn_hierarchical.constants import ROOT
from sklearn_hierarchical.tests.matchers import matches_graph


def make_class_hierarchy(n, n_intermediate=None, n_leaf=None):
    """Create a mock class hierarchy for testing purposes.

    Parameters
    ----------
    n : int
        Number of nodes in the returned graph

    n_intermediate : int
        Number of intermediate (non-root, non-terminal) nodes in the returned graph

    n_leaf : int
        Number of leaf (terminal) nodes in the returned graph

    """
    if n_leaf is None and n_intermediate is None:
        # No specific structure specified, use a general purpose graph generator
        G = gn_graph(n=n, create_using=DiGraph())

    if n_intermediate == 0:
        # No intermediate nodes, build a 1-level rooted tree
        if n_leaf is None:
            n_leaf = n - 1

        G = DiGraph()
        G.add_node(ROOT)
        for idx in range(n_leaf):
            G.add_edge(ROOT, idx)

    return G


def make_base_classifier():
    return None


def make_classifier_and_data(n_classes=10):
    X, y = make_classification(
        n_samples=100,
        n_features=50,
        n_informative=n_classes,
        n_classes=n_classes,
    )
    class_hierarchy = make_class_hierarchy(
        n=n_classes+1,
        n_intermediate=0,
    )
    base_classifier = make_base_classifier()
    clf = HierarchicalClassifier(
        class_hierarchy=to_dict_of_lists(class_hierarchy),
        base_classifier=base_classifier,
    )

    return clf, (X, y)


def test_fit():
    n_classes = 10
    clf, (X, y) = make_classifier_and_data(n_classes=n_classes)

    clf.fit(X, y)

    assert_that(clf.graph_, matches_graph(DiGraph(clf.class_hierarchy)))
    assert_that(clf.n_classes_, is_(equal_to(n_classes)))
    assert_that(clf.classes_, is_(equal_to(list(range(10)))))
