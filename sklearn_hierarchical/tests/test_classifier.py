"""
Unit-tests for the classifier interface.

"""

from hamcrest import (
    assert_that,
    equal_to,
    is_,
)
from networkx import gn_graph
from sklearn.datasets import make_classification

from sklearn_hierarchical.classifier import HierarchicalClassifier
from sklearn_hierarchical.tests.matchers import matches_graph


def make_class_hierarchy(n):
    """Create a mock class hierarchy for testing purposes.

    Parameters
    ----------
    n : int
        Number of nodes in the returned graph

    """
    return gn_graph(n=n)


def make_base_classifier():
    return None


def test_class_hierarchy():
    """Test that a class hierarchy represented as a networkx.DiGraph is parsed correctly."""
    class_hierarchy = make_class_hierarchy(n=10)
    base_classifier = make_base_classifier()

    clf = HierarchicalClassifier(
        class_hierarchy=class_hierarchy,
        base_classifier=base_classifier,
    )

    assert_that(clf.n_classes_, is_(equal_to(10)))
    assert_that(clf.classes_, is_(equal_to(list(class_hierarchy.nodes()))))


def test_fit():
    X, y = make_classification(
        n_samples=100,
        n_features=50,
        n_informative=10,
        n_classes=10,
    )
    class_hierarchy = make_class_hierarchy(n=10)
    base_classifier = make_base_classifier()
    clf = HierarchicalClassifier(
        class_hierarchy=class_hierarchy,
        base_classifier=base_classifier,
    )

    clf.fit(X, y)

    assert_that(clf.graph_, matches_graph(class_hierarchy))
