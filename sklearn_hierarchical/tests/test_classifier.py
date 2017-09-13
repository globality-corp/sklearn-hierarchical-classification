"""
Unit-tests for the classifier interface.

"""

from hamcrest import (
    assert_that,
    equal_to,
    is_,
)
from networkx import DiGraph

from sklearn_hierarchical.tests.fixtures import make_classifier_and_data
from sklearn_hierarchical.tests.matchers import matches_graph


def test_fit():
    n_classes = 10
    clf, (X, y) = make_classifier_and_data(n_classes=n_classes)

    clf.fit(X, y)

    assert_that(clf.graph_, matches_graph(DiGraph(clf.class_hierarchy)))
    assert_that(clf.classes_, is_(equal_to(list(range(n_classes)))))
    assert_that(clf.n_classes_, is_(equal_to(n_classes)))
