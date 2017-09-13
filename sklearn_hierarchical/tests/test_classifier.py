"""
Unit-tests for the classifier interface.

"""

from hamcrest import (
    assert_that,
    equal_to,
    is_,
)
from networkx import DiGraph
from sklearn.utils.estimator_checks import check_estimator

from sklearn_hierarchical.tests.fixtures import make_classifier_and_data
from sklearn_hierarchical.tests.matchers import matches_graph


def test_estimator_inteface():
    clf, _ = make_classifier_and_data(n_classes=4)

    class _Estimator(clf.__class__):
        def __init__(self):
            super().__init__(
                base_classifier=clf.base_classifier,
                class_hierarchy=clf.class_hierarchy,
            )

    check_estimator(_Estimator)


def test_fit():
    n_classes = 10
    clf, (X, y) = make_classifier_and_data(n_classes=n_classes)

    clf.fit(X, y)

    assert_that(clf.graph_, matches_graph(DiGraph(clf.class_hierarchy)))
    assert_that(clf.classes_, is_(equal_to(list(range(n_classes)))))
    assert_that(clf.n_classes_, is_(equal_to(n_classes)))
