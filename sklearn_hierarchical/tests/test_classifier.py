"""
Unit-tests for the classifier interface.

"""
from hamcrest import (
    assert_that,
    close_to,
    contains_inanyorder,
    equal_to,
    is_,
)
from networkx import DiGraph
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils.estimator_checks import check_estimator

from sklearn_hierarchical.classifier import HierarchicalClassifier
from sklearn_hierarchical.tests.fixtures import make_classifier_and_data
from sklearn_hierarchical.tests.matchers import matches_graph


RANDOM_STATE = 42


def test_estimator_inteface():
    """Run the scikit-learn estimator compatability test suite"""
    check_estimator(HierarchicalClassifier())


def test_fitted_properties():
    n_classes = 10
    clf, (X, y) = make_classifier_and_data(n_classes=n_classes)

    clf.fit(X, y)

    assert_that(DiGraph(clf.class_hierarchy_), matches_graph(DiGraph(clf.class_hierarchy)))
    assert_that(clf.graph_, matches_graph(DiGraph(clf.class_hierarchy)))
    assert_that(clf.classes_, contains_inanyorder(*range(n_classes)))
    assert_that(clf.n_classes_, is_(equal_to(n_classes)))


def test_trivial_hierarchy_classification():
    """Test that a trivial/degenerate hierarchy behaves as expected."""
    clf, (X, y) = make_classifier_and_data(class_hierarchy=False, use_wine_dataset=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    assert_that(accuracy, is_(close_to(1., delta=0.001)))


def test_nontrivial_hierarchy_classification():
    """Test that a nontrivial hierarchy behaves as expected."""
    clf, (X, y) = make_classifier_and_data(n_classes=5)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    assert_that(accuracy, is_(close_to(1., delta=0.05)))
