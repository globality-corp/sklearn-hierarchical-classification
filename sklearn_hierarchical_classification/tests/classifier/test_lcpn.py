"""
Unit-tests for the classifier interface.

"""
from hamcrest import (
    assert_that,
    close_to,
    has_item,
    is_,
)
from numpy import where
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from sklearn_hierarchical_classification.constants import ROOT
from sklearn_hierarchical_classification.tests.fixtures import (
    make_classifier,
    make_classifier_and_data,
    make_digits_dataset,
    make_mlb_classifier_and_data_with_feature_extraction_pipeline,
)


RANDOM_STATE = 42


def test_trivial_hierarchy_classification():
    """Test that a trivial (degenerate) hierarchy behaves as expected."""
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


def test_nmlnp_strategy_with_float_stopping_criteria():
    # since NMLNP results in a mix of intermediate and leaf nodes,
    # make sure they are all of same dtype (str)
    class_hierarchy = {
        ROOT: ["A", "B"],
        "A": ["1", "5", "6", "7"],
        "B": ["2", "3", "4", "8", "9"],
    }
    base_estimator = svm.SVC(
        gamma=0.001,
        kernel="rbf",
        probability=True
    )
    clf = make_classifier(
        base_estimator=base_estimator,
        class_hierarchy=class_hierarchy,
        prediction_depth="nmlnp",
        stopping_criteria=0.9,
    )

    X, y = make_digits_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    assert_that(list(y_pred), has_item("B"))


def test_nmlnp_strategy_on_tree_with_dummy_classifier():
    """Test classification works on a tree graph when one of the nodes has out-degree 1 resulting in
    creation of a "dummy" classifier at that node to trivially predict its child."""
    # since NMLNP results in a mix of intermediate and lefa nodes,
    # make sure they are all of same dtype (str)
    class_hierarchy = {
        ROOT: ["A", "B", "C"],
        "A": ["1", "5", "6", "7"],
        "B": ["2", "3", "8", "9"],
        "C": ["4"],
    }
    base_estimator = svm.SVC(
        gamma=0.001,
        kernel="rbf",
        probability=True
    )
    clf = make_classifier(
        base_estimator=base_estimator,
        class_hierarchy=class_hierarchy,
        prediction_depth="nmlnp",
        stopping_criteria=0.9,
    )

    X, y = make_digits_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    assert_that(list(y_pred), has_item("4"))


def test_nmlnp_strategy_on_dag_with_dummy_classifier():
    """Test classification works on a "deep" DAG when one of the nodes has out-degree 1,
    resulting in creation of a "dummy" classifier at that node to trivially predict its child.
    This test case actually tests a few more subtle edge cases:
    - String-based target labels with length > 1
    - Multi-level degenerate sub-graphs, e.g some nodes having a sub-graph which is a path.
    """
    # since NMLNP results in a mix of intermediate and lefa nodes,
    # make sure they are all of same dtype (str)
    class_hierarchy = {
        ROOT: ["A", "B", "C"],
        "A": ["1", "5", "6", "7"],
        "B": ["2", "BC.1", "8", "9"],
        "BC.1": ["3a"],
        "C": ["BC.1"],
    }
    base_estimator = svm.SVC(
        gamma=0.001,
        kernel="rbf",
        probability=True
    )
    clf = make_classifier(
        base_estimator=base_estimator,
        class_hierarchy=class_hierarchy,
        prediction_depth="nmlnp",
        stopping_criteria=0.9,
    )

    X, y = make_digits_dataset()
    y[where(y == "3")] = "3a"
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    assert_that(list(y_pred), has_item("3a"))


def test_mlb_hierarchy_classification_with_feature_extraction_pipeline():
    """Test multi-label classification with a feature extraction pipeline"""
    clf, (X, y) = make_mlb_classifier_and_data_with_feature_extraction_pipeline()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)
    y_pred[where(y_pred == 0)] = -1
    accuracy = accuracy_score(y_test, y_pred > -0.2)

    assert_that(accuracy, is_(close_to(.8, delta=0.05)))
