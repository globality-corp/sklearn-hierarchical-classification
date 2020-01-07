"""
Unit-tests for the classifier interface.

"""
from hamcrest import (
    assert_that,
    close_to,
    contains_inanyorder,
    equal_to,
    has_entries,
    has_item,
    is_,
)
from networkx import DiGraph
from numpy import where
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.estimator_checks import check_estimator

from sklearn_hierarchical_classification.classifier import HierarchicalClassifier
from sklearn_hierarchical_classification.constants import CLASSIFIER, DEFAULT, ROOT
from sklearn_hierarchical_classification.tests.fixtures import (
    make_classifier,
    make_classifier_and_data,
    make_clothing_graph_and_data,
    make_digits_dataset,
    make_classifier_and_data_own_preprocessing
)
from sklearn_hierarchical_classification.tests.matchers import matches_graph


RANDOM_STATE = 42


def test_estimator_inteface():
    """Run the scikit-learn estimator compatability test suite."""
    check_estimator(HierarchicalClassifier())


def test_fitted_attributes():
    """Test classifier attributes are set correctly after fitting."""
    n_classes = 10
    clf, (X, y) = make_classifier_and_data(n_classes=n_classes)

    clf.fit(X, y)

    assert_that(DiGraph(clf.class_hierarchy_), matches_graph(DiGraph(clf.class_hierarchy)))
    assert_that(clf.graph_, matches_graph(DiGraph(clf.class_hierarchy)))
    assert_that(clf.classes_, contains_inanyorder(*range(n_classes)))
    assert_that(clf.n_classes_, is_(equal_to(n_classes)))
    assert_that(
        clf.graph_.nodes[ROOT],
        has_entries(
            metafeatures=has_entries(
                n_samples=X.shape[0],
                n_targets=n_classes,
            ),
        ),
    )


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


def test_trivial_hierarchy_classification_own_preprocessing_integration():
    """Test that an integration with 20news groups and own preprocessing"""
    print("integration test with 20news")
    clf, (X, y) = make_classifier_and_data_own_preprocessing()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)
    import numpy as np
    y_pred[np.where(y_pred==0)]=-1
    accuracy = accuracy_score(y_test, y_pred>-0.2)
    print("accuracy",accuracy)

    assert_that(accuracy, is_(close_to(.8, delta=0.05)))
    print("finished integration test")


def test_base_estimator_as_dict():
    """Test that specifying base_estimator as a dictionary mappings nodes to base estimators works."""
    class_hierarchy = {
        ROOT: ["A", "B"],
        "A": [1, 7],
        "B": [3, 8, 9],
    }
    clf = make_classifier(
        base_estimator={
            ROOT: KNeighborsClassifier(),
            "B": svm.SVC(probability=True),
            DEFAULT: MultinomialNB(),
        },
        class_hierarchy=class_hierarchy,
    )
    X, y = make_digits_dataset(
        targets=[1, 7, 3, 8, 9],
        as_str=False,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    clf.fit(X_train, y_train)

    assert_that(isinstance(clf.graph_.nodes[ROOT][CLASSIFIER], KNeighborsClassifier))
    assert_that(isinstance(clf.graph_.nodes["B"][CLASSIFIER], svm.SVC))
    assert_that(isinstance(clf.graph_.nodes["A"][CLASSIFIER], MultinomialNB))


def test_nontrivial_hierarchy_leaf_classification():
    r"""Test that a nontrivial hierarchy leaf classification behaves as expected
    under the default parameters.

    We build the following class hierarchy along with data from the handwritten digits dataset:

            <ROOT>
           /      \
          A        B
         / \      / \ \
        1   7    3   8  9

    """
    class_hierarchy = {
        ROOT: ["A", "B"],
        "A": [1, 7],
        "B": [3, 8, 9],
    }
    base_estimator = svm.SVC(
        gamma=0.001,
        kernel="rbf",
        probability=True
    )
    clf = make_classifier(
        base_estimator=base_estimator,
        class_hierarchy=class_hierarchy,
    )
    X, y = make_digits_dataset(
        targets=[1, 7, 3, 8, 9],
        as_str=False,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    assert_that(accuracy, is_(close_to(1., delta=0.02)))


def test_intermediate_node_training_data():
    r"""Test that a training set which includes intermediate (non-leaf) nodes
    as labels, as well as leaf nodes, constructs a correct classifier hierarchy

    """
    G, (X, y) = make_clothing_graph_and_data(root=ROOT)

    # Add a new node rendering "Bottoms" an intermediate node with training data
    G.add_edge("Bottoms", "Pants")

    assert_that(any(yi == "Pants" for yi in y), is_(False))
    assert_that(any(yi == "Bottoms" for yi in y), is_(True))

    base_estimator = LogisticRegression(
        solver="lbfgs",
        max_iter=1_000,
        multi_class="multinomial",
    )

    clf = HierarchicalClassifier(
        base_estimator,
        class_hierarchy=G,
        algorithm="lcpn",
        root=ROOT,
    )
    clf.fit(X, y)

    # Ensure non-terminal node with training data is included in its' parent classifier classes
    assert_that(clf.graph_.nodes()["Mens"]["classifier"].classes_, has_item("Bottoms"))


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

