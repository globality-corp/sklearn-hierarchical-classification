#!/usr/bin/env python
"""
Example of using the hierarchical classifier to classify (a subset of) the digits data set.

Demonstrated some of the capabilities, e.g using a Pipeline as the base estimator,
defining a non-trivial class hierarchy, etc.

"""
from sklearn import svm
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from sklearn_hierarchical_classification.classifier import HierarchicalClassifier
from sklearn_hierarchical_classification.constants import ROOT
from sklearn_hierarchical_classification.metrics import h_fbeta_score, multi_labeled
from sklearn_hierarchical_classification.tests.fixtures import make_digits_dataset


# Used for seeding random state
RANDOM_STATE = 42


def classify_digits():
    r"""Test that a nontrivial hierarchy leaf classification behaves as expected.

    We build the following class hierarchy along with data from the handwritten digits dataset:

            <ROOT>
           /      \
          A        B
         / \       |  \
        1   7      C   9
                 /   \
                3     8

    """
    class_hierarchy = {
        ROOT: ["A", "B"],
        "A": ["1", "7"],
        "B": ["C", "9"],
        "C": ["3", "8"],
    }
    base_estimator = make_pipeline(
        TruncatedSVD(n_components=24),
        svm.SVC(
            gamma=0.001,
            kernel="rbf",
            probability=True
        ),
    )
    clf = HierarchicalClassifier(
        base_estimator=base_estimator,
        class_hierarchy=class_hierarchy,
    )
    X, y = make_digits_dataset(
        targets=[1, 7, 3, 8, 9],
        as_str=False,
    )
    # cast the targets to strings so we have consistent typing of labels across hierarchy
    y = y.astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Demonstrate using our hierarchical metrics module with MLB wrapper
    with multi_labeled(y_test, y_pred, clf.graph_) as (y_test_, y_pred_, graph_):
        h_fbeta = h_fbeta_score(
            y_test_,
            y_pred_,
            graph_,
        )
        print("h_fbeta_score: ", h_fbeta)


if __name__ == "__main__":
    classify_digits()
