#!/usr/bin/env python
"""
Example of using the hierarchical classifier to classify (a subset of) the MNIST digits data set.

Demonstrated some of the capabilities, e.g using a Pipeline as the base estimator,
defining a non-trivial class hierarchy, etc.

"""
from sklearn import svm
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from sklearn_hierarchical.classifier import HierarchicalClassifier
from sklearn_hierarchical.constants import ROOT
from sklearn_hierarchical.tests.fixtures import make_digits_dataset


# Used for seeding random state
RANDOM_STATE = 42


def classify_mnist():
    """Test that a nontrivial hierarchy leaf classification behaves as expected.

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
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))


if __name__ == "__main__":
    classify_mnist()
