"""
Unit-test fixtures and factory methods.

"""
from itertools import product

import numpy as np
from networkx import DiGraph, gn_graph, to_dict_of_lists
from sklearn.datasets import load_digits, make_blobs

from sklearn_hierarchical.classifier import HierarchicalClassifier
from sklearn_hierarchical.constants import ROOT


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

    Returns
    -------
    G : dict of lists adjacency matrix format representing the class hierarchy
    """
    if n_leaf is None and n_intermediate is None:
        # No specific structure specified, use a general purpose graph generator
        G = gn_graph(n=n, create_using=DiGraph())

    if n_intermediate == 0:
        # No intermediate nodes, build a 1-level rooted tree
        if n_leaf is None:
            n_leaf = n - 1

        G = DiGraph(product((ROOT,), range(n_leaf)))

    return to_dict_of_lists(G)


def make_digits_dataset(targets=None, as_str=True):
    X, y = load_digits(return_X_y=True)
    if targets:
        ix = np.isin(y, targets)
        X, y = X[np.where(ix)], y[np.where(ix)]

    if as_str:
        # Convert targets (classes) to strings
        y = y.astype(str)

    return X, y


def make_classifier(base_estimator=None, class_hierarchy=None, **kwargs):
    return HierarchicalClassifier(
        class_hierarchy=class_hierarchy,
        base_estimator=base_estimator,
        **kwargs
    )


def make_classifier_and_data(
    n_classes=10,
    n_samples=1000,
    n_features=10,
    class_hierarchy=None,
    **classifier_kwargs
):
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_classes,
    )

    class_hierarchy = class_hierarchy or make_class_hierarchy(
        n=n_classes+1,
        n_intermediate=0,
    )

    clf = make_classifier(
        class_hierarchy=class_hierarchy,
        **classifier_kwargs
    )

    return clf, (X, y)
