"""
Unit-test fixtures and factory methods.

"""
from itertools import product

from networkx import DiGraph, gn_graph, to_dict_of_lists
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

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
    G : networkx.DiGraph object
    """
    if n_leaf is None and n_intermediate is None:
        # No specific structure specified, use a general purpose graph generator
        G = gn_graph(n=n, create_using=DiGraph())

    if n_intermediate == 0:
        # No intermediate nodes, build a 1-level rooted tree
        if n_leaf is None:
            n_leaf = n - 1

        G = DiGraph(data=product((ROOT,), range(n_leaf)))

    return G


def make_classifier_and_data(n_classes=10, n_samples=100, n_features=50, base_classifier_cls=LogisticRegression):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_classes,
        n_classes=n_classes,
    )
    class_hierarchy = make_class_hierarchy(
        n=n_classes+1,
        n_intermediate=0,
    )

    clf = HierarchicalClassifier(
        class_hierarchy=to_dict_of_lists(class_hierarchy),
        base_classifier=base_classifier_cls(),
    )

    return clf, (X, y)
