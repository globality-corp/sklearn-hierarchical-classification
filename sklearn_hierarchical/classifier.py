"""
Hierarchical classifier interface.

"""
import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin
from sklearn.utils.validation import check_consistent_length, check_is_fitted, check_X_y

from sklearn_hierarchical.constants import ROOT
from sklearn_hierarchical.decorators import logger
from sklearn_hierarchical.graph import root_nodes


@logger
class HierarchicalClassifier(BaseEstimator, ClassifierMixin, MetaEstimatorMixin):

    def __init__(self, base_classifier, class_hierarchy):
        """Hierarchical classification strategy

        Hierarchical classification in general deals with the scenario where our target classes
        have inherent structure that can generally be represented as a tree or a directed acyclic graph (DAG),
        with nodes representing the target classes themselves, and edges representing their inter-relatedness,
        e.g 'IS A' semantics.

        Within this general framework, several distinctions can be made based on a few key modelling decisions:

        - Multi-label classification - Do we support classifying into more than a single target class/label
        - Mandatory / Non-mandatory leaf node prediction - Do we require that classification always results with
            classes corresponding to leaf nodes, or can intermediate nodes also be treated as valid output predictions.
        - Local classifiers - the local (or "base") classifiers can theoretically be chosen to be of any kind, but we
            distinguish between three main modes of local classification:
                * "One classifier per parent node" - where each non-terminal node can be fitted with a multi-class
                    classifier to predict which one of its child nodes is relevant for given example.
                * "One classifier per node" - where each node is fitted with a binary "membership" classifier which
                    returns a binary (or a probability) score indicating the fitness for that node and the current
                    example.
                * Global / "big bang" classifiers - where a single classifier predicts the full path in the hierarchy
                    for a given example.

        The nomenclature used here is based on the following papers:

            "A survey of hierarchical classification across different application domains":
            https://link.springer.com/article/10.1007/s10618-010-0175-9

        Parameters
        ----------
        class_hierarchy: networkx.DiGraph object
            A directed graph which represents the target classes and their relations. Must be a tree/DAG (no cycles).

        base_classifier : classifier object
            A classifier object implementing 'fit' and 'predict' to be used as the base classifier.

        Attributes
        ----------
        classes_ : array, shape = [`n_classes`]
            Flat array of class labels

        """
        self.base_classifier = base_classifier
        self.class_hierarchy = class_hierarchy

    def fit(self, X, y=None, sample_weight=None):
        """Fit underlying classifiers.

        Parameters
        ----------
        X : (sparse) array-like, shape = [n_samples, n_features]
            Data.

        y : (sparse) array-like, shape = [n_samples, ], [n_samples, n_classes]
            Multi-class targets. An indicator matrix turns on multilabel
            classification.

        sample_weight : array-like, shape (n_samples,), optional (default=None)
            Weights applied to individual samples (1. for unweighted).

        Returns
        -------
        self
        """
        check_X_y(X, y)
        if sample_weight is not None:
            check_consistent_length(y, sample_weight)

        # Initialize NetworkX Graph from input class hierarchy
        self.graph_ = nx.DiGraph(self.class_hierarchy)

        # Recursively build training feature set for each node in graph
        nodeset = root_nodes(self.graph_)
        for node_id in nodeset:
            self._recursive_build_features(X, y, node_id=node_id)

    def predict(self, X):
        """Predict multi-class targets using underlying estimators.

        Parameters
        ----------
        X : (sparse) array-like, shape = [n_samples, n_features]
            Data.

        Returns
        -------
        y : (sparse) array-like, shape = [n_samples, ], [n_samples, n_classes].
            Predicted multi-class targets.

        """
        check_is_fitted(self, "graph_")

    def predict_proba(self, X):
        # TODO
        pass

    @property
    def classes_(self):
        return list(
            node
            for node in self.graph_.nodes()
            if node != ROOT
        )

    @property
    def n_classes_(self):
        return len(self.classes_)

    def _recursive_build_features(self, X, y, node_id):
        self.logger.debug("Building features for node: %s", node_id)
        if self.graph_.out_degree(node_id) == 0:
            # Terminal node
            indices = np.flatnonzero(y == node_id)
            self.graph_.node[node_id]["X"] = self._build_features(
                X=X,
                indices=indices,
            )
            return self.graph_.node[node_id]["X"]

        self.graph_.node[node_id]["X"] = csr_matrix(
            X.shape,
            dtype=X.dtype,
        )
        for child_node_id in self.graph_.successors(node_id):
            self.graph_.node[node_id]["X"] += \
                self._recursive_build_features(X, y, node_id=child_node_id)

        return self.graph_.node[node_id]["X"]

    def _build_features(self, X, indices):
        X_ = lil_matrix(X.shape, dtype=X.dtype)
        X_[indices, :] = X[indices, :]
        return X_.tocsr()
