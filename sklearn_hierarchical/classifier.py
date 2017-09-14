"""
Hierarchical classifier interface.

"""
from collections import defaultdict

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_array, check_consistent_length, check_is_fitted, check_X_y
from sklearn.utils.multiclass import check_classification_targets

from sklearn_hierarchical.constants import ROOT
from sklearn_hierarchical.decorators import logger
from sklearn_hierarchical.graph import rollup_nodes, root_nodes


@logger
class HierarchicalClassifier(BaseEstimator, ClassifierMixin, MetaEstimatorMixin):

    def __init__(self, base_estimator=None, class_hierarchy=None, min_num_samples=1):
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

            "A survey of hierarchical classification across different application domains" - CN Silla et al. 2011

        Parameters
        ----------
        base_estimator: classifier object
            A classifier object implementing 'fit' and 'predict' to be used as the base classifier.

        class_hierarchy: networkx.DiGraph object
            A directed graph which represents the target classes and their relations. Must be a tree/DAG (no cycles).
            If not provided, this will be initialized during the `fit` operation into a trivial graph structure linking
            all classes given in `y` to an artificial "ROOT" node.

        min_num_samples : int
            Minimum number of training samples required to train a local classifier on a node (class)

        Attributes
        ----------
        classes_ : array, shape = [`n_classes`]
            Flat array of class labels

        """
        self.base_estimator = base_estimator
        self.class_hierarchy = class_hierarchy
        self.min_num_samples = min_num_samples

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
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        if sample_weight is not None:
            check_consistent_length(y, sample_weight)

        # Initialize NetworkX Graph from input class hierarchy
        self.class_hierarchy_ = self.class_hierarchy or make_flat_hierarchy(set(y))
        self.graph_ = nx.DiGraph(self.class_hierarchy_)
        self.classes_ = list(set(y))

        # Initialize the base estimator
        self.base_estimator_ = self.base_estimator or make_base_estimator()

        # Recursively build training feature sets for each node in graph
        # based on the passed in "global" feature set
        for node_id in root_nodes(self.graph_):
            self._recursive_build_features(X, y, node_id=node_id)

        # Recursively train base classifiers
        for node_id in root_nodes(self.graph_):
            self._recursive_train_local_classifiers(X, y, node_id=node_id)

        return self

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
        X = check_array(X, accept_sparse=True)

        def classify(x):
            y_pred = []
            for node_id in root_nodes(self.graph_):
                path, scores = self._recursive_predict(x.reshape(1, -1), node_id=node_id)
                y_pred.append(path[-1])
            # TODO
            return y_pred[0]

        y_pred = np.apply_along_axis(
            classify,
            axis=1,
            arr=X,
        )

        return y_pred

    def predict_proba(self, X):
        """
        Return probability estimates for the test vector X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        C : array-like, shape = [n_samples, n_classes]
            Returns the probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute `classes_`.
        """
        check_is_fitted(self, "graph_")
        X = check_array(X, accept_sparse=True)

        def _classify(x):
            y_pred = []
            for node_id in root_nodes(self.graph_):
                path, class_probabilities = self._recursive_predict(x.reshape(1, -1), node_id=node_id)
                y_pred.append((path[-1], class_probabilities))
            # TODO support multi-label
            return y_pred[0][1]

        y_pred = np.apply_along_axis(
            _classify,
            axis=1,
            arr=X,
        )
        return y_pred

    @property
    def n_classes_(self):
        return len(self.classes_)

    def _recursive_build_features(self, X, y, node_id):
        self.logger.debug("Building features for node: %s", node_id)
        if self.graph_.out_degree(node_id) == 0:
            # Terminal node
            self.logger.debug("_recursive_build_features() - node_id: %s, set(y): %s", node_id, set(y))
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

    def _train_local_classifier(self, X, y, node_id):
        X = self.graph_.node[node_id]["X"]
        nnz_rows = np.where(X.todense().any(axis=1))[0]

        if len(nnz_rows) < self.min_num_samples:
            self.logger.warning(
                "*** Not enough samples to train classifier for node %s, skipping (%s < %s)",
                node_id,
                len(nnz_rows),
                self.min_num_samples,
            )
            return

        y_train = rollup_nodes(
            graph=self.graph_,
            root=node_id,
            targets=y,
        )

        if len(set(y_train)) < 2:
            self.logger.warning(
                "*** Not enough targets to train classifier for node %s, Will trivially predict %s",
                node_id,
                y_train[0],
            )
            clf = DummyClassifier(strategy="constant", constant=y_train[0])
        else:
            clf = clone(self.base_estimator_)
        clf.fit(X=X, y=y)
        self.graph_.node[node_id]["classifier"] = clf

    def _recursive_train_local_classifiers(self, X, y, node_id):

        if self.graph_.node[node_id].get("classifier", None):
            # Already encountered this node, skip
            return

        self._train_local_classifier(X, y, node_id)

        for child_node_id in self.graph_.successors(node_id):
            out_degree = self.graph_.out_degree(child_node_id)
            if not out_degree:
                # Terminal node, skip
                continue

            if out_degree < 2:
                # If node has less than 2 children, no point training a local classifier
                self.logger.warning(
                    "*** Not enough children to train classifier for node %s, skipping (%s < 2)",
                    child_node_id,
                    self.graph_.out_degree(child_node_id),
                )
                continue

            self._recursive_train_local_classifiers(X, y, child_node_id)

    def _recursive_predict(self, X, node_id):
        clf = self.graph_.node[node_id]["classifier"]
        path = [node_id]
        path_probability = 1.0
        class_probabilities = np.zeros_like(self.classes_, dtype=np.float64)

        while clf:
            probs = clf.predict_proba(X)[0]
            argmax = np.argmax(probs)
            prediction = clf.classes_[argmax]
            score = probs[argmax]

            # Update current path
            path_probability *= score
            path.append(prediction)

            # Report probabilities in terms of complete class hierarchy
            for local_class_idx, class_ in enumerate(clf.classes_):
                class_idx = self.classes_.index(class_)
                class_probabilities[class_idx] = probs[local_class_idx]

            clf = self.graph_.node[prediction].get("classifier", None)

        return path, class_probabilities


def make_flat_hierarchy(targets):
    """
    Create a trivial "flat" hiearchy, linking all given targets to an artificial ROOT node.

    """
    adjacency = defaultdict(list)
    for target in targets:
        adjacency[ROOT].append(target)
    return adjacency


def make_base_estimator():
    return LogisticRegression()
