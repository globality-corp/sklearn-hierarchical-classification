"""
Hierarchical classifier interface.

"""
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin


class HierarchicalClassifier(BaseEstimator, ClassifierMixin, MetaEstimatorMixin):

    def __init__(self, class_hierarchy, base_classifier):
        """
        Hierarchical classification strategy

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
        self.class_hierarchy = class_hierarchy
        self.base_classifier = base_classifier

    def fit(self, X, y=None):
        """Fit underlying classifiers.

        Parameters
        ----------
        X : (sparse) array-like, shape = [n_samples, n_features]
            Data.
        y : (sparse) array-like, shape = [n_samples, ], [n_samples, n_classes]
            Multi-class targets. An indicator matrix turns on multilabel
            classification.

        Returns
        -------
        self
        """

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

    @property
    def classes_(self):
        return list(self.class_hierarchy.nodes())

    @property
    def n_classes_(self):
        return len(self.classes_)
