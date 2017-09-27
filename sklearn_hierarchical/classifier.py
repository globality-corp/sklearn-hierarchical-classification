"""
Hierarchical classifier interface.

"""
import numpy as np
from networkx import DiGraph, dfs_edges
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_array, check_consistent_length, check_is_fitted, check_X_y
from sklearn.utils.multiclass import check_classification_targets
from tqdm import tqdm_notebook

from sklearn_hierarchical.array import apply_along_rows, nnz_rows_ix
from sklearn_hierarchical.constants import CLASSIFIER, DEFAULT, ROOT
from sklearn_hierarchical.decorators import logger
from sklearn_hierarchical.dummy import DummyProgress
from sklearn_hierarchical.graph import make_flat_hierarchy, rollup_nodes, root_nodes
from sklearn_hierarchical.validation import is_estimator, validate_parameters


@logger
class HierarchicalClassifier(BaseEstimator, ClassifierMixin, MetaEstimatorMixin):
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
            * "One classifier per parent node" - where each non-leaf node can be fitted with a multi-class
                classifier to predict which one of its child nodes is relevant for given example.
            * "One classifier per node" - where each node is fitted with a binary "membership" classifier which
                returns a binary (or a probability) score indicating the fitness for that node and the current
                example.
            * Global / "big bang" classifiers - where a single classifier predicts the full path in the hierarchy
                for a given example.

    The nomenclature used here is based on the framework outlined in [1].

    Parameters
    ----------
    base_estimator : classifier object, function, dict, or None
        A scikit-learn compatible classifier object implementing 'fit' and 'predict_proba' to be used as the
        base classifier.
        If a callable function is given, it will be called to evaluate which classifier to instantiate for
        current node. The function will be called with the current node and the graph instance.
        Alternatively, a dictionary mapping classes to classifier objects can be given. In this case,
        when building the classifier tree, the dictionary will be consulted and if a key is found matching
        a particular node, the base classifier pointed to in the dict will be used. Since this is most often
        useful for specifying classifiers on only a handlful of objects, a special 'DEFAULT' key can be used to
        set the base classifier to use as a 'catch all'.
        If not provided, a base estimator will be chosen by the framework using various meta-learning
        heuristics (WIP).

    class_hierarchy : networkx.DiGraph object
        A directed graph which represents the target classes and their relations. Must be a tree/DAG (no cycles).
        If not provided, this will be initialized during the `fit` operation into a trivial graph structure linking
        all classes given in `y` to an artificial "ROOT" node.

    prediction_depth : "mlnp", "nmlnp"
        Prediction depth requirements. This corresponds to whether we wish the classifier to always terminate at
        a leaf node (mandatory leaf-node prediction, "mlnp"), or wish to support early termination via some
        stopping criteria (non-mandatory leaf-node prediction, "nmlnp"). When "nmlnp" is specified, the
        stopping_criteria parameter is used to control the behaviour of the classifier.

    algorithm : "lcn", "lcpn"
        The algorithm type to use for building the hierarchical classification, according to the
        taxonomy defined in [1].

        "lcpn" (which is the default) stands for "local classifier per parent node". Under this model,
        a multi-class classifier is trained at each parent node, to distinguish between each child nodes.

        "lcn", which stands for "local classifier per node". Under this model, a binary classifier is trained
        at each node. Under this model, a further distinction is made based on how the training data set is constructed.
        This is controlled by the "training_strategy" parameter.

    training_strategy: "exclusive", "less_exclusive", "inclusive", "less_inclusive",
                       "siblings", "exclusive_siblings", or None.
        This parameter is used when the 'algorithm' parameter is to set to "lcn", and dictates how training data
        is constructed for training the binary classifier at each node.

    stopping_criteria: function, float, or None.
        This parameter is used when the 'prediction_depth' parameter is set to "nmlnp", and is used to evaluate
        at a given node whether classification should terminate or continue further down the hierarchy.

        When set to a float, the prediction will stop if the reported confidence at current classifier is below
        the provided value.

        When set to a function, the callback function will be called with the current node attributes,
        including its metafeatures, and the current classification results.
        This allows the user to define arbitrary logic that can decide whether classification should stop at
        the current node or continue. The function should return True if classification should continue,
        or False if classification should stop at current node.

    interactive : bool
        If set to True, functionality which is useful for interactive usage (e.g in a Jupyter notebook) will be
        enabled. Specifically, fitting the model will display progress bars (via tqdm) where appropriate, and more
        verbose logging will be emitted.

    Attributes
    ----------
    classes_ : array, shape = [`n_classes`]
        Flat array of class labels

    References
    ----------

    .. [1] CN Silla et al., "A survey of hierarchical classification across
           different application domains", 2011.

    """
    def __init__(self, base_estimator=None, class_hierarchy=None, prediction_depth="mlnp",
                 algorithm="lcpn", training_strategy=None, stopping_criteria=None,
                 interactive=False):
        self.base_estimator = base_estimator
        self.class_hierarchy = class_hierarchy
        self.prediction_depth = prediction_depth
        self.algorithm = algorithm
        self.training_strategy = training_strategy
        self.stopping_criteria = stopping_criteria
        self.interactive = interactive

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
        X, y = check_X_y(X, y, accept_sparse='csr')
        check_classification_targets(y)
        if sample_weight is not None:
            check_consistent_length(y, sample_weight)

        # Check that parameter assignment is consistent
        self._check_parameters()

        # Initialize NetworkX Graph from input class hierarchy
        self.class_hierarchy_ = self.class_hierarchy or make_flat_hierarchy(list(np.unique(y)), root=ROOT)
        self.graph_ = DiGraph(self.class_hierarchy_)
        self.classes_ = list(
            node
            for node in self.graph_.nodes()
            if node != ROOT
        )

        # Recursively build training feature sets for each node in graph
        # based on the passed in "global" feature set
        with self._progress(total=self.n_classes_, desc="Building features") as progress:
            for node_id in root_nodes(self.graph_):
                self._recursive_build_features(X, y, node_id=node_id, progress=progress)

        # Recursively train base classifiers
        with self._progress(total=self.n_classes_, desc="Training base classifiers") as progress:
            for node_id in root_nodes(self.graph_):
                self._recursive_train_local_classifiers(X, y, node_id=node_id, progress=progress)

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
        X = check_array(X, accept_sparse='csr')

        def _classify(x):
            y_pred = []
            for node_id in root_nodes(self.graph_):
                path, scores = self._recursive_predict(x, node_id=node_id)
                y_pred.append(path[-1])
            # TODO support multi-label / paths?
            return y_pred[0]

        y_pred = apply_along_rows(_classify, X=X)

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
        X = check_array(X, accept_sparse="csr")

        def _classify(x):
            y_pred = []
            for node_id in root_nodes(self.graph_):
                path, class_probabilities = self._recursive_predict(x, node_id=node_id)
                y_pred.append((path[-1], class_probabilities))
            # TODO support multi-label / paths?
            return y_pred[0][1]

        y_pred = apply_along_rows(_classify, X=X)

        return y_pred

    @property
    def n_classes_(self):
        return len(self.classes_)

    def _check_parameters(self):
        """Check the parameter assignment is valid and internally consistent."""
        validate_parameters(self)

    def _recursive_build_features(self, X, y, node_id, progress):
        if "X" in self.graph_.node[node_id]:
            # Already visited this node in feature building phase
            return self.graph_.node[node_id]["X"]

        self.logger.debug("Building features for node: %s", node_id)
        progress.update(1)

        if self.graph_.out_degree(node_id) == 0:
            # Leaf node
            self.logger.debug("_build_features() - node_id: %s, set(y): %s", node_id, set(y))
            indices = np.flatnonzero(y == node_id)
            self.graph_.node[node_id]["X"] = self._build_features(
                X=X,
                indices=indices,
            )
            return self.graph_.node[node_id]["X"]

        # Non-leaf node
        self.graph_.node[node_id]["X"] = csr_matrix(
            X.shape,
            dtype=X.dtype,
        )
        for child_node_id in self.graph_.successors(node_id):
            self.graph_.node[node_id]["X"] += \
                self._recursive_build_features(
                    X=X,
                    y=y,
                    node_id=child_node_id,
                    progress=progress,
                )

        self.graph_.node[node_id]["metafeatures"] = self._build_metafeatures(
            X=self.graph_.node[node_id]["X"],
            y=y,
        )

        return self.graph_.node[node_id]["X"]

    def _build_features(self, X, indices):
        X_ = lil_matrix(X.shape, dtype=X.dtype)
        X_[indices, :] = X[indices, :]
        return X_.tocsr()

    def _build_metafeatures(self, X, y):
        """
        Build the meta-features associated with a particular node.

        These are various features that can be used in training and prediction time,
        e.g the number of training samples available for the classifier trained at that node,
        the number of targets (classes) to be predicted at that node, etc.

        Parameters
        ----------
        X : (sparse) array-like, shape = [num_samples, num_features]
            The training data matrix at current node.

        Returns
        -------
        metafeatures : dict
            Python dictionary of meta-features. The following meta-features are computed by default:
            * 'num_samples' - Number of samples used to train classifier at given node.
            * 'num_targets' - Number of targets (classes) to classify into at given node.

        """
        # Indices of non-zero rows in X, i.e rows corresponding to relevant samples for this node.
        ix = nnz_rows_ix(X)

        return dict(
            num_samples=len(ix),
            num_targets=len(np.unique(y[ix])),
        )

    def _train_local_classifier(self, X, y, node_id):
        if self.graph_.out_degree(node_id) == 0:
            # Leaf node
            if self.algorithm == "lcpn":
                # Leaf nodes do not get a classifier assigned in LCPN algorithm mode.
                self.logger.debug(
                    "_train_local_classifier() - skipping leaf node %s when algorithm is 'lcpn'",
                    node_id,
                )
                return

        X = self.graph_.node[node_id]["X"]
        nnz_rows = nnz_rows_ix(X)
        X = X[nnz_rows, :]

        y = np.array(
            rollup_nodes(
                graph=self.graph_,
                root=node_id,
                targets=[y[idx] for idx in nnz_rows],
            )
        )

        self.logger.debug(
            "_train_local_classifier() - Training local classifier for node: %s, X.shape: %s, len(unique(y)): %s",
            node_id,
            X.shape,
            len(np.unique(y)),
        )

        if len(set(y)) == 0:
            # No training data could be materialized for current node
            # TODO: support a 'strict' mode flag to explicitly enable/disable fallback logic here?
            self.logger.warning(
                "_train_local_classifier() - not enough training data available to train classifier, classification in branch will terminate at node %s",  # noqa:E501
                node_id,
            )
            return
        elif len(set(y)) == 1:
            # Training data could be materialized for only a single target at current node
            # TODO: support a 'strict' mode flag to explicitly enable/disable fallback logic here?
            self.logger.warning(
                "_train_local_classifier() - not enough training data available to train classifier for node %s, Will trivially predict %s",  # noqa:E501
                node_id,
                y[0],
            )
            clf = DummyClassifier(strategy="constant", constant=y[0])
        else:
            clf = self._base_estimator_for(node_id)

        clf.fit(X=X, y=y)
        self.graph_.node[node_id][CLASSIFIER] = clf

    def _recursive_train_local_classifiers(self, X, y, node_id, progress):
        if self.graph_.node[node_id].get(CLASSIFIER, None):
            # Already encountered and trained classifier at this node, no-op
            return

        progress.update(1)
        self._train_local_classifier(X, y, node_id)

        for child_node_id in self.graph_.successors(node_id):
            out_degree = self.graph_.out_degree(child_node_id)
            if not out_degree:
                # Leaf node, skip
                progress.update(1)
                continue

            if out_degree < 2:
                # If node has less than 2 children, no point training a local classifier
                self.logger.warning(
                    "*** Not enough children to train classifier for node %s, skipping (%s < 2)",
                    child_node_id,
                    self.graph_.out_degree(child_node_id),
                )
                # For tracking progress, count this node as well as all of its descendants
                progress.update(1 + len(list(dfs_edges(self.graph_, child_node_id))))
                continue

            self._recursive_train_local_classifiers(
                X=X,
                y=y,
                node_id=child_node_id,
                progress=progress,
            )

    def _recursive_predict(self, X, node_id):
        clf = self.graph_.node[node_id][CLASSIFIER]
        path = [node_id]
        path_probability = 1.0
        class_probabilities = np.zeros_like(self.classes_, dtype=np.float64)

        while clf:
            probs = clf.predict_proba(X)[0]
            argmax = np.argmax(probs)
            prediction = clf.classes_[argmax]
            score = probs[argmax]

            # Report probabilities in terms of complete class hierarchy
            for local_class_idx, class_ in enumerate(clf.classes_):
                class_idx = self.classes_.index(class_)
                class_probabilities[class_idx] = probs[local_class_idx]

            if self._should_early_terminate(
                current_node=path[-1],
                prediction=prediction,
                score=score,
            ):
                break

            # Update current path
            path_probability *= score
            path.append(prediction)

            clf = self.graph_.node[prediction].get(CLASSIFIER, None)

        return path, class_probabilities

    def _should_early_terminate(self, current_node, prediction, score):
        """
        Evaluate whether classification should terminate at given step.

        This depends on whether early-termination, as dictated by the the 'prediction_depth'
          and 'stopping_criteria' parameters, is triggered.

        """
        if self.prediction_depth != "nmlnp":
            # Prediction depth parameter does not allow for early termination
            return False

        if (
            isinstance(self.stopping_criteria, float)
            and current_node != ROOT
            and score < self.stopping_criteria
        ):
            self.logger.debug(
                "_should_early_terminate() - score %s < %s, terminating at node %s",
                score,
                self.stopping_criteria,
                current_node,
            )
            return True

        if callable(self.stopping_criteria):
            return self.stopping_criteria(
                current_node=self.graph_.nodes[current_node],
                prediction=prediction,
                score=score,
            )

        # Shouldn't really ever get here
        return False

    def _make_base_estimator(self, node_id):
        return LogisticRegression()

    def _base_estimator_for(self, node_id):
        if not self.base_estimator:
            # No base estimator specified by user, try to pick best one
            return self._make_base_estimator(node_id)

        if isinstance(self.base_estimator, dict):
            # user provided dictionary mapping nodes to estimators
            if node_id in self.base_estimator:
                return clone(self.base_estimator[node_id])
            else:
                return clone(self.base_estimator[DEFAULT])

        if is_estimator(self.base_estimator):
            # single base estimator object, return a copy
            return clone(self.base_estimator)

        # By default, treat as callable factory
        self.base_estimator(node_id=node_id, graph=self.graph_)

    def _progress(self, total, desc, **kwargs):
        if self.interactive:
            return tqdm_notebook(total=total, desc=desc)
        else:
            return DummyProgress()
