"""
Evaluation metrics for hierarchical classification.

"""
import numpy as np
from networkx import all_pairs_shortest_path_length


def fill_ancestors(y, graph, copy=True):
    """
    Compute the full ancestor set for y given as a matrix of 0-1.

    Each row will be processed and filled in with 1s in indexes corresponding
    to the (integer) id of the ancestor nodes of those already marked with 1
    in that row, based on the given class hierarchy graph.

    Parameters
    ----------
    y : array-like, shape = [n_samples, n_classes].
        multi-class targets, corresponding to graph node integer ids.

    graph : the class hierarchy graph, given as a `networkx.DiGraph` instance

    Returns
    -------
    y_ : array-like, shape = [n_samples, n_classes].
        multi-class targets, corresponding to graph node integer ids with
        all ancestors of existing labels in matrix filled in, per row.

    """
    y_ = y.copy() if copy else y
    paths = all_pairs_shortest_path_length(graph.reverse(copy=False))
    for target, distances in paths:
        ix_rows = np.where(y[:, target] > 0)[0]
        ancestors = list(distances.keys())
        y_[np.meshgrid(ix_rows, ancestors)] = 1
    graph.reverse(copy=False)
    return y_


def h_precision_score(y_true, y_pred, class_hierarchy):
    """
    Calculate the hierarchical precision ("hR") metric based on
    given set of true class labels and predicated class labels, and the
    class hierarchy graph.

    For motivation and definition details, see:

        Functional Annotation of Genes Using Hierarchical Text
        Categorization, Kiritchenko et al 2008

        http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.68.5824&rep=rep1&type=pdf

    Parameters
    ----------
    y_true : array-like, shape = [n_samples, n_classes].
        Ground truth multi-class targets.

    y_pred : array-like, shape = [n_samples, n_classes].
        Predicted multi-class targets.

    class_hierarchy : the class hierarchy graph, given as a `networkx.DiGraph` instance
        Node ids must be integer and correspond to the indices into the y_true / y_pred matrices.

    Returns
    -------
    hP : float
        The computed hierarchical precision score.

    """
    y_true_ = fill_ancestors(y_true, graph=class_hierarchy)
    y_pred_ = fill_ancestors(y_pred, graph=class_hierarchy)

    ix = np.where((y_true_ != 0) & (y_pred_ != 0))

    true_positives = len(ix[0])
    all_results = np.count_nonzero(y_pred_)

    return true_positives / all_results


def h_recall_score(y_true, y_pred, class_hierarchy):
    """
    Calculate the hierarchical recall ("hR") metric based on
    given set of true class labels and predicated class labels, and the
    class hierarchy graph.

    For motivation and definition details, see:

        Functional Annotation of Genes Using Hierarchical Text
        Categorization, Kiritchenko et al 2008

        http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.68.5824&rep=rep1&type=pdf

    Parameters
    ----------
    y_true : array-like, shape = [n_samples, n_classes].
        Ground truth multi-class targets.

    y_pred : array-like, shape = [n_samples, n_classes].
        Predicted multi-class targets.

    class_hierarchy : the class hierarchy graph, given as a `networkx.DiGraph` instance.
        Node ids must be integer and correspond to the indices into the y_true / y_pred matrices.

    Returns
    -------
    hR : float
        The computed hierarchical recall score.

    """
    y_true_ = fill_ancestors(y_true, graph=class_hierarchy)
    y_pred_ = fill_ancestors(y_pred, graph=class_hierarchy)

    ix = np.where((y_true_ != 0) & (y_pred_ != 0))

    true_positives = len(ix[0])
    all_positives = np.count_nonzero(y_true_)

    return true_positives / all_positives


def h_fbeta_score(y_true, y_pred, class_hierarchy, beta=1.):
    """
    Calculate the hierarchical F-beta ("hF_{\beta}") metric based on
    given set of true class labels and predicated class labels, and the
    class hierarchy graph.

    For motivation and definition details, see:

        Functional Annotation of Genes Using Hierarchical Text
        Categorization, Kiritchenko et al 2008

        http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.68.5824&rep=rep1&type=pdf

    Parameters
    ----------
    y_true : array-like, shape = [n_samples, n_classes].
        Ground truth multi-class targets.

    y_pred : array-like, shape = [n_samples, n_classes].
        Predicted multi-class targets.

    class_hierarchy : the class hierarchy graph, given as a `networkx.DiGraph` instance
        Node ids must be integer and correspond to the indices into the y_true / y_pred matrices.

    beta: float
        the beta parameter for the F-beta score. Defaults to F1 score (beta=1).

    Returns
    -------
    hFscore : float
        The computed hierarchical F-score.

    """
    hP = h_precision_score(y_true, y_pred, class_hierarchy)
    hR = h_recall_score(y_true, y_pred, class_hierarchy)
    return (1. + beta ** 2.) * hP * hR / (beta ** 2. * hP + hR)
