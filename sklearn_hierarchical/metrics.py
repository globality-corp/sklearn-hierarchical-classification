"""
Evaluation metrics for hierarchical classification.

"""
import numpy as np


def transitive_closure(y, class_hierarchy):
    """
    Compute the full transitive closure of the ancestor set
    based on the given class hierarchy graph, for a given label set y
    corresponding to nodes in the class hierarchy (classes).

    """
    # TODO


def h_precision_score(y_true, y_pred, class_hierarchy):
    # TODO
    pass


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

    y_true : (sparse) array-like, shape = [n_samples, n_classes].
        Ground truth multi-class targets.

    y_pred : (sparse) array-like, shape = [n_samples, n_classes].
        Predicted multi-class targets.

    graph : the class hierarchy graph, given as a `networkx.DiGraph` instance

    """

    y_true_ = transitive_closure(y_true, graph=class_hierarchy)
    y_pred_ = transitive_closure(y_pred, graph=class_hierarchy)

    ix = np.where((y_true_ != 0) & (y_pred_ != 0))

    true_positives = len(ix[0])
    all_positives = np.count_nonzero(y_true_)

    return true_positives / all_positives


def h_f1_score(y_true, class_hierarchy):
    # TODO
    pass
