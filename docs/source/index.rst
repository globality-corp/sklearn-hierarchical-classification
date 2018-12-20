.. sklearn-hierarchical-classification documentation master file, created by
   sphinx-quickstart on Tue Sep 19 10:55:04 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

sklearn-hierarchical-classification: Hierarchical Classification with scikit-learn
==================================================================================

sklearn-hierarchical-classification is a scikit-learn compatible implementation of hierarchical classification.

Hierarchical classification is a useful paradigm whenever we are trying to classify data into a set of target labels for which internal structure exists; Typically, this structure can be expressed as a graph, e.g. as a tree or more generally as a DAG (`directed acyclic graph <https://en.wikipedia.org/wiki/Directed_acyclic_graph>`_).

A good example of such a large-scale hierarchical text classification task is the `LSHTC Challenge <https://www.kaggle.com/c/lshtc>`_ on Kaggle; This example actually exemplifies a few other challenges that are often associated with hierarchical classification:

* The space of target labels can be very large; a taxonomy of categories can easily grow into the tens of thousands
* The task can be `multi-label <https://en.wikipedia.org/wiki/Multi-label_classification>`_ - a single data point can be associated with multiple target labels
* The hiearchy itself in theory could contain cycles when viewed as a graph; Currently, this package does not support such cases, although support might be added in the future.



This package leverages standardized conventions and naming that was introduced by the excellent survey paper [REF001]_.

.. [REF001] `A survey of hierarchical classification across different application domains <https://www.researchgate.net/publication/225716424_A_survey_of_hierarchical_classification_across_different_application_domains>`_ - CN Silla et al. 2011


.. automodule:: sklearn_hierarchical_classification
   :members:

-------------------

**Sub-Modules:**

.. toctree::

   sklearn_hierarchical.classifier
   sklearn_hierarchical.metrics


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
