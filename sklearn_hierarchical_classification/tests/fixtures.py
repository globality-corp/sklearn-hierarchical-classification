"""
Unit-test fixtures and factory methods.

"""
from itertools import product

import numpy as np
from networkx import DiGraph, gn_graph, to_dict_of_lists
from sklearn.datasets import load_digits, make_blobs

from sklearn_hierarchical_classification.classifier import HierarchicalClassifier
from sklearn_hierarchical_classification.constants import ROOT


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
    """Create a classifier as well as a synthetic dataset, with optional support for
    user-specific class hierarchy.

    """
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


def make_classifier_and_data_own_preprocessing():
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.pipeline import make_pipeline
    
    newsgroups_train = fetch_20newsgroups(subset='train')
    X , Y= newsgroups_train.data, newsgroups_train.target

    vectorizer = TfidfVectorizer(
                strip_accents=None,
                lowercase=True,
                analyzer='word',
                ngram_range=(1, 3),
                max_df=1.0,
                min_df=0.0,
                binary=False,
                use_idf=True,
                smooth_idf=True,
                sublinear_tf=True,
                max_features=70000
            )
    hierarchy={}
    hierarchy[ROOT]=["alt","comp","rec","misc","sci","soc","talk"]
    hierarchy["alt"]=['alt.atheism']
    hierarchy["comp"]= ['comp.graphics',
                        'comp.os.ms-windows.misc',
                        'comp.sys.ibm.pc.hardware',
                        'comp.sys.mac.hardware',
                        'comp.windows.x']
    hierarchy["misc"]=[ 'misc.forsale']
    hierarchy["rec"]=[ 'rec.autos',
                      'rec.motorcycles',
                      'rec.sport.baseball',
                      'rec.sport.hockey']
    hierarchy["sci"]= ['sci.crypt',
                       'sci.electronics',
                       'sci.med',
                       'sci.space']
    hierarchy["soc"]=[ 'soc.religion.christian']
    hierarchy["talk"]=[ 'talk.politics.guns',
                       'talk.politics.mideast',
                       'talk.politics.misc',
                       'talk.religion.misc']

    class_hierarchy = hierarchy
    
    names = newsgroups_train.target_names
    bclf = OneVsRestClassifier(LinearSVC())
    base_estimator = make_pipeline(
        vectorizer, bclf)
    labels = [ [names[tk]]+[names[tk].split(".")[0]] for tk in Y]
    mlb = MultiLabelBinarizer()
    y=mlb.fit_transform(labels)
    
    clf = make_classifier(
        base_estimator=base_estimator,
        class_hierarchy=class_hierarchy,
        algorithm="lcn",training_strategy= "siblings",
        preprocessing=True,
        mlb=mlb,
        use_decision_function=True
    )


    return clf, (X, y)


def make_clothing_graph(root=ROOT):
    """Create a mock hierarchy of clothing items."""
    G = DiGraph()
    G.add_edge(root, "Mens")
    G.add_edge("Mens", "Shirts")
    G.add_edge("Mens", "Bottoms")
    G.add_edge("Mens", "Jackets")
    G.add_edge("Mens", "Swim")

    return G


def make_clothing_graph_and_data(root=ROOT):
    """Create a graph for hierarchical classification
    of clothing items, along with mock training data.

    """
    G = make_clothing_graph(root)

    labels = list(G.nodes() - [root])
    y = np.random.choice(labels, size=50)
    X = np.random.normal(size=(len(y), 10))

    return G, (X, y)
