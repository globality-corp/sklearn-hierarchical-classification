# sklearn-hierarchical-classification

Hierarchical classification module based on scikit-learn's interfaces and conventions.


## Installation

To install, simply install this package via pip into your desired virtualenv, e.g:

    pip install sklearn-hierarchical-classification


## Usage

See [examples/](./examples/) for usage examples.


### Jupyter notebooks

Support for interactive development is built in to the `HierarchicalClassifier` class. This will enable progress bars (using the excellent [tqdm](https://pypi.python.org/pypi/tqdm) library) in various places during training and may otherwise enable more visibility into the classifier which is useful during interactive use. To enable this make sure widget extensions are enabled by running:

    jupyter nbextension enable --py --sys-prefix widgetsnbextension

You can then instantiate a classifier with the `interactive=True` flag set:

```python
clf = HierarchicalClassifier(
    base_estimator=svm.LinearSVC(),
    class_hierarchy=class_hierarchy,
    interactive=True,
)
```


## Documentation

Auto-generated documentation is provided via sphinx. To build / view:

    $ cd docs/
    $ make html
    $ open _build/index.html


## Further Reading

this module is heavily influenced by the following previous work and papers:

* ["Functional Annotation of Genes Using Hierarchical Text Categorization"](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.68.5824&rep=rep1&type=pdf) - Kiritchenko et al. 2005
* ["A survey of hierarchical classification across different application domains"](https://www.researchgate.net/publication/225716424_A_survey_of_hierarchical_classification_across_different_application_domains) - CN Silla et al. 2011
* ["A Survey of Automated Hierarchical Classification of Patents"](https://lirias.kuleuven.be/bitstream/123456789/457904/1/GomezMoens%20Mumia_book_chapter_camera_ready2014.pdf) - JC Gomez et al. 2014
* ["Bayesian Aggregation for Hierarchical Classification"](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.89.3312&rep=rep1&type=pdf) - Barutcuoglu et al. 2008
* ["Kaggle LSHTC4 Winning Solution"](https://kaggle2.blob.core.windows.net/forum-message-attachments/43550/1230/lshtc4.pdf) - Puurula et al. 2014
* ["Feature-Weighted Linear Stacking"](https://arxiv.org/pdf/0911.0460.pdf) - Sill et al. 2009
