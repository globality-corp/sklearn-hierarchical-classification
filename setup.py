#!/usr/bin/env python
from setuptools import find_packages, setup


project = "sklearn-hierarchical-classification"
version = "1.3.1"


setup(
    name=project,
    version=version,
    description="Hierarchical classification interface extensions for scikit-learn",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Globality Engineering",
    author_email="engineering@globality.com",
    url="https://github.com/globality-corp/sklearn-hierarchical-classification",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        "networkx>=2.4",
        "numpy>=1.13.1",
        "scikit-learn>=0.19.0",
        "scipy>=0.19.1",
        "six>=1.10.0",
    ],
    setup_requires=[
        "nose>=1.3.7",
    ],
    tests_require=[
        "coverage>=3.7.1",
        "inflect>=4.0.0",
        "parameterized>=0.7.1",
        "PyHamcrest>=1.9.0",
    ],
)
