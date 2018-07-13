"""Test validation logic."""
from hamcrest import (
    assert_that,
    calling,
    raises,
)

from sklearn_hierarchical_classification.tests.fixtures import make_classifier_and_data


def test_parameter_validation():
    """Test parameter validation checks for consistent assignment."""
    test_cases = [
        dict(
            prediction_depth="nmlnp",
            stopping_criteria=None,
        ),
        dict(
            prediction_depth="nmlnp",
            stopping_criteria="not_a_float_or_a_callable",
        ),
        dict(
            prediction_depth="mlnp",
            stopping_criteria=123.4,
        ),
        dict(
            prediction_depth="some_invalid_prediction_depth_value",
        ),
        dict(
            algorithm="lcn",
            training_strategy=None,
        ),
        dict(
            algorithm="lcn",
            training_strategy="some_invalid_training_strategy",
        ),
        dict(
            algorithm="lcpn",
            training_strategy="exclusive",
        ),
        dict(
            algorithm="some_invalid_algorithm_value",
        ),
    ]

    for classifier_kwargs in test_cases:
        clf, (X, y) = make_classifier_and_data(**classifier_kwargs)
        assert_that(calling(clf.fit).with_args(X=X, y=y), raises(TypeError))
