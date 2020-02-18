"""Test validation logic."""
from hamcrest import assert_that, calling, raises
from parameterized import param, parameterized

from sklearn_hierarchical_classification.tests.fixtures import make_classifier_and_data


# Each test case below correpsonds to classifier constructor arguments
VALIDATION_TEST_CASES = [
    param(
        prediction_depth="nmlnp",
        stopping_criteria=None
    ),
    param(
        prediction_depth="nmlnp",
        stopping_criteria="not_a_float_or_a_callable",
    ),
    param(
        prediction_depth="mlnp",
        stopping_criteria=123.4,
    ),
    param(
        prediction_depth="some_invalid_prediction_depth_value",
    ),
    param(
        algorithm="lcn",
        training_strategy=None,
    ),
    param(
        algorithm="lcn",
        training_strategy="some_invalid_training_strategy",
    ),
    param(
        algorithm="lcpn",
        training_strategy="exclusive",
    ),
    param(
        algorithm="some_invalid_algorithm_value",
    ),
]


@parameterized(VALIDATION_TEST_CASES)
def test_parameter_validation(**classifier_kwargs):
    """Test parameter validation checks for consistent assignment."""
    clf, (X, y) = make_classifier_and_data(**classifier_kwargs)
    assert_that(calling(clf.fit).with_args(X=X, y=y), raises(TypeError))
