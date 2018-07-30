"""Validation helpers."""
from sklearn_hierarchical_classification.constants import (
    VALID_ALGORITHM,
    VALID_PREDICTION_DEPTH,
    VALID_TRAINING_STRATEGY,
)


class ParameterValidator(object):
    """Parameter validation logic for the HierarchicalClassifier class."""
    def __init__(self, instance):
        self.instance = instance

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def __call__(self):
        return self._validate()

    def _validate(self):
        if self.algorithm not in VALID_ALGORITHM:
            raise TypeError(
                "'algorithm' must be set to one of: {}.".format(
                    ", ".join(VALID_ALGORITHM),
                )
            )

        if (self.algorithm == "lcn") ^ bool(self.training_strategy):
            raise TypeError(
                """When 'algorithm' is set to "lcn", 'training_strategy' must be set
                to a float or callable. Conversly, training_strategy should not be specified
                when algorithm is not set to "lcn"."""
            )

        if self.training_strategy and self.training_strategy not in VALID_TRAINING_STRATEGY:
            raise TypeError(
                "'training_strategy' must be set to one of: {}.".format(
                    ", ".join(VALID_TRAINING_STRATEGY),
                )
            )

        if self.prediction_depth not in VALID_PREDICTION_DEPTH:
            raise TypeError(
                "'prediction_depth' must be set to one of: {}.".format(
                    ", ".join(VALID_PREDICTION_DEPTH),
                )
            )

        if (self.prediction_depth == "nmlnp") ^ bool(self.stopping_criteria):
            raise TypeError(
                """When 'prediction_depth' is set to "nmlnp", 'stopping_criteria' must be set
                to a float or callable. Conversly, stopping_criteria should not be specified
                when prediction_depth is not set to "nmlnp"."""
            )

        if self.stopping_criteria and not any((
            isinstance(self.stopping_criteria, float),
            callable(self.stopping_criteria),
        )):
            raise TypeError(
                """'stopping_criteria' must be set to a float or a callable."""
            )


def validate_parameters(instance):
    return ParameterValidator(instance)()


def is_estimator(obj):
    if hasattr(obj.__class__, "fit"):
        return True
