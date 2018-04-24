"""
Constants.

"""

# Special id reserved for an artificial 'root node' that may be added to class hierarchy
# when using a 'one classifier per parent node' strategy.
ROOT = -1

# Dictionary keys used in various places by classifier
CLASSIFIER = "classifier"
DEFAULT = "default"
METAFEATURES = "metafeatures"

# Enumeration of valid configuration types
VALID_ALGORITHM = ("lcn", "lcpn")
VALID_PREDICTION_DEPTH = ("mlnp", "nmlnp")
VALID_TRAINING_STRATEGY = ("exclusive", "less_exclusive", "inclusive", "less_inclusive",
                           "siblings", "exclusive_siblings")
