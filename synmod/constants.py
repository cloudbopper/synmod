"""List of constants"""

# Misc
SEED = 94756812597749967251918284421845484907  # using np.random.SeedSequence.entropy

# Feature
DISCRETE = "discrete"
NUMERIC = "numeric"
BINARY = "binary"
CATEGORICAL = "categorical"
VARIANCE_TEST_COUNT = 100  # number of instances to test a feature's variance with

# Models
CLASSIFIER = "classifier"
REGRESSOR = "regressor"

# Synthesis types
TABULAR = "tabular"
TEMPORAL = "temporal"

# Output files
FEATURES_FILENAME = "features.cpkl"
MODEL_FILENAME = "model.cpkl"
INSTANCES_FILENAME = "instances.npy"
SUMMARY_FILENAME = "summary.json"
