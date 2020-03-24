"""List of constants"""

# Misc
SEED = 94756812597749967251918284421845484907  # using np.random.SeedSequence.entropy

# Feature
DISCRETE = "discrete"
CONTINUOUS = "continuous"
BINARY = "binary"
CATEGORICAL = "categorical"

# Models
CLASSIFIER = "classifier"
REGRESSOR = "regressor"

# Synthesis types
STATIC = "static"
TEMPORAL = "temporal"

# Output files
FEATURES_FILENAME = "features.cpkl"
MODEL_FILENAME = "model.cpkl"
INSTANCES_FILENAME = "instances.npy"
