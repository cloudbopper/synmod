"""Model generation"""

from sklearn.base import RegressorMixin
from mihifepe.simulation.simulation import get_relevant_features

from synmod.models.operations import Average, Max


# pylint: disable = invalid-name
# TODO: add classifier
class Regressor(RegressorMixin):
    """Regression model"""
    operations = [Average, Max]  # TODO: add other ops

    def __init__(self, operation, *args, **kwargs):
        self._operation = operation
        super().__init__(*args, **kwargs)

    def fit(self, X, y):
        """The model is defined in advance, not fitted to data, so throw exception"""
        # TODO: check what happens if this isn't implemented
        raise NotImplementedError("The model is defined in advance, not fitted to data")

    def predict(self, X):
        """Compute outputs on sequences in X by applying property"""
        return self._operation.operate(X)


def get_model(args, features):
    """Generate and return model"""
    # Get relevant features
    relevant_feature_ids = get_relevant_features(args)
    # Select time window for each feature
    windows = [get_window(args) if fid in relevant_feature_ids else None for fid, _ in enumerate(features)]
    # Select model type
    model_type = args.rng.choice([Regressor])  # TODO: include classifier
    # Select model operation
    operation = args.rng.choice(model_type.operations)(windows)
    # Instantiate and return model
    return model_type(operation)


def get_window(args):
    """Randomly select appropriate window for model to operate in"""
    # TODO: allow soft-edged windows (smooth decay of influence of feature values outside window)
    right = args.sequence_length - 1  # Anchor half the windows on the right
    if args.rng.uniform() < 0.5:
        right = args.rng.choice(range(args.sequence_length // 2, args.sequence_length))
    left = args.rng.choice(range(0, right))
    return (left, right)
