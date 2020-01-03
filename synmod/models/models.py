"""Model generation"""

from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.linear_model import LogisticRegression
from mihifepe.simulation.simulation import get_relevant_features, gen_polynomial

from synmod.constants import REGRESSOR
from synmod.models.operations import Average, Max


# pylint: disable = invalid-name
class Classifier(ClassifierMixin):
    """Classifier model"""
    estimators = [LogisticRegression]  # TODO: maybe add other estimators

    def __init__(self, operation, estimator, X, y, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._operation = operation
        # Fit estimator on data (X, y) after applying operation
        self._estimator = estimator()  # TODO: estimator params incl. random_state for reproducibility
        self._estimator.fit(self._operation.operate(X), y)

    def fit(self, X, y):
        """The model is defined in advance, not fitted to data, so throw exception"""
        # TODO: check what happens if this isn't implemented
        raise NotImplementedError("The model is defined in advance, not fitted to data")

    def predict(self, X):
        """Compute outputs on sequences in X by applying operation to features followed by classification using estimator"""
        return self._estimator.predict(self._operation.operate(X))


class Regressor(RegressorMixin):
    """Regression model"""
    def __init__(self, operation, polynomial_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._operation = operation
        self._polynomial_fn = polynomial_fn

    def fit(self, X, y):
        """The model is defined in advance, not fitted to data, so throw exception"""
        # TODO: check what happens if this isn't implemented
        raise NotImplementedError("The model is defined in advance, not fitted to data")

    def predict(self, X):
        """Compute outputs on sequences in X by applying operation to features followed by polynomial"""
        return self._polynomial_fn(self._operation.operate(X).transpose())


def get_model(args, features, sequences):
    """Generate and return model"""
    # Select relevant features
    relevant_feature_ids = get_relevant_features(args)
    # Select time window for each feature
    windows = [get_window(args) if fid in relevant_feature_ids else None for fid, _ in enumerate(features)]
    # Select operation to perform on features
    operation = args.rng.choice([Average, Max])(windows)
    if args.model_type == REGRESSOR:
        # Get polynomial function over relevant features with linear and pairwise interaction terms
        _, _, polynomial_fn = gen_polynomial(args, relevant_feature_ids)
        return Regressor(operation, polynomial_fn)
    # Model type: classifier
    # TODO: maybe allow interaction features for classifier
    estimator = args.rng.choice(Classifier.estimators)
    labels = args.rng.binomial(1, 0.5, size=len(sequences))  # generate random labels
    return Classifier(operation, estimator, sequences, labels)  # fit estimator to random labels


def get_window(args):
    """Randomly select appropriate window for model to operate in"""
    # TODO: allow soft-edged windows (smooth decay of influence of feature values outside window)
    right = args.sequence_length - 1  # Anchor half the windows on the right
    if args.rng.uniform() < 0.5:
        right = args.rng.choice(range(args.sequence_length // 2, args.sequence_length))
    left = args.rng.choice(range(0, right))
    return (left, right)
