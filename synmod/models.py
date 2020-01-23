"""Model generation"""

import functools
import itertools
import sympy
from sympy.utilities.lambdify import lambdify

import numpy as np
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.linear_model import LogisticRegression

from synmod import constants
from synmod.operations import Average, Max, Identity


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
        """Compute outputs on instances in X by applying operation to features followed by classification using estimator"""
        return self._estimator.predict(self._operation.operate(X))

    @staticmethod
    def loss(y_true, y_pred):
        """Loss function used by classifier"""
        return -np.log(y_pred) if y_true else -np.log(y_pred)  # Binary cross-entropy


class Regressor(RegressorMixin):
    """Regression model"""
    def __init__(self, operation, polynomial_fn, relevant_feature_map, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._operation = operation
        self._polynomial_fn = polynomial_fn
        self.relevant_feature_map = relevant_feature_map  # map of feature id's to polynomial coefficients

    def fit(self, X, y):
        """The model is defined in advance, not fitted to data, so throw exception"""
        # TODO: check what happens if this isn't implemented
        raise NotImplementedError("The model is defined in advance, not fitted to data")

    def predict(self, X):
        """Compute outputs on instances in X by applying operation to features followed by polynomial"""
        return self._polynomial_fn(self._operation.operate(X).transpose())

    @staticmethod
    def loss(y_true, y_pred):
        """Loss function used by regressor"""
        # Don't use sklearn.metrics.mean_squared_error here - it's much slower
        return np.abs(y_true - y_pred)  # RMSE - possibly replace with MSE


def get_model(args, features, instances):
    """Generate and return model"""
    # Select relevant features
    relevant_features = get_relevant_features(args)
    if args.synthesis_type == constants.STATIC:
        relevant_feature_map, polynomial_fn = gen_polynomial(args, relevant_features)
        return Regressor(Identity(), polynomial_fn, relevant_feature_map)
    # Select time window for each feature
    windows = [get_window(args) if fid in relevant_features else None for fid, _ in enumerate(features)]
    # Select operation to perform on features
    operation = args.rng.choice([Average, Max])(windows)
    if args.model_type == constants.REGRESSOR:
        # Get polynomial function over relevant features with linear and pairwise interaction terms
        relevant_feature_map, polynomial_fn = gen_polynomial(args, relevant_features)
        return Regressor(operation, polynomial_fn, relevant_feature_map)
    # Model type: classifier
    # TODO: allow interaction features for classifier
    estimator = args.rng.choice(Classifier.estimators)
    labels = args.rng.binomial(1, 0.5, size=len(instances))  # generate random labels
    return Classifier(operation, estimator, instances, labels)  # fit estimator to random labels


def get_window(args):
    """Randomly select appropriate window for model to operate in"""
    # TODO: allow soft-edged windows (smooth decay of influence of feature values outside window)
    right = args.sequence_length - 1  # Anchor half the windows on the right
    if args.rng.uniform() < 0.5:
        right = args.rng.choice(range(args.sequence_length // 2, args.sequence_length))
    left = args.rng.choice(range(0, right))
    return (left, right)


def gen_polynomial(args, relevant_features):
    """Generate polynomial which decides the ground truth and noisy model"""
    # Note: using sympy to build function appears to be 1.5-2x slower than erstwhile raw numpy implementation (for linear terms)
    # TODO: possibly negative coefficients
    sym_features = sympy.symbols(["x%d" % x for x in range(args.num_features)])
    relevant_feature_map = {}  # map of relevant feature sets to coefficients
    # Generate polynomial expression
    # Pairwise interaction terms
    sym_polynomial_fn = 0
    sym_polynomial_fn = update_interaction_terms(args, relevant_features, relevant_feature_map, sym_features, sym_polynomial_fn)
    # Linear terms
    sym_polynomial_fn = update_linear_terms(args, relevant_features, relevant_feature_map, sym_features, sym_polynomial_fn)
    args.logger.info("Ground truth polynomial:\ny = %s" % sym_polynomial_fn)
    # Generate model expression
    polynomial_fn = lambdify([sym_features], sym_polynomial_fn, "numpy")
    return relevant_feature_map, polynomial_fn


def get_relevant_features(args):
    """Get set of relevant feature identifiers"""
    num_relevant_features = max(1, round(args.num_features * args.fraction_relevant_features))
    coefficients = np.zeros(args.num_features)
    coefficients[:num_relevant_features] = 1
    args.rng.shuffle(coefficients)
    relevant_features = {idx for idx in range(args.num_features) if coefficients[idx]}
    return relevant_features


def update_interaction_terms(args, relevant_features, relevant_feature_map, sym_features, sym_polynomial_fn):
    """Pairwise interaction terms for polynomial"""
    # TODO: higher-order interactions
    num_relevant_features = len(relevant_features)
    num_interactions = min(args.num_interactions, num_relevant_features * (num_relevant_features - 1) / 2)
    if not num_interactions:
        return sym_polynomial_fn
    potential_pairs = list(itertools.combinations(sorted(relevant_features), 2))
    potential_pairs_arr = np.empty(len(potential_pairs), dtype=np.object)
    potential_pairs_arr[:] = potential_pairs
    interaction_pairs = args.rng.choice(potential_pairs_arr, size=num_interactions, replace=False)
    for interaction_pair in interaction_pairs:
        coefficient = args.rng.uniform()
        relevant_feature_map[frozenset(interaction_pair)] = coefficient
        sym_polynomial_fn += coefficient * functools.reduce(lambda sym_x, y: sym_x * sym_features[y], interaction_pair, 1)
    return sym_polynomial_fn


def update_linear_terms(args, relevant_features, relevant_feature_map, sym_features, sym_polynomial_fn):
    """Order one terms for polynomial"""
    interaction_features = set()
    for interaction in relevant_feature_map.keys():
        interaction_features.update(interaction)
    # Let half the interaction features have nonzero interaction coefficients but zero linear coefficients
    interaction_only_features = []
    if interaction_features and args.include_interaction_only_features:
        interaction_only_features = args.rng.choice(sorted(interaction_features),
                                                    len(interaction_features) // 2,
                                                    replace=False)
    linear_features = sorted(relevant_features.difference(interaction_only_features))
    coefficients = np.zeros(args.num_features)
    coefficients[linear_features] = args.rng.uniform(size=len(linear_features))
    for linear_feature in linear_features:
        relevant_feature_map[frozenset([linear_feature])] = coefficients[linear_feature]
    sym_polynomial_fn += coefficients.dot(sym_features)
    return sym_polynomial_fn
