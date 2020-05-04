"""Model generation"""

from abc import ABC
from collections import namedtuple
import functools
import itertools

import numpy as np
from scipy.special import expit  # pylint: disable = no-name-in-module
import sympy
from sympy.utilities.lambdify import lambdify

from synmod import constants
from synmod.operations import Average, Max, Identity

Polynomial = namedtuple("Polynomial", ["relevant_feature_map", "sym_polynomial_fn", "polynomial_fn"])


# pylint: disable = invalid-name
class Model(ABC):
    """Model base class"""
    def __init__(self, operation, polynomial, X=None):
        # pylint: disable = unused-argument
        self._operation = operation  # operation to perform aggregation over time and generate feature vector
        # relevant_feature-map: Mapping from frozensets containing one or more feature names to their polynomial coefficients
        self.relevant_feature_map, self.sym_polynomial_fn, self._polynomial_fn = polynomial

    @property
    def relevant_feature_names(self):
        """Convenience function to get feature names"""
        return list(functools.reduce(set.union, self.relevant_feature_map, set()))

    def predict(self, X, **kwargs):
        """Predict outputs on input instances"""

    @staticmethod
    def loss(y_true, y_pred):
        """Compute loss vector for given target-prediction pairs"""


class Classifier(Model):
    """Classification model"""
    def __init__(self, operation, polynomial, X):
        super().__init__(operation, polynomial)
        assert X is not None
        self._threshold = np.median(self._polynomial_fn(self._operation.operate(X).transpose()))

    def predict(self, X, **kwargs):
        """
        Predict output probabilities on instances in X by aggregating features over time, applying a polynomial,
        thresholding, then applying a sigmoid. If 'labels' is asserted, return output labels
        """
        labels = kwargs.get("labels", False)
        values = expit(self._polynomial_fn(self._operation.operate(X).transpose()) - self._threshold)  # Sigmoid output
        if labels:
            values = (values > 0.5).astype(np.int32)
        return values

    @staticmethod
    def loss(y_true, y_pred):
        """Logistic loss"""
        # TODO: 0-1 loss
        # TODO: Handle case when y_pred components are 1 or 0 (due to very small/large sigmoid inputs)
        return -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)  # Binary cross-entropy


class Regressor(Model):
    """Regression model"""
    def predict(self, X, **kwargs):
        """Predict outputs on instances in X by aggregating features over time and applying a polynomial"""
        return self._polynomial_fn(self._operation.operate(X).transpose())

    @staticmethod
    def loss(y_true, y_pred):
        """RMSE loss"""
        # Don't use sklearn.metrics.mean_squared_error here - it's much slower
        return np.abs(y_true - y_pred)  # RMSE - possibly replace with MSE


def get_model(args, features, instances):
    """Generate and return model"""
    # Select relevant features
    relevant_features = get_relevant_features(args)
    polynomial = gen_polynomial(args, relevant_features)
    if args.synthesis_type == constants.STATIC:
        return Regressor(Identity(), polynomial)
    # Select time window for each feature
    windows = [feature.window if fid in relevant_features else None for fid, feature in enumerate(features)]
    for fid in relevant_features:
        args.logger.info("Window for feature id %d: (%d, %d)" % (fid, windows[fid][0], windows[fid][1]))
    # Select operation to perform on features
    operation = args.rng.choice([Average, Max])()
    operation.set_windows(windows)
    # Select model
    model_class = {constants.CLASSIFIER: Classifier, constants.REGRESSOR: Regressor}[args.model_type]
    return model_class(operation, polynomial, instances)


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
    return Polynomial(relevant_feature_map, sym_polynomial_fn, polynomial_fn)


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
        if args.model_type == constants.CLASSIFIER:
            coefficient *= args.rng.choice([-1, 1])  # Randomly flip sign
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
    if args.model_type == constants.CLASSIFIER:
        # TODO: always flip sign randomly, even for regression
        coefficients[linear_features] *= args.rng.choice([-1, 1], size=len(linear_features))  # Randomly flip sign
    for linear_feature in linear_features:
        relevant_feature_map[frozenset([linear_feature])] = coefficients[linear_feature]
    sym_polynomial_fn += coefficients.dot(sym_features)
    return sym_polynomial_fn
