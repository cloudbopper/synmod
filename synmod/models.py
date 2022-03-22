"""Model generation"""

from abc import ABC
from collections import namedtuple
from copy import copy
import functools
import itertools

import numpy as np
from scipy.special import expit  # pylint: disable = no-name-in-module
import sympy
from sympy.utilities.lambdify import lambdify

from synmod import constants
from synmod.aggregators import Aggregator, TabularAggregator

Polynomial = namedtuple("Polynomial", ["relevant_feature_map", "sym_polynomial_fn", "polynomial_fn"])


# pylint: disable = invalid-name
class Model(ABC):
    """Model base class"""
    def __init__(self, aggregator, polynomial, X=None):
        # pylint: disable = unused-argument
        self._aggregator = aggregator  # object to perform aggregation over time and generate feature vector
        # relevant_feature-map: Mapping from frozensets containing one or more feature names to their polynomial coefficients
        self.relevant_feature_map, self.sym_polynomial_fn, self._polynomial_fn = polynomial

    @property
    def relevant_feature_names(self):
        """Convenience function to get feature names"""
        return list(functools.reduce(set.union, self.relevant_feature_map, set()))

    def predict(self, X, **kwargs):
        """Predict outputs on input instances"""


class Classifier(Model):
    """Classification model"""
    def __init__(self, aggregator, polynomial, X):
        super().__init__(aggregator, polynomial)
        assert X is not None
        self._threshold = np.median(self._polynomial_fn(self._aggregator.operate(X).transpose(), 0))

    def predict(self, X, **kwargs):
        """
        Predict output probabilities on instances in X by aggregating features over time, applying a polynomial,
        thresholding, then applying a sigmoid.

        Parameters
        ----------
        X: Matrix/tensor
            Instances to predict model outputs for
        labels: bool, optional, default False
            Flag to return output labels instead of probabilities
        noise: 1D float array, optional, default 0
            Noise term(s) to add to polynomial before applying sigmoid
        """
        labels = kwargs.get("labels", False)
        noise = kwargs.get("noise", 0)
        values = expit(self._polynomial_fn(self._aggregator.operate(X).transpose(), noise) - self._threshold)  # Sigmoid output
        if labels:
            values = (values > 0.5).astype(np.int32)
        return values


class Regressor(Model):
    """Regression model"""
    def predict(self, X, **kwargs):
        """
        Predict outputs on instances in X by aggregating features over time and applying a polynomial

        Parameters
        ----------
        X: Matrix/tensor
            Instances to predict model outputs for
        noise: 1D float array, optional, default 0
            Noise term(s) to add to polynomial
        """
        noise = kwargs.get("noise", 0)  # TODO: this is the noise multiplier
        return self._polynomial_fn(self._aggregator.operate(X).transpose(), noise)


def get_model(args, features, instances):
    """Generate and return model"""
    args = copy(args)
    args.rng = np.random.default_rng(args.seed)  # Reset RNG for consistent model independent of instances
    # Select relevant features
    relevant_features = get_relevant_features(args)
    polynomial = gen_polynomial(args, relevant_features)
    if args.synthesis_type == constants.TABULAR:
        aggregator = TabularAggregator()
    else:
        # Select time window for each feature
        windows = [feature.window for feature in features]
        for fid, _ in enumerate(features):
            relevance = "relevant" if fid in relevant_features else "irrelevant"
            args.logger.info(f"Window for {relevance} feature id {fid}: ({windows[fid][0]}, {windows[fid][1]})")
        aggregator = Aggregator([feature.aggregation_fn for feature in features], windows, instances, args.standardize_features)
    # Select model
    model_class = {constants.CLASSIFIER: Classifier, constants.REGRESSOR: Regressor}[args.model_type]
    return model_class(aggregator, polynomial, instances)


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
    sym_features = sympy.symbols([f"x_{x}" for x in range(args.num_features)])
    sym_noise = sympy.Symbol("beta", real=True)  # multiplier for irrelevant features in approximate model
    relevant_feature_map = {}  # map of relevant feature sets to coefficients
    # Generate polynomial expression
    # Pairwise interaction terms
    sym_polynomial_fn = 0
    sym_polynomial_fn = update_interaction_terms(args, relevant_features, relevant_feature_map, sym_features, sym_polynomial_fn)
    # Linear terms
    sym_polynomial_fn = update_linear_terms(args, relevant_features, relevant_feature_map, sym_features, sym_noise, sym_polynomial_fn)
    args.logger.info(f"Ground truth polynomial:\ny = {sym_polynomial_fn}")
    # Generate model expression
    polynomial_fn = lambdify([sym_features, sym_noise], sym_polynomial_fn, "numpy")
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


# pylint: disable = too-many-arguments
def update_linear_terms(args, relevant_features, relevant_feature_map, sym_features, sym_noise, sym_polynomial_fn):
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
    coefficients = sym_noise * np.ones(args.num_features)
    coefficients[list(relevant_features)] = 1
    coefficients *= args.rng.uniform(-1, 1, size=args.num_features)
    for linear_feature in linear_features:
        relevant_feature_map[frozenset([linear_feature])] = coefficients[linear_feature]
    sym_polynomial_fn += coefficients.dot(sym_features)
    return sym_polynomial_fn
