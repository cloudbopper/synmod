"""Master pipeline"""

# pylint: disable = fixme, unused-argument, unused-variable, unused-import

import argparse
from distutils.util import strtobool
import os
import pickle

import cloudpickle
import numpy as np

from synmod import constants
from synmod import features as F
from synmod import models as M
from synmod.utils import get_logger


def main():
    """Parse args and launch pipeline"""
    parser = argparse.ArgumentParser("python synmod")
    # Required arguments
    required = parser.add_argument_group("Required parameters")
    required.add_argument("-output_dir", help="Output directory", required=True)
    required.add_argument("-num_features", help="Number of features",
                          type=int, required=True)
    required.add_argument("-num_instances", help="Number of instances",
                          type=int, required=True)
    required.add_argument("-synthesis_type", help="Type of data/model synthesis to perform",
                          default=constants.TEMPORAL, choices=[constants.TEMPORAL, constants.STATIC])
    # Optional common arguments
    common = parser.add_argument_group("Common optional parameters")
    common.add_argument("-fraction_relevant_features", help="Fraction of features relevant to model",
                        type=float, default=1)
    common.add_argument("-num_interactions", help="number of pairwise in aggregation model (default 0)",
                        type=int, default=0)
    common.add_argument("-include_interaction_only_features", help="include interaction-only features in aggregation model"
                        " in addition to linear + interaction features (excluded by default)", type=strtobool)
    common.add_argument("-seed", help="Seed for RNG, random by default",
                        default=None, type=int)
    common.add_argument("-write_outputs", help="flag to enable writing outputs (alternative to using python API)",
                        type=strtobool)
    # Temporal synthesis arguments
    temporal = parser.add_argument_group("Temporal synthesis parameters")
    temporal.add_argument("-sequence_length", help="Length of regularly sampled sequence",
                          type=int, required=True)
    temporal.add_argument("-sequences_independent_of_windows", help="If enabled, Markov chain sequence data doesn't depend on timesteps being"
                          " inside vs. outside the window (default random)", type=strtobool, dest="window_independent")
    temporal.set_defaults(window_independent=None)
    # TODO: model_type should be common to both synthesis types
    temporal.add_argument("-model_type", help="type of model (classifier/regressor) - default random",
                          choices=[constants.CLASSIFIER, constants.REGRESSOR], default=None)
    args = parser.parse_args()
    return pipeline(args)


def config(args):
    """Configure arguments before execution"""
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.rng = np.random.default_rng(args.seed)
    args.logger = get_logger(__name__, "%s/synmod.log" % args.output_dir)
    if args.model_type is None:
        args.model_type = args.rng.choice([constants.CLASSIFIER, constants.REGRESSOR])
    if args.window_independent is None:
        args.window_independent = args.rng.choice([True, False])


def pipeline(args):
    """Pipeline"""
    config(args)
    args.logger.info("Begin generating sequence data with args: %s" % args)
    aggregation_fn = M.get_aggregation_fn(args)
    features = generate_features(args, aggregation_fn)
    instances = generate_instances(args, features)
    model = M.get_model(args, features, instances, aggregation_fn)
    write_outputs(args, features, instances, model)
    return features, instances, model


def generate_features(args, aggregation_fn):
    """Generate features"""
    def check_feature_variance(args, feature, aggregation_fn):
        """Check variance of feature's raw/temporally aggregated values"""
        instances = np.array([feature.sample(args.sequence_length) for _ in range(constants.VARIANCE_TEST_COUNT)])
        aggregated = instances
        if args.synthesis_type == constants.TEMPORAL:
            left, right = feature.window
            aggregated = aggregation_fn.operate_on_feature(instances[:, left: right + 1])
        return np.var(aggregated) > 1e-10

    # TODO: allow across-feature interactions
    features = [None] * args.num_features
    fid = 0
    while fid < args.num_features:
        feature = F.get_feature(args, str(fid), aggregation_fn)
        if not check_feature_variance(args, feature, aggregation_fn):
            # Reject feature if its raw/aggregated values have low variance
            args.logger.info(f"Rejecting feature {feature.__class__} due to low variance")
            continue
        features[fid] = feature
        fid += 1
    return features


def generate_instances(args, features):
    """Generate instances"""
    if args.synthesis_type == constants.STATIC:
        instances = np.empty((args.num_instances, args.num_features), dtype=np.int64)
        for sid in range(args.num_instances):
            instances[sid] = [feature.sample() for feature in features]
    else:
        instances = np.empty((args.num_instances, args.num_features, args.sequence_length))
        for sid in range(args.num_instances):
            instances[sid] = [feature.sample(args.sequence_length) for feature in features]
    return instances


def write_outputs(args, features, instances, model):
    """Write outputs to file"""
    if not args.write_outputs:
        return
    with open(f"{args.output_dir}/{constants.FEATURES_FILENAME}", "wb") as features_file:
        cloudpickle.dump(features, features_file, protocol=pickle.DEFAULT_PROTOCOL)
    np.save(f"{args.output_dir}/{constants.INSTANCES_FILENAME}", instances)
    with open(f"{args.output_dir}/{constants.MODEL_FILENAME}", "wb") as model_file:
        cloudpickle.dump(model, model_file, protocol=pickle.DEFAULT_PROTOCOL)


def generate_labels(args, model, instances):
    """Generate labels"""
    # TODO: decide how to handle multivariate case
    # TODO: joint generation of labels and features
    return model.predict(instances)


if __name__ == "__main__":
    main()
