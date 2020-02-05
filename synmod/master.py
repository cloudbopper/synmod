"""Master pipeline"""

# pylint: disable = fixme, unused-argument, unused-variable, unused-import

import argparse
import os

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
                        " in addition to linear + interaction features (excluded by default)", action="store_true")
    common.add_argument("-seed", help="Seed for RNG, random by default",
                        default=None, type=int)
    # Temporal synthesis arguments
    temporal = parser.add_argument_group("Temporal synthesis parameters")
    temporal.add_argument("-sequence_length", help="Length of regularly sampled sequence",
                          type=int, required=True)
    # TODO: model type should be common to both synthesis types
    temporal.add_argument("-model_type", help="type of model (classifier/regressor) - default random",
                          choices=[constants.CLASSIFIER, constants.REGRESSOR], default=None)
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.rng = np.random.default_rng(args.seed)
    args.logger = get_logger(__name__, "%s/synmod.log" % args.output_dir)
    if not args.model_type:
        args.model_type = args.rng.choice([constants.CLASSIFIER, constants.REGRESSOR])
    return pipeline(args)


def pipeline(args):
    """Pipeline"""
    args.logger.info("Begin generating sequence data with args with args: %s" % args)
    features = generate_features(args)
    instances = generate_instances(args, features)
    model = generate_model(args, features, instances)
    return features, instances, model


def generate_features(args):
    """Generate features"""
    # TODO: allow across-feature interactions
    features = []
    for fid in range(args.num_features):
        features.append(F.get_feature(args, str(fid)))
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


def generate_model(args, features, instances):
    """Generate model"""
    return M.get_model(args, features, instances)


def generate_labels(args, model, instances):
    """Generate labels"""
    # TODO: decide how to handle multivariate case
    # TODO: joint generation of labels and features
    return model.predict(instances)


if __name__ == "__main__":
    main()
