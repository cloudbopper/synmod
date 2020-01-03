"""Master pipeline"""

# pylint: disable = fixme, unused-argument, unused-variable, unused-import

import argparse
import os

import numpy as np
from mihifepe import utils
from mihifepe.constants import EPSILON_IRRELEVANT, ADDITIVE_GAUSSIAN, NO_NOISE

from synmod.constants import CLASSIFIER, REGRESSOR
from synmod.features import features as F
from synmod.models import models as M


def main():
    """Parse args and launch pipeline"""
    parser = argparse.ArgumentParser()
    # Required arguments
    parser.add_argument("-output_dir", help="Output directory", required=True)
    parser.add_argument("-sequence_length", help="Length of regularly sampled sequence",
                        type=int, required=True)
    parser.add_argument("-num_sequences", help="Number of sequences (instances)",
                        type=int, required=True)
    parser.add_argument("-num_features", help="Number of features",
                        type=int, required=True)
    parser.add_argument("-fraction_relevant_features", help="Fraction of features relevant to model",
                        type=float, default=1)
    parser.add_argument("-num_interactions", help="number of pairwise in aggregation model (default 0)",
                        type=int, default=0)
    parser.add_argument("-include_interaction_only_features", help="include interaction-only features in aggregation model"
                        " in addition to linear + interaction features (excluded by default)", action="store_true")
    # FIXME: noise_type is needed to generate polynomial, but not used by model.predict.
    parser.add_argument("-noise_type", help="type of noise to add to aggregation model (default none)",
                        choices=[EPSILON_IRRELEVANT, ADDITIVE_GAUSSIAN, NO_NOISE], default=NO_NOISE)
    parser.add_argument("-model_type", help="type of model (classifier/regressor) - default random",
                        choices=[CLASSIFIER, REGRESSOR], default=None)
    parser.add_argument("-seed", help="Seed for RNG, random by default",
                        default=None, type=int)
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.rng = np.random.default_rng(np.random.SeedSequence(args.seed))
    args.logger = utils.get_logger(__name__, "%s/master.log" % args.output_dir)
    if not args.model_type:
        args.model_type = args.rng.choice([CLASSIFIER, REGRESSOR])
    sequences, labels = pipeline(args)
    return sequences, labels


def pipeline(args):
    """Pipeline"""
    args.logger.info("Begin generating sequence data with args with args: %s" % args)
    features = generate_features(args)
    sequences = generate_sequences(args, features)
    model = generate_model(args, features, sequences)
    labels = generate_labels(args, model, sequences)
    return sequences, labels


def generate_features(args):
    """Generate features"""
    # TODO: allow across-feature interactions
    features = []
    for fid in range(args.num_features):
        features.append(F.get_feature(args, str(fid)))
    return features


def generate_sequences(args, features):
    """Generate sequences"""
    sequences = np.empty((args.num_sequences, args.sequence_length, args.num_features))
    for sid in range(args.num_sequences):
        subseq = [feature.sample(args.sequence_length) for feature in features]
        sequences[sid] = np.transpose(subseq)
    return sequences


def generate_model(args, features, sequences):
    """Generate model"""
    return M.get_model(args, features, sequences)


def generate_labels(args, model, sequences):
    """Generate labels"""
    # TODO: decide how to handle multivariate case
    # TODO: joint generation of labels and features
    return model.predict(sequences)


if __name__ == "__main__":
    main()
