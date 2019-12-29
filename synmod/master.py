"""Master pipeline"""

# pylint: disable = fixme, unused-argument, unused-variable, unused-import

import argparse
import logging
import os

import numpy as np

from synmod import constants
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

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    np.random.seed(constants.SEED)
    logging.basicConfig(level=logging.INFO, filename="%s/master.log" % args.output_dir,
                        format="%(asctime)s: %(message)s")
    logger = logging.getLogger(__name__)
    args.logger = logger
    sequences, labels = pipeline(args)
    return sequences, labels


def pipeline(args):
    """Pipeline"""
    args.logger.info("Begin generating sequence data with args with args: %s" % args)
    features = generate_features(args)
    model = generate_model(args, features)
    sequences = generate_sequences(args, features)
    labels = generate_labels(args, model, sequences)
    return sequences, labels


def generate_features(args):
    """Generate features"""
    # TODO: allow across-feature interactions
    features = []
    for fid in range(args.num_features):
        features.append(F.get_feature(str(fid)))
    return features


def generate_sequences(args, features):
    """Generate sequences"""
    sequences = np.empty((args.num_sequences, args.sequence_length, args.num_features))
    for sid in range(args.num_sequences):
        subseq = [feature.sample(args.sequence_length) for feature in features]
        sequences[sid] = np.transpose(subseq)
    return sequences


def generate_model(args, features):
    """Generate model"""
    return M.get_model(args, features)


def generate_labels(args, model, sequences):
    """Generate labels"""
    # TODO: decide how to handle multivariate case
    # TODO: joint generation of labels and features
    return model.predict(sequences)


if __name__ == "__main__":
    main()
