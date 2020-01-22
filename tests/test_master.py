"""Tests for master script"""

import subprocess
import sys
from unittest.mock import patch

from synmod import master, constants


# pylint: disable = invalid-name, redefined-outer-name, protected-access
def get_output_dir(tmpdir, func_name):
    """Get unique output directory name"""
    return "{0}/output_dir_{1}".format(tmpdir, func_name)


def test_regressor1(tmpdir):
    """Test synthetic data generation"""
    output_dir = get_output_dir(tmpdir, sys._getframe().f_code.co_name)
    cmd = ("python -m synmod -model_type regressor -num_sequences 100 -num_features 10 -sequence_length 20 "
           "-fraction_relevant_features 0.5 -include_interaction_only_features -output_dir {0} -seed {1}"
           .format(output_dir, constants.SEED))
    pass_args = cmd.split()[2:]
    with patch.object(sys, 'argv', pass_args):
        master.main()


def test_subprocess1(tmpdir):
    """Test synthetic data generation"""
    output_dir = get_output_dir(tmpdir, sys._getframe().f_code.co_name)
    cmd = ("python -m synmod -model_type regressor -num_sequences 100 -num_features 10 -sequence_length 20 "
           "-fraction_relevant_features 0.5 -include_interaction_only_features -output_dir {0} -seed {1}"
           .format(output_dir, constants.SEED))
    subprocess.check_call(cmd, shell=True)


def test_classifier1(tmpdir):
    """Test synthetic data generation"""
    output_dir = get_output_dir(tmpdir, sys._getframe().f_code.co_name)
    cmd = ("python -m synmod -model_type classifier -num_sequences 100 -num_features 10 -sequence_length 20 "
           "-fraction_relevant_features 0.5 -include_interaction_only_features -output_dir {0} -seed {1}"
           .format(output_dir, constants.SEED))
    pass_args = cmd.split()[2:]
    with patch.object(sys, 'argv', pass_args):
        master.main()


def test_reproducible1(tmpdir, data_regression):
    """Reproducibility of results regression test"""
    output_dir = get_output_dir(tmpdir, sys._getframe().f_code.co_name)
    cmd = ("python -m synmod -model_type classifier -num_sequences 100 -num_features 10 -sequence_length 20 "
           "-fraction_relevant_features 0.8 -include_interaction_only_features -output_dir {0} -seed {1}"
           .format(output_dir, constants.SEED))
    pass_args = cmd.split()[2:]
    with patch.object(sys, 'argv', pass_args):
        sequences, labels = master.main()
    data_regression.check(sequences.tostring() + labels.tostring())
