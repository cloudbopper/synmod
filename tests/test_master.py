"""Tests for master script"""

import subprocess
import sys
from unittest.mock import patch

from synmod import master, constants


# pylint: disable = invalid-name, redefined-outer-name, protected-access
def get_output_dir(tmpdir, func_name):
    """Get unique output directory name"""
    return "{0}/output_dir_{1}".format(tmpdir, func_name)


def test_main1(tmpdir):
    """Test synthetic data generation"""
    output_dir = get_output_dir(tmpdir, sys._getframe().f_code.co_name)
    cmd = ("python -m synmod -num_sequences 100 -num_features 10 -sequence_length 20 "
           "-output_dir {0} -seed {1}".format(output_dir, constants.SEED))
    pass_args = cmd.split()[2:]
    with patch.object(sys, 'argv', pass_args):
        master.main()


def test_main2(tmpdir):
    """Test synthetic data generation"""
    output_dir = get_output_dir(tmpdir, sys._getframe().f_code.co_name)
    cmd = ("python -m synmod.master -num_sequences 100 -num_features 10 -sequence_length 20 "
           "-output_dir {0} -seed {1}".format(output_dir, constants.SEED))
    subprocess.check_call(cmd, shell=True)
