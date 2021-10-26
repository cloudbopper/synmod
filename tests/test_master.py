"""Tests for master script"""
import json
import subprocess
import sys
from unittest.mock import patch

import cloudpickle
import numpy as np

import synmod
from synmod import master, constants
from tests.utils import pre_test, post_test, round_fp


# pylint: disable = invalid-name, redefined-outer-name, protected-access
def test_regressor1(tmpdir, caplog):
    """Test synthetic data generation"""
    output_dir = pre_test(sys._getframe().f_code.co_name, tmpdir, caplog)
    cmd = ("python -m synmod -synthesis_type temporal -model_type regressor -num_instances 100 -num_features 10 -sequence_length 20 "
           f"-fraction_relevant_features 0.5 -include_interaction_only_features 1 -output_dir {output_dir} -seed {constants.SEED}")
    pass_args = cmd.split()[2:]
    with patch.object(sys, 'argv', pass_args):
        master.main()
    post_test(caplog, output_dir)


def test_subprocess1(tmpdir, caplog):
    """Test synthetic data generation"""
    output_dir = pre_test(sys._getframe().f_code.co_name, tmpdir, caplog)
    cmd = ("python -m synmod -synthesis_type temporal -model_type regressor -num_instances 100 -num_features 10 -sequence_length 20 "
           f"-fraction_relevant_features 0.5 -include_interaction_only_features 1 -output_dir {output_dir} -seed {constants.SEED}")
    subprocess.check_call(cmd, shell=True)
    post_test(caplog, output_dir)


def test_classifier1(tmpdir, caplog):
    """Test synthetic data generation"""
    output_dir = pre_test(sys._getframe().f_code.co_name, tmpdir, caplog)
    cmd = ("python -m synmod -synthesis_type temporal -model_type classifier -num_instances 100 -num_features 10 -sequence_length 20 "
           f"-fraction_relevant_features 0.5 -include_interaction_only_features 1 -output_dir {output_dir} -seed {constants.SEED}")
    pass_args = cmd.split()[2:]
    with patch.object(sys, 'argv', pass_args):
        master.main()
    post_test(caplog, output_dir)


def test_reproducible_classifier(tmpdir, data_regression, caplog):
    """Reproducibility of results regression test"""
    output_dir = pre_test(sys._getframe().f_code.co_name, tmpdir, caplog)
    cmd = ("python -m synmod -synthesis_type temporal -model_type classifier -num_instances 100 -num_features 10 -sequence_length 20 "
           f"-fraction_relevant_features 0.8 -include_interaction_only_features 1 -output_dir {output_dir} -seed {constants.SEED}")
    pass_args = cmd.split()[2:]
    with patch.object(sys, 'argv', pass_args):
        _, data, model = master.main()
    post_test(caplog, output_dir)
    labels = model.predict(data, labels=True)
    data_regression.check(round_fp(data).tobytes() + labels.tobytes())


def test_reproducible_regressor(tmpdir, data_regression, caplog):
    """Reproducibility of results regression test"""
    output_dir = pre_test(sys._getframe().f_code.co_name, tmpdir, caplog)
    cmd = ("python -m synmod -synthesis_type temporal -model_type regressor -num_instances 100 -num_features 10 -sequence_length 20 "
           f"-fraction_relevant_features 0.8 -include_interaction_only_features 1 -output_dir {output_dir} -seed {constants.SEED}")
    pass_args = cmd.split()[2:]
    with patch.object(sys, 'argv', pass_args):
        _, data, model = master.main()
    post_test(caplog, output_dir)
    labels = model.predict(data)
    data_regression.check(round_fp(data).tobytes() + round_fp(labels).tobytes())


def test_reproducible_write_outputs(tmpdir, data_regression, file_regression, caplog):
    """Regression test to test reproducible human-readable summary of config/model/features and output files"""
    output_dir = pre_test(sys._getframe().f_code.co_name, tmpdir, caplog)
    cmd = ("python -m synmod -synthesis_type temporal -model_type classifier -num_instances 100 -num_features 10 -sequence_length 20 "
           f"-fraction_relevant_features 0.8 -include_interaction_only_features 1 -write_outputs 1 -output_dir {output_dir} -seed {constants.SEED}")
    pass_args = cmd.split()[2:]
    with patch.object(sys, 'argv', pass_args):
        master.main()
    data = np.load(f"{output_dir}/{constants.INSTANCES_FILENAME}")
    post_test(caplog, output_dir)
    with open(f"{output_dir}/{constants.SUMMARY_FILENAME}", "rb") as summary_file:
        summary = json.load(summary_file)
    file_regression.check(json.dumps(round_fp(summary), indent=2), extension=".json")
    with open(f"{output_dir}/{constants.MODEL_FILENAME}", "rb") as model_file:
        model = cloudpickle.load(model_file)
    labels = model.predict(data, labels=True)
    data_regression.check(round_fp(data).tobytes() + labels.tobytes())


def test_reproducible_standardize_features(tmpdir, data_regression, file_regression, caplog):
    """Regression test to test reproducibility with standardized features"""
    output_dir = pre_test(sys._getframe().f_code.co_name, tmpdir, caplog)
    cmd = ("python -m synmod -synthesis_type temporal -model_type classifier -num_instances 100 -num_features 10 -sequence_length 20 "
           "-fraction_relevant_features 0.8 -include_interaction_only_features 1 -write_outputs 1 "
           f"-standardize_features 1 -output_dir {output_dir} -seed {constants.SEED}")
    pass_args = cmd.split()[2:]
    with patch.object(sys, 'argv', pass_args):
        master.main()
    data = np.load(f"{output_dir}/{constants.INSTANCES_FILENAME}")
    post_test(caplog, output_dir)
    with open(f"{output_dir}/{constants.SUMMARY_FILENAME}", "rb") as summary_file:
        summary = json.load(summary_file)
    file_regression.check(json.dumps(round_fp(summary), indent=2), extension=".json")
    with open(f"{output_dir}/{constants.MODEL_FILENAME}", "rb") as model_file:
        model = cloudpickle.load(model_file)
    labels = model.predict(data, labels=True)
    data_regression.check(round_fp(data).tobytes() + labels.tobytes())


def test_interface(tmpdir, caplog):
    """Test API"""
    output_dir = pre_test(sys._getframe().f_code.co_name, tmpdir, caplog)
    _ = synmod.synthesize(output_dir=output_dir, num_features=2, num_instances=10, synthesis_type=constants.TEMPORAL, sequence_length=5)
