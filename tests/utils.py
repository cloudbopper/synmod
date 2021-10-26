"""Utility functions for tests"""

import logging

import numpy as np


def setup_logfile(caplog):
    """Set up logfile for test"""
    caplog.set_level(logging.INFO)
    formatting = "%(asctime)s: %(levelname)s: %(name)s: %(message)s"
    handler = caplog.handler
    handler.setFormatter(logging.Formatter(formatting))


def post_test(caplog, output_dir):
    """Post-test management"""
    # Write log file
    log_filename = f"{output_dir}/test.log"
    with open(log_filename, "w", encoding="utf-8") as log_file:
        log_file.write(caplog.text)
    caplog.clear()


def pre_test(func_name, tmpdir, caplog):
    """Pre-test setup"""
    output_dir = f"{tmpdir}/output_dir_{func_name}"
    setup_logfile(caplog)
    return output_dir


def round_fp(data):
    """Round FP values in data to avoid regression test errors due to FP precision variations across platforms"""
    dtype = type(data)
    if dtype == float:
        return round(data, 8)
    elif dtype == np.ndarray:
        return np.round(data, 8)
    elif dtype == dict:
        for key, value in data.items():
            data[key] = round_fp(value)
    elif dtype == list:
        return [round_fp(item) for item in data]
    return data
