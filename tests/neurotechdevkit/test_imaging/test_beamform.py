import pytest

import numpy as np
from scipy.sparse import csr_array

from neurotechdevkit.imaging.beamform import (
    beamform_delay_and_sum, delay_and_sum_matrix, InterpolationMethod
)


@pytest.fixture
def simple_inputs():
    fs = 1e5  # Hz
    inputs = {
        'num_time_samples': 10,
        'num_channels': 5,
        'x': np.array([[0, 1], [0, 1]]) * 1e-4,  # meters
        'z': np.array([[1, 1], [2, 2]]) * 1e-4,  # meters
        'pitch': 0.5,
        'tx_delays': np.arange(5) / fs,
        'fs': fs,
        'fc': fs / 4,
    }
    return inputs


def test_delay_and_sum_matrix(simple_inputs):
    das_matrix = delay_and_sum_matrix(**simple_inputs)
    assert isinstance(das_matrix, csr_array)
    assert das_matrix.shape == (
        simple_inputs["x"].size,
        simple_inputs["num_time_samples"] * simple_inputs["num_channels"]
    )


def test_delay_and_sum_matrix_outside_aperture(simple_inputs):
    # Error-case where image is outside of aperture
    simple_inputs["x"] = np.array([[1, 2], [1, 2]]) * 1e3
    simple_inputs["f_number"] = 1.5
    with pytest.raises(ValueError):
        delay_and_sum_matrix(**simple_inputs)


def test_delay_and_sum(simple_inputs):
    das_matrix = delay_and_sum_matrix(**simple_inputs)
    num_time_samples = simple_inputs["num_time_samples"]
    num_channels = simple_inputs["num_channels"]

    iq_signals = (
        np.random.rand(num_time_samples, num_channels) +
        1j * np.random.rand(num_time_samples, num_channels)
    )
    del simple_inputs["num_time_samples"]
    del simple_inputs["num_channels"]

    beamformed_iq_signals = beamform_delay_and_sum(iq_signals, **simple_inputs)
    assert beamformed_iq_signals.shape == simple_inputs["x"].shape
