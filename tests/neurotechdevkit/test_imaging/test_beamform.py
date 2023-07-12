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
        'delaysTX': np.arange(5) / fs,
        'fs': fs,
        'fc': fs / 4,
    }
    return inputs


def test_delay_and_sum_matrix(simple_inputs):
    das_matrix = delay_and_sum_matrix(**simple_inputs)
    assert isinstance(das_matrix, csr_array)
    assert das_matrix.get_shape() == (
        simple_inputs["x"].size,
        simple_inputs["num_time_samples"] * simple_inputs["num_channels"]
    )
