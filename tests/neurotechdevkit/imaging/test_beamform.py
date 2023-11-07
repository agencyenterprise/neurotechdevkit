from typing import Any

import numpy as np
import pytest
import xarray as xr
from scipy.sparse import csr_array

from neurotechdevkit.imaging.beamform import (
    InterpolationMethod,
    _calculate_time_of_flight,
    _directivity,
    _optimize_f_number,
    beamform_delay_and_sum,
    delay_and_sum_matrix,
)


@pytest.fixture
def simple_inputs() -> dict[str, Any]:
    fs = 1e5  # Hz
    inputs = {
        "num_time_samples": 10,
        "num_channels": 5,
        "x": np.array([[0, 1, 2], [0, 1, 2]]) * 1e-4,  # meters
        "z": np.array([[0, 0, 0], [1, 1, 1]]) * 1e-4,  # meters
        "pitch": 0.5,
        "tx_delays": np.arange(5) / fs,
        "freq_sampling": fs,
        "freq_carrier": fs / 4,
        "speed_sound": 1540,
    }
    return inputs


@pytest.fixture
def optimize_f_number_inputs() -> dict[str, Any]:
    inputs = {
        "element_width": 1e-4,
        "bandwidth_fractional": 0.5,
        "freq_carrier": 1e6,
        "speed_sound": 1540,
    }
    return inputs


@pytest.fixture
def xarray_inputs(simple_inputs: dict[str, Any]) -> dict[str, Any]:
    """Used as input to _calculate_time_of_flight()"""
    IMG_DIMS = ["z", "x"]
    num_channels = simple_inputs["num_channels"]
    pitch = simple_inputs["pitch"]
    depth_pixels, width_pixels = simple_inputs["x"].shape

    inputs = {
        "x_dataarray": xr.DataArray(simple_inputs["x"], dims=IMG_DIMS),
        "z_dataarray": xr.DataArray(simple_inputs["z"], dims=IMG_DIMS),
        "x_channels": xr.DataArray(
            (np.arange(num_channels) - (num_channels - 1) / 2) * pitch,
            dims=("channel",),
        ),
        "z_channels": xr.DataArray(np.zeros(num_channels), dims=("channel",)),
        "tx_delays_dataarray": xr.DataArray(
            simple_inputs["tx_delays"], dims=("channel",)
        ),
        "speed_sound": simple_inputs["speed_sound"],
        "width_pixels": width_pixels,
        "depth_pixels": depth_pixels,
        "num_channels": num_channels,
    }

    return inputs


# Tests for main functions
def test_delay_and_sum_matrix(simple_inputs):
    das_matrix = delay_and_sum_matrix(**simple_inputs)
    assert isinstance(das_matrix, csr_array)
    assert das_matrix.shape == (
        simple_inputs["x"].size,
        simple_inputs["num_time_samples"] * simple_inputs["num_channels"],
    )


def test_delay_and_sum_matrix_outside_aperture(simple_inputs):
    # Error-case where image is outside of aperture
    simple_inputs["x"] = np.array([[1, 2, 3], [1, 2, 3]]) * 1e3
    simple_inputs["f_number"] = 1.5
    with pytest.raises(ValueError):
        delay_and_sum_matrix(**simple_inputs)


def test_delay_and_sum(simple_inputs):
    num_time_samples = simple_inputs["num_time_samples"]
    num_channels = simple_inputs["num_channels"]

    iq_signals = np.random.rand(num_time_samples, num_channels) + 1j * np.random.rand(
        num_time_samples, num_channels
    )
    del simple_inputs["num_time_samples"]
    del simple_inputs["num_channels"]

    beamformed_iq_signals = beamform_delay_and_sum(iq_signals, **simple_inputs)
    assert beamformed_iq_signals.shape == simple_inputs["x"].shape


def test_delay_and_sum_nearest_neighbors_interpolation(simple_inputs):
    simple_inputs["method"] = InterpolationMethod.NEAREST

    num_time_samples = simple_inputs["num_time_samples"]
    num_channels = simple_inputs["num_channels"]

    iq_signals = np.random.rand(num_time_samples, num_channels) + 1j * np.random.rand(
        num_time_samples, num_channels
    )
    del simple_inputs["num_time_samples"]
    del simple_inputs["num_channels"]

    beamformed_iq_signals = beamform_delay_and_sum(iq_signals, **simple_inputs)
    assert beamformed_iq_signals.shape == simple_inputs["x"].shape


def test_expect_complex(simple_inputs):
    # create a scenario where AssertionError is raised when iq_signals is not complex.
    with pytest.raises(AssertionError):
        num_time_samples = simple_inputs["num_time_samples"]
        num_channels = simple_inputs["num_channels"]

        iq_signals = np.ones((num_time_samples, num_channels), dtype=float)
        beamform_delay_and_sum(iq_signals, **simple_inputs)


# Tests for helper functions
def test_optimize_f_number(optimize_f_number_inputs):
    """Test that the f-number is a positive float"""
    f_number = _optimize_f_number(**optimize_f_number_inputs)
    assert isinstance(f_number, float), "F-number should be a float"
    assert f_number > 0, "F-number should be positive"


def test_directivity(optimize_f_number_inputs):
    """Test that directivity is a non-negative float"""
    speed_sound = optimize_f_number_inputs["speed_sound"]
    freq_carrier = optimize_f_number_inputs["freq_carrier"]
    element_width = optimize_f_number_inputs["element_width"]

    # Example receive angle between -pi/2 and pi/2
    theta = np.pi / 4
    assert (
        (-np.pi / 2) < theta < (np.pi / 2)
    ), "Receive angle should be between -pi/2 and pi/2 for this example."
    # Example wavelength
    wavelength = speed_sound / freq_carrier
    directivity = _directivity(theta, element_width, wavelength)

    assert isinstance(directivity, float), "Directivity should be a float"
    assert directivity > 0, "Directivity should be positive for non-perpendicular angle"

    # Example perpendicular receive angle of pi/2
    theta = np.pi / 2
    directivity = _directivity(theta, element_width, wavelength)
    np.testing.assert_almost_equal(directivity, 0, decimal=8)


def test_calculate_time_of_flight(xarray_inputs):
    """Test that the time of flight is a non-negative float"""
    tof = _calculate_time_of_flight(**xarray_inputs)
    assert isinstance(tof, xr.DataArray), "Time of flight should be a DataArray"
    assert tof.dtype == float, "Time of flight should be a float"
    assert (tof >= 0).all(), "Time of flight should be non-negative"
    assert tof.z.isel(z=-1) == tof.z.max(), "Expected z to be monotonically increasing"
    assert (
        tof.isel(z=-1) >= tof
    ).all(), "Time of flight should be monotonically increasing in depth"
