from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Union

import hdf5storage
import numpy as np
import pytest

from neurotechdevkit.results._metrics import (
    Conversions,
    calculate_all_metrics,
    calculate_focal_fwhm,
    calculate_focal_gain,
    calculate_focal_pressure,
    calculate_focal_volume,
    calculate_i_pa_off_target,
    calculate_i_pa_target,
    calculate_i_ta_off_target,
    calculate_i_ta_target,
    calculate_mechanical_index,
)
from neurotechdevkit.results._results import SteadyStateResult2D
from neurotechdevkit.scenarios.built_in import Scenario1_2D

GRID_SHAPE = (21, 31)
CENTER_FREQUENCY = 1.5e6
AMBIENT_PRESSURE = 2.5e6
PEAK_PRESSURE = 6.3e6
PEAK_PRESSURE_SHAPE = (2, 2)
SPEED_OF_SOUND = 1800.0
DENSITY = 1200.0


@pytest.fixture
def patched_metric_fns(monkeypatch):
    """Stub metric functions to avoid depending on their implementation during tests"""
    import neurotechdevkit.results._metrics as metrics

    monkeypatch.setattr(
        metrics, "calculate_mechanical_index", lambda result, layer: 0.1234
    )
    monkeypatch.setattr(
        metrics, "calculate_focal_pressure", lambda result, layer: 8.2e5
    )
    monkeypatch.setattr(
        metrics, "calculate_focal_position", lambda result, layer: (15, 24)
    )
    monkeypatch.setattr(metrics, "calculate_focal_volume", lambda result, layer: 123)
    monkeypatch.setattr(metrics, "calculate_focal_gain", lambda result: 2.4)
    monkeypatch.setattr(metrics, "calculate_focal_fwhm", lambda result, axis, layer: 6)
    monkeypatch.setattr(metrics, "calculate_i_ta_target", lambda result: 2.34e-3)
    monkeypatch.setattr(metrics, "calculate_i_ta_off_target", lambda result: 3.45e-3)
    monkeypatch.setattr(metrics, "calculate_i_pa_target", lambda result: 45.6)
    monkeypatch.setattr(metrics, "calculate_i_pa_off_target", lambda result: 56.7)


@pytest.fixture
def fake_steady_state_matrix():
    """Create a matrix that produces easy to evaluate metric values.

    The array has a shape of (21, 31) with a constant value of 2.5e6 everywhere except
    for four pixels which have a value of 6.3e6 in the lower half.
    """
    steady_state = AMBIENT_PRESSURE * np.ones(GRID_SHAPE)
    steady_state[
        15 : (15 + PEAK_PRESSURE_SHAPE[0]), 24 : (24 + PEAK_PRESSURE_SHAPE[1])
    ] = PEAK_PRESSURE
    return steady_state


@pytest.fixture
def fake_result(fake_steady_state_matrix):
    """Returns a fake SteadyStateResult2D that can be used for testing."""
    return SteadyStateResult2D(
        scenario=SimpleNamespace(
            material_masks={
                "water": fake_layer_masks("water"),
                "brain": fake_layer_masks("brain"),
            },
            material_layers=["water", "brain"],
            get_target_mask=lambda: fake_layer_masks("target"),
            get_field_data=fake_fields,
        ),
        center_frequency=CENTER_FREQUENCY,
        effective_dt=1e-4,
        pde=None,
        shot=None,
        wavefield=None,
        traces=None,
        steady_state=fake_steady_state_matrix,
    )


def fake_layer_masks(layer):
    """Returns a mask for a particular layer for testing purposes.

    Details for each layer:
        layer=='water': mask is True in the top half of the mask
        layer=='brain': mask is True in the bottom half of the mask
        layer=='target': mask is True in a small region in the bottom half of the mask
            which surrounds the high pressure area plus a pixel margin.

    Args:
        layer: the name of the layer. One of: water, brain, or target.

    Returns:
        A boolean mask representing the layer.
    """
    mask = np.zeros((21, 31), dtype=bool)
    mid = mask.shape[0] // 2
    if layer == "water":
        mask[:mid] = True
    elif layer == "brain":
        mask[mid:] = True
    elif layer == "target":
        mask[14:18, 23:27] = True
    else:
        raise ValueError(layer)
    return mask


def fake_fields(field):
    if field == "vp":
        return SPEED_OF_SOUND * np.ones((21, 31))
    if field == "rho":
        return DENSITY * np.ones((21, 31))
    raise ValueError(field)


def test_calculate_all_metrics_has_correct_structure(fake_result, patched_metric_fns):
    """Verify that the metric data has the correct structure and keys"""
    metrics = calculate_all_metrics(fake_result)
    for _, data in metrics.items():
        assert set(data.keys()) == {"value", "unit-of-measurement", "description"}
        assert isinstance(data["value"], (float, int, tuple))


def test_calculate_all_metrics_has_expected_metrics(fake_result, patched_metric_fns):
    """Verify that the metrics data contains the expected set of metrics"""
    expected_metrics = [
        "focal_pressure",
        "focal_position",
        "focal_volume",
        "focal_gain",
        "FWHM_x",
        "FWHM_y",
        "mechanical_index_all",
        "mechanical_index_brain",
        "mechanical_index_water",
        "I_ta_target",
        "I_ta_off_target",
        "I_pa_target",
        "I_pa_off_target",
    ]
    metrics = calculate_all_metrics(fake_result)
    assert set(metrics.keys()) == set(expected_metrics)


def test_calculate_all_metrics_conversions(fake_result, patched_metric_fns):
    """Verify that the unit-of-measurement conversion applied to each metric is
    correct.
    """
    metrics = calculate_all_metrics(fake_result)
    np.testing.assert_allclose(metrics["focal_pressure"]["value"], 8.2e5)
    assert metrics["focal_position"]["value"] == (15, 24)
    assert metrics["focal_volume"]["value"] == 123
    np.testing.assert_allclose(metrics["focal_gain"]["value"], 2.4)
    assert metrics["FWHM_x"]["value"] == 6
    assert metrics["FWHM_y"]["value"] == 6
    np.testing.assert_allclose(metrics["mechanical_index_all"]["value"], 0.1234)
    np.testing.assert_allclose(metrics["mechanical_index_brain"]["value"], 0.1234)
    np.testing.assert_allclose(metrics["mechanical_index_water"]["value"], 0.1234)
    np.testing.assert_allclose(metrics["I_ta_target"]["value"], 2.34e-4)
    np.testing.assert_allclose(metrics["I_ta_off_target"]["value"], 3.45e-4)
    np.testing.assert_allclose(metrics["I_pa_target"]["value"], 4.56e-3)
    np.testing.assert_allclose(metrics["I_pa_off_target"]["value"], 5.67e-3)


def test_calculate_mechanical_index_with_no_layer(fake_result):
    """Verify that the mechanical index result is as expected"""
    metric_value = calculate_mechanical_index(fake_result, layer=None)
    expected = PEAK_PRESSURE / np.sqrt(CENTER_FREQUENCY)
    np.testing.assert_allclose(metric_value, expected)


def test_calculate_mechanical_index_with_top_layer(fake_result):
    """Verify that the mechanical index is computed only over the specified masked
    portion of the grid.
    """
    metric_value = calculate_mechanical_index(fake_result, layer="water")
    expected = AMBIENT_PRESSURE / np.sqrt(CENTER_FREQUENCY)
    np.testing.assert_allclose(metric_value, expected)


def test_calculate_mechanical_index_with_bottom_layer(fake_result):
    """Verify that the mechanical index is computed only over the specified masked
    portion of the grid.
    """
    metric_value = calculate_mechanical_index(fake_result, layer="brain")
    expected = PEAK_PRESSURE / np.sqrt(CENTER_FREQUENCY)
    np.testing.assert_allclose(metric_value, expected)


@pytest.mark.parametrize(
    "layer, expected_focal_pressure",
    [(None, PEAK_PRESSURE), ("water", AMBIENT_PRESSURE), ("brain", PEAK_PRESSURE)],
)
def test_calculate_focal_pressure(
    fake_result: SteadyStateResult2D,
    layer: Optional[str],
    expected_focal_pressure: float,
):
    """Verify that the focal pressure is calculated as expected"""
    metric_value = calculate_focal_pressure(fake_result, layer=layer)
    np.testing.assert_allclose(metric_value, expected_focal_pressure)


@pytest.mark.parametrize(
    "layer, expected_focal_volume",
    [(None, np.prod(PEAK_PRESSURE_SHAPE)), ("brain", np.prod(PEAK_PRESSURE_SHAPE))],
)
def test_calculate_focal_volume(
    fake_result: SteadyStateResult2D, layer: Optional[str], expected_focal_volume: int
):
    """Verify that the focal pressure is calculated as expected"""
    metric_value = calculate_focal_volume(fake_result, layer=layer)
    np.testing.assert_allclose(metric_value, expected_focal_volume)


def test_calculate_focal_gain(fake_result: SteadyStateResult2D):
    """Verify that the focal gain is calculated as expected"""
    metric_value = calculate_focal_gain(fake_result)
    # 16 pixels in the target region, 4 with high pressure
    numerator = (PEAK_PRESSURE * 4 + AMBIENT_PRESSURE * 12) / 16
    denominator = AMBIENT_PRESSURE
    expected = 10 * np.log10(numerator / denominator)
    np.testing.assert_allclose(metric_value, expected)


@pytest.mark.parametrize(
    "axis,layer,expected_fwhm",
    [
        (0, None, PEAK_PRESSURE_SHAPE[0]),
        (1, None, PEAK_PRESSURE_SHAPE[1]),
        ("x", "brain", PEAK_PRESSURE_SHAPE[0]),
        ("y", "brain", PEAK_PRESSURE_SHAPE[1]),
    ],
)
def test_calculate_focal_fwhm(
    fake_result: SteadyStateResult2D,
    axis: Union[str, int],
    layer: Optional[str],
    expected_fwhm: int,
):
    """Verify that the full-width at half-maximum is calculated as expected"""
    assert PEAK_PRESSURE > (AMBIENT_PRESSURE * 2), (
        "This test's expected values are invalid if the peak pressure is not"
        " 2x larger than the ambient pressure"
    )
    metric_value = calculate_focal_fwhm(fake_result, axis=axis, layer=layer)
    np.testing.assert_allclose(metric_value, expected_fwhm)


def test_calculate_i_ta_off_target(fake_result):
    """Verify that the time-averaged intensity out of the target is correct"""
    metric_value = calculate_i_ta_off_target(fake_result)
    denominator = 2 * CENTER_FREQUENCY * DENSITY * SPEED_OF_SOUND
    expected = AMBIENT_PRESSURE**2 / denominator
    np.testing.assert_allclose(metric_value, expected)


def test_calculate_i_ta_target(fake_result):
    """Verify that the time-averaged intensity in the target is correct"""
    metric_value = calculate_i_ta_target(fake_result)
    denominator = 2 * CENTER_FREQUENCY * DENSITY * SPEED_OF_SOUND
    # 16 pixels in the target region, 4 with high pressure
    numerator = (PEAK_PRESSURE**2 * 4 + AMBIENT_PRESSURE**2 * 12) / 16
    expected = numerator / denominator
    np.testing.assert_allclose(metric_value, expected)


def test_calculate_i_pa_off_target(fake_result):
    """Verify that the pulse-averaged intensity equals the time-averaged intensity.

    For steady-state, all pulse-averaged metrics should equal time-averaged metrics.
    """
    pulse_averaged = calculate_i_pa_off_target(fake_result)
    time_averaged = calculate_i_ta_off_target(fake_result)
    np.testing.assert_allclose(pulse_averaged, time_averaged)


def test_calculate_i_pa_target(fake_result):
    """Verify that the pulse-averaged intensity equals the time-averaged intensity.

    For steady-state, all pulse-averaged metrics should equal time-averaged metrics.
    """
    pulse_averaged = calculate_i_pa_target(fake_result)
    time_averaged = calculate_i_ta_target(fake_result)
    np.testing.assert_allclose(pulse_averaged, time_averaged)


@pytest.mark.parametrize(
    "from_uom, to_uom, value, expected",
    [
        ("W/m²", "mW/cm²", 46.2, 4.62),
        ("W/m²", "mW/cm²", 3.4e-5, 3.4e-6),
        ("W/m²", "W/cm²", 1.6, 1.6e-4),
        ("W/m²", "W/cm²", 9465.0, 0.9465),
    ],
)
def test_known_conversions(from_uom, to_uom, value, expected):
    """Verify that each known conversion returns the expected result."""
    result = Conversions.convert(from_uom, to_uom, value)
    np.testing.assert_allclose(result, expected)


def test_unknown_conversion():
    """Unknown conversions should raise a ValueError"""
    with pytest.raises(ValueError):
        Conversions.convert("foo", "bar", 2.3)


def test_compare_metrics_to_aubry2022():
    """Tests FWHM calculation against Aubry et al. (2022) code:

    To download the test data, see:
        https://zenodo.org/record/6020543

    To calculate the reference metrics, see:
        https://github.com/agencyenterprise/transcranial-ultrasound-benchmarks/blob/master/strideMetrics.m
    which is forked from:
        https://github.com/ucl-bug/transcranial-ultrasound-benchmarks
    """
    steady_state_file = Path(__file__).parent.joinpath(
        "test_data", "PH1-BM4-SC1_STRIDE.mat"
    )
    steady_state = hdf5storage.loadmat(str(steady_state_file))["p_amp"]

    # _metrics.py expects a Results object
    # Benchmark 4 corresponds to scenario 1
    # The simulation resolution was slightly different, but
    # let's use the object to wrap the data anyway
    scenario = Scenario1_2D()
    scenario.make_grid()
    scenario.compile_problem()
    result = SteadyStateResult2D(
        scenario=scenario,
        steady_state=steady_state,
        center_frequency=scenario.center_frequency,
        effective_dt=None,
        pde=None,
        shot=None,
        wavefield=None,
        traces=None,
    )

    # Calculate metrics on Aubry et al. 2022 results
    metrics = calculate_all_metrics(result)

    # Load in metrics calculated with published MATLAB code
    expected_metrics_file = Path(__file__).parent.joinpath(
        "test_data", "metrics-BM4-SC1_STRIDE.mat"
    )
    expected_metrics = hdf5storage.loadmat(str(expected_metrics_file))

    # Check expected metrics
    np.testing.assert_approx_equal(
        metrics["focal_pressure"]["value"],
        expected_metrics["max_amp_field1"].squeeze(),
    )
    assert (
        metrics["focal_volume"]["value"]
        == expected_metrics["focal_volume_num_vox_field1"].squeeze()
    ), "Focal volume does not match expected"
    for dim in ("x", "y"):
        # FWHM in NDK is calculated as a whole number of pixels,
        # while the MATLAB code estimates the FWHM with interpolation
        np.testing.assert_almost_equal(
            metrics[f"FWHM_{dim}"]["value"],
            expected_metrics[f"FWHM_{dim}_field1"].squeeze(),
            decimal=0,
        )
