import numpy as np
import pytest
import stride

from neurotechdevkit.grid import Grid
from neurotechdevkit.scenarios._utils import (
    choose_wavelet_for_mode,
    create_grid_circular_mask,
    create_grid_elliptical_mask,
    create_grid_spherical_mask,
    drop_column,
    drop_element,
    slice_field,
    wavelet_helper,
)


@pytest.fixture
def field_fake():
    """A 3D test fake for a field"""
    return np.arange(60).reshape((3, 4, 5))


@pytest.fixture
def scenario_fake():
    class FakeScenario:
        origin = np.array([0.0, -0.3, -0.2])
        dx = 0.1
        shape = np.array([3, 4, 5])
        extent = np.array([0.3, 0.4, 0.5])

    return FakeScenario()


@pytest.mark.parametrize(
    "position, expected_idx",
    [
        (0.0, 0),
        (0.1, 1),
        (0.2, 2),
        (-0.15, 0),
        (0.45, 2),
        (0.12, 1),
        (0.18, 2),
    ],
)
def test_slice_field_first_axis(field_fake, scenario_fake, position, expected_idx):
    """Verify that slices along the first axis use the expected slice index."""
    result = slice_field(field_fake, scenario_fake, 0, position)
    np.testing.assert_array_equal(result, field_fake[expected_idx, :, :])


@pytest.mark.parametrize("position, expected_idx", [(0.1, 3), (-0.4, 0)])
def test_slice_field_second_axis(field_fake, scenario_fake, position, expected_idx):
    """Verify that slices along the second axis use the expected slice index.

    Positions aligned with the grid and rounding were already tested along the first
    axis, so we're just testing the range of bounds.
    """
    result = slice_field(field_fake, scenario_fake, 1, position)
    np.testing.assert_array_equal(result, field_fake[:, expected_idx, :])


@pytest.mark.parametrize("position, expected_idx", [(-0.4, 0), (0.5, 4)])
def test_slice_field_third_axis(field_fake, scenario_fake, position, expected_idx):
    """Verify that slices along the third axis use the expected slice index.

    Positions aligned with the grid and rounding were already tested along the first
    axis, so we're just testing the range of bounds.
    """
    result = slice_field(field_fake, scenario_fake, 2, position)
    np.testing.assert_array_equal(result, field_fake[:, :, expected_idx])


@pytest.mark.parametrize("idx", [0, 1, 2])
def test_drop_element(idx):
    """Verify that the correct element is dropped for each possible index."""
    test_vector = np.array([2.0, 7.0, 42.0])
    result = drop_element(test_vector, idx)
    expected = test_vector[[n for n in range(3) if n != idx]]
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("idx", [0, 1, 2])
def test_drop_column(idx):
    """Verify that the correct column is dropped for each possible index."""
    test_array = np.arange(18).reshape((6, 3))
    result = drop_column(test_array, idx)
    expected = test_array[:, [n for n in range(3) if n != idx]]
    np.testing.assert_array_equal(result, expected)


def test_create_grid_elliptical_mask():
    """Verify that the returned mask has the expected elements turned on."""
    grid = Grid.make_grid(
        extent=(0.6, 0.8),
        speed_water=100000,
        ppw=2,
        center_frequency=5e5,
    )
    origin = np.array([-0.2, -0.3])
    center = np.array([0.05, 0.15])
    mask = create_grid_elliptical_mask(grid, origin, center, a=0.2, b=0.3)

    expected = np.zeros((7, 9), dtype=bool)
    expected[1, 3:7] = True
    expected[2:4, 2:8] = True
    expected[4, 3:7] = True

    np.testing.assert_allclose(mask, expected)


def test_create_grid_circular_mask():
    """Verify that the returned mask has the expected elements turned on."""
    grid = Grid.make_grid(
        extent=(0.6, 0.8),
        speed_water=100000,
        ppw=2,
        center_frequency=5e5,
    )
    origin = np.array([-0.2, -0.3])
    center = np.array([0.05, 0.15])
    mask = create_grid_circular_mask(grid, origin, center, radius=0.2)

    expected = np.zeros((7, 9), dtype=bool)
    expected[1, 4:6] = True
    expected[2:4, 3:7] = True
    expected[4, 4:6] = True

    np.testing.assert_allclose(mask, expected)


def test_create_grid_spherical_mask():
    """Verify that the returned mask has the expected elements turned on."""
    grid = Grid.make_grid(
        extent=(0.6, 0.8, 1.0),
        speed_water=100000,
        ppw=2,
        center_frequency=5e5,
    )
    origin = np.array([-0.2, -0.3, 0.0])
    center = np.array([0.05, 0.15, 0.45])
    mask = create_grid_spherical_mask(grid, origin, center, radius=0.2)

    expected = np.zeros((7, 9, 11), dtype=bool)
    expected[1, 4:6, 4:6] = True
    expected[2:4, 3:7, 4:6] = True
    expected[4, 4:6, 4:6] = True
    expected[2:4, 4:6, 3] = True
    expected[2:4, 4:6, 6] = True

    np.testing.assert_allclose(mask, expected)


@pytest.mark.parametrize(
    "sim_mode, expected_name",
    [("steady-state", "continuous_wave"), ("pulsed", "tone_burst")],
)
def test_choose_wavelet_for_mode(sim_mode, expected_name):
    """Verify that returned wavelet name is correct."""
    wavelet_name = choose_wavelet_for_mode(sim_mode)
    assert wavelet_name == expected_name


def test_choose_wavelet_for_mode_with_invalid_mode():
    """Verify that a ValueError is raised if the simulation mode is invalid."""
    with pytest.raises(ValueError):
        _ = choose_wavelet_for_mode("invalid")


def test_wavelet_helper_continuous_wave_frequency():
    """Verify that the continuous wave wavelet has the expected frequency.

    To verify the frequency, we compare the wavelet against a sinusoidal wave at the
    desired frequency. In reality, this only validates dt * freq_hz, but it shows that
    the results are consistent with the given freq_hz and dt.

    The wavelet has a transient at the start where the amplitude ramps up from 0, so we
    are only comparing a few wavelengths of data at the end of the wavelet.
    """
    freq_hz = 5.0e5
    dt = 1.0e-7
    wavelet = wavelet_helper(
        "continuous_wave",
        time=stride.Time(start=0.0, step=dt, num=1000),
        freq_hz=freq_hz,
        pressure=1.0,
    )
    ten_cycles = wavelet[-200:]
    t = np.arange(200) * dt
    expected = np.sin(2 * np.pi * freq_hz * t)
    np.testing.assert_allclose(ten_cycles, expected, rtol=0, atol=1e-6)


def test_wavelet_helper_continuous_wave_time_properties():
    """Verify that the time step and time length of the wavelet are correct.

    We can't directly verify the dt of the wavelet, but we can verify that the wavelet
    size is consistent with the specified start time, stop time, and dt.
    """
    dt = 1.0e-7
    n_desired = 1234
    time = stride.Time(start=0.0, stop=dt * (n_desired - 1), step=dt)
    wavelet = wavelet_helper(
        "continuous_wave",
        time=time,
    )
    assert wavelet.shape == (n_desired,)
