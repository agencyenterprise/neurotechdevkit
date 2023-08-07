import numpy as np
import pytest

from neurotechdevkit.grid import Grid
from neurotechdevkit.scenarios import materials
from neurotechdevkit.scenarios._time import (
    create_time_grid,
    find_largest_delay_in_sources,
    select_simulation_time_for_pulsed,
    select_simulation_time_for_steady_state,
)
from neurotechdevkit.sources import PhasedArraySource2D


def a_source_to_test(delay=0.0, tilt_angle=0.0, focal_length=np.inf, pitch=1.0):
    """Returns a Source instance that can be used for testing."""
    return PhasedArraySource2D(
        position=np.array([0.0, 0.1]),
        direction=np.array([1.0, 0.0]),
        tilt_angle=tilt_angle,
        focal_length=focal_length,
        num_points=10,
        num_elements=2,
        element_width=1.0,
        pitch=pitch,
        delay=delay,
    )


@pytest.fixture
def scenario_fake():
    """A scenario with simple parameters that can be used for testing."""

    class FakeScenario:
        origin = np.array([0.0, -0.3, -0.2])
        shape = np.array([3, 4, 5])
        extent = np.array([0.3, 0.4, 0.5])

    return FakeScenario()


@pytest.fixture
def grid_fake(scenario_fake):
    """A grid built on top of scenario_fake that can be used for testing."""
    return Grid.make_grid(
        extent=scenario_fake.extent,
        speed_water=100000,
        ppw=2,
        center_frequency=5e5,
    )


def test_select_simulation_time_for_steady_state_with_defined_time_to_ss(grid_fake):
    """Verify that the returned simulation time is correct.

    When time_to_steady_state is provided, the function should use it to calculate the
    simulation time.
    """
    test_materials = {"brain": materials.get_material("brain")}
    test_freq = 2.0e4
    n_cycles = 5
    test_time_ss = 1e-4
    source_delay = 13.0
    sim_time = select_simulation_time_for_steady_state(
        grid=grid_fake,
        materials=test_materials,
        freq_hz=test_freq,
        time_to_steady_state=test_time_ss,
        n_cycles_steady_state=n_cycles,
        delay=source_delay,
    )
    period = 1.0 / test_freq
    expected_time = source_delay + test_time_ss + n_cycles * period
    np.testing.assert_allclose(sim_time, expected_time)


def test_select_simulation_time_for_steady_state_with_default_time_to_ss(grid_fake):
    """Verify that the returned simulation time is correct.

    When time_to_steady_state is not provided, the function should estimate a value
    for it based on the size of the scenario and the lowest speed of sound.
    """
    test_materials = {
        "fast": materials.get_material("cortical_bone"),
        "slow": materials.get_material("brain"),
    }
    test_freq = 2.0e4
    n_cycles = 5
    sim_time = select_simulation_time_for_steady_state(
        grid=grid_fake,
        materials=test_materials,
        freq_hz=test_freq,
        time_to_steady_state=None,
        n_cycles_steady_state=n_cycles,
        delay=0.0,
    )
    period = 1.0 / test_freq
    length = np.sqrt(0.3**2 + 0.4**2 + 0.5**2)
    expected_time_ss = 2 * length / materials.get_material("brain").vp
    expected_time = expected_time_ss + n_cycles * period
    np.testing.assert_allclose(sim_time, expected_time)


def test_select_simulation_time_for_pulsed(grid_fake):
    """Verify that the returned simulation time is correct for a pulsed simulation."""
    test_materials = {
        "fast": materials.get_material("cortical_bone"),
        "slow": materials.get_material("brain"),
    }
    source_delay = 31.0
    sim_time = select_simulation_time_for_pulsed(
        grid=grid_fake,
        materials=test_materials,
        delay=source_delay,
    )
    length = np.sqrt(0.3**2 + 0.4**2 + 0.5**2)
    expected_time = source_delay + length / materials.get_material("brain").vp
    np.testing.assert_allclose(sim_time, expected_time)


def test_create_time_grid():
    """Verify that the properties of the returned Time object are correct."""
    test_freq = 2.0e4
    ppp = 20
    time = create_time_grid(freq_hz=test_freq, ppp=ppp, sim_time=0.0024)
    np.testing.assert_allclose(time.start, 0.0)
    np.testing.assert_allclose(time.stop, 0.0024)
    period = 1 / test_freq
    np.testing.assert_allclose(time.step, period / ppp)


def test_find_largest_delay_in_sources():
    """Verify that delay is detected correctly from source delay only."""
    s1 = a_source_to_test(delay=0)
    s2 = a_source_to_test(delay=2)
    s3 = a_source_to_test(delay=3)
    assert np.isclose(find_largest_delay_in_sources([s1]), 0)
    assert find_largest_delay_in_sources([s1, s2, s3]) == 3.0


def test_find_largest_delay_in_sources_combines_with_tilt():
    """Verify that `.delay` is combined with delays coming from tilt."""
    source = a_source_to_test(delay=2.5, tilt_angle=30.0, pitch=1500.0)
    assert find_largest_delay_in_sources([source]) == 3.0


def test_find_largest_delay_in_sources_combines_with_focus():
    """Verify that `.delay` is combined with delays coming from focus."""
    source = a_source_to_test(delay=2.0, pitch=1.0, focal_length=1000.0, tilt_angle=30)
    assert np.isclose(find_largest_delay_in_sources([source]), 2.0 + 3.3333e-4)
